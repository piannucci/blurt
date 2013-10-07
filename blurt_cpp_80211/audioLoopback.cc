#include "blurt.h"
#include "upsample.h"
#include "util.h"
#include <algorithm>
#include <numeric>
#include <random>
#include "audioLoopback.h"
#include <chrono>
#include <cstring>

static const double delay = .005;
mutex cerr_mutex;

using std::random_device;
using std::mt19937;
using std::uniform_int_distribution;
using std::to_string;
using std::once_flag;
using std::call_once;
namespace chrono {
    using namespace std::chrono;
}
namespace this_thread {
    using namespace std::this_thread;
}

static random_device rd;
static mt19937 engine(rd());

template <class T>
void die_vector(const vector<T> & output) {
    auto l = lock_io(cout);
    for (int i=0; i<output.size(); i++)
        l << output[i] << " ";
    l << endl;
    abort();
}

void processOutput(const vector<fcomplex> &input,
                   double Fs,
                   double Fc,
                   size_t upsample_factor,
                   const vector<float> &mask_noise,
                   vector<stereo> &output) {
    vector<fcomplex> upsampled_output;
    upsample(input, upsample_factor, upsampled_output);

    vector<float> modulated_output(upsampled_output.size());
    for (size_t i=0; i<upsampled_output.size(); i++)
        modulated_output[i] = real(upsampled_output[i] * expj(float((2*pi*Fc*i)/Fs)));

    vector<float> amplitude(modulated_output.size());
    for (size_t i=0; i<modulated_output.size(); i++)
        amplitude[i] = fabsf(modulated_output[i]);

    size_t idx = size_t(amplitude.size() * .95f);
    nth_element(amplitude.begin(), amplitude.begin() + (ssize_t)idx, amplitude.end());
    float percentile_95_amplitude = amplitude[idx];

    vector<float> scaled_output(modulated_output.size());
    for (size_t i=0; i<modulated_output.size(); i++)
        scaled_output[i] = modulated_output[i] * 0.5f / percentile_95_amplitude;

    // output = np.r_[output, np.zeros(int(.1*loopback_Fs))]

    if (mask_noise.size()) {
        if (mask_noise.size() > scaled_output.size())
            scaled_output.resize(mask_noise.size());
        for (size_t i=0; i<mask_noise.size(); i++)
            scaled_output[i] += mask_noise[i];
    }

    // delay one channel slightly relative to the other:
    // this breaks up the spatially-dependent frequency-correlated
    // nulls of our speaker array
    size_t delay_samples = size_t(delay * Fs);
    output.resize(delay_samples + scaled_output.size());
    for (size_t i=0; i<scaled_output.size() + delay_samples; i++) {
        output[i] = stereo{i < delay_samples ? 0 : scaled_output[i-delay_samples],
                           i < scaled_output.size() ? scaled_output[i] : 0};
    }
}

void processInput(const vector<stereo> &input,
                  double Fs,
                  double Fc,
                  size_t upsample_factor,
                  vector<fcomplex> &output) {
    vector<fcomplex> baseband_signal(input.size());
    for (size_t i=0; i<input.size(); i++)
        baseband_signal[i] = fcomplex(.5f * (input[i].l + input[i].r), 0) * expj(float(-(2*pi*Fc*i)/Fs));
    size_t order = 6;
    string order_str = to_string(order);
    string cutoff_str = to_string(.8 / upsample_factor);
    const char *args[] = {"", "-Bu", "-Lp", "-o", order_str.c_str(), "-a", cutoff_str.c_str(), 0};
    IIRFilter<fcomplex> lp(args);
    vector<fcomplex> filtered_signal;
    lp.apply(baseband_signal, filtered_signal);
    output.resize(size_t(filtered_signal.size()/(float)upsample_factor));
    for (size_t i=0; i<output.size(); i++)
        output[i] = filtered_signal[i*upsample_factor];
}

packetProducer::packetProducer() {
    iota(permutation.begin(), permutation.end(), 0);
    random_shuffle(permutation.begin(), permutation.end());
}

frame packetProducer::nextPacket() {
    string octets(length, 0);
    auto u = uniform_int_distribution<>('A', 'Z');
    auto uniform = bind(u, engine);
    generate(octets.begin(), octets.end(), uniform);
    size_t step = permutation[i] / N_per_step;
    if (step >= steps)
        return {"",0,0};
    snprintf(&octets[0], 9, "%06zds%02zd", i, step);
    double level = (step / 8) * 2.5 - 5.0;
    size_t rate = step % 8;
    if (i % N_per_step == 0) {
        double pct = double(i) / (N_per_step * steps) * 100.;
        lock_io(cerr) << i << "/" << N_per_step * steps << " (" << pct << "%)";
    }
    i++;
    return {octets, rate, level};
}

packetTransmitter::packetTransmitter(double Fs_, double Fc_, size_t upsample_factor_, const WiFi80211 & wifi_)
    : Fs(Fs_), Fc(Fc_), upsample_factor(upsample_factor_), wifi(wifi_) {
    cutoff = Fc - Fs/upsample_factor;
    const size_t order = 6;
    string order_str = to_string(order);
    string cutoff_str = to_string(cutoff/Fs);
    const char *args[] = {"", "-Bu", "-Hp", "-o", order_str.c_str(), "-a", cutoff_str.c_str(), 0};
    hp = new IIRFilter<stereo>(args);
}

packetTransmitter::~packetTransmitter() {
    delete hp;
}

void packetTransmitter::encode(const frame & f, vector<stereo> & output) {
    vector<fcomplex> encoded_output;
    vector<uint8_t> octets_vector(f.contents.cbegin(), f.contents.cend());
    wifi.encode(octets_vector, f.rate, encoded_output);
    vector<stereo> acoustic_output;
    processOutput(encoded_output, Fs, Fc, upsample_factor, mask_noise, acoustic_output);
    double gain = pow(10., (f.gain_dB - 15.0) / 20.);
    for (size_t i=0; i<acoustic_output.size(); i++) {
        acoustic_output[i].l *= gain;
        acoustic_output[i].r *= gain;
    }
    hp->apply(acoustic_output, output);
}

static once_flag PaInit;
static void cleanupPortAudio() { Pa_Terminate(); }
static void initPortAudio() { Pa_Initialize(); atexit(cleanupPortAudio); }

int audioFIFO::myCallback( const void *inputBuffer, void *outputBuffer, size_t
    framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags, void *userData ) {
    audioFIFO *af = (audioFIFO *)userData;
    return af->callback(inputBuffer, outputBuffer, framesPerBuffer, timeInfo, statusFlags);
}

int audioFIFO::callback(const void *inputBuffer, void *outputBuffer, size_t
    framesPerBuffer_, const PaStreamCallbackTimeInfo*,
    PaStreamCallbackFlags) {

    callbacks++;
    const stereo *in  = (const stereo *)inputBuffer;
    stereo *out = (stereo *)outputBuffer;    

    size_t outputFrames = framesPerBuffer_;
    while (outputFrames) {
        if (!current_output_buffer && output_audio.read_available()) {
            output_audio.get(current_output_buffer);
            current_output_index = 0;
        }

        size_t n = outputFrames;
        if (current_output_buffer)
            n = min(n, current_output_buffer->size() - current_output_index);

        if (current_output_buffer) {
            memcpy(out, &(*current_output_buffer)[current_output_index], n * sizeof(stereo));
            //lock_io(cerr) << ".";
        }
        else {
            stereo zero = 0;
            for (size_t i=0; i<n; i++) {
                out[i] = zero;
                //lock_io(cerr) << (i + (out - (stereo *)outputBuffer)) << "/" << framesPerBuffer << endl;
            }
        }

        current_output_index += n;
        outputFrames -= n;
        out += n;

        if (current_output_buffer && current_output_buffer->size() == current_output_index) {
            output_audio_finished.put(current_output_buffer);
            current_output_buffer = nullptr;
        }
    }

    size_t inputFrames = framesPerBuffer_;
    while (inputFrames) {
        if (!current_input_buffer && input_audio_ready.read_available()) {
            input_audio_ready.get(current_input_buffer);
            current_input_index = 0;
        }

        size_t n = inputFrames;
        if (current_input_buffer)
            n = min(n, current_input_buffer->size() - current_input_index);

        if (current_input_buffer) {
            memcpy(&(*current_input_buffer)[current_input_index], in, n * sizeof(stereo));
            //lock_io(cerr) << ",";
        }

        current_input_index += n;
        inputFrames -= n;
        in += n;

        if (current_input_buffer && current_input_buffer->size() == current_input_index) {
            input_audio.put(current_input_buffer);
            current_input_buffer = nullptr;
        }
    }

    return 0;
}

void audioFIFO::nanny_thread_func() {
    lock_io(cerr) << "nanny active" << endl;
    while (!shutdown) {
        while (output_audio_finished.read_available()) {
            vector<stereo> * output;
            if (output_audio_finished.get(output))
                delete output;
        }
        while (input_audio_ready.write_available()) {
            vector<stereo> * output = new vector<stereo>(framesPerBuffer);
            assert(output->size() == framesPerBuffer && framesPerBuffer != 0);
            if (!input_audio_ready.put(output))
                delete output;
        }
        this_thread::sleep_for(chrono::milliseconds(100));
        //lock_io(cerr) << callbacks << " " << baseband_signal.length << endl;
    }
    lock_io(cerr) << "nanny finished" << endl;
}

void audioFIFO::encoder_thread_func() {
    lock_io(cerr) << "encoder active" << endl;
    unique_lock<mutex> sl(send_mutex);
    while (!shutdown) {
        while (output_frames.size()) {
            const frame & output_frame = output_frames.front();
            sl.unlock();
            vector<stereo> *output = new vector<stereo>;
            transmitter.encode(output_frame, *output);
            sl.lock();
            output->resize(output->size() + size_t(Fs*.1));
            output_frames.pop_front();
            if (!output_audio.put(output))
                delete output; // no room in queue
        }
        send_condition.wait(sl);
    }
    lock_io(cerr) << "encoder finished" << endl;
}

size_t audioFIFO::try_decode() {
    vector<fcomplex> input;
    baseband_signal.peek(baseband_signal.maximum, input);
    size_t endIndex = 0;
    try {
        vector<DecodeResult> results;
        wifi.decode(input, results);
        if (results.size()) {
            auto l = lock_io(cerr) ;
            l << string(results.size(), '>');
            endIndex = results.back().endIndex;
            unique_lock<mutex> rl(recv_mutex);
            for (auto result : results) {
                string s = string(result.payload.begin(), result.payload.end());
                l << "\"" << s << "\" ";
                input_frames.push_back(result);
                recv_condition.notify_one();
            }
        }
    } catch (...) {
        lock_io(cerr) << "exception in try_decode" << endl;
    }
    if (endIndex)
        return endIndex;
    else
        return trigger/2;
}

void audioFIFO::decoder_thread_func() {
    lock_io(cerr) << "decoder active" << endl;
    while (!shutdown) {
        frame f;
        while (input_audio.read_available()) {
            // convert to complex baseband signal
            vector<stereo> * input;
            if (!input_audio.get(input))
                continue;

            const double s = -2 * M_PI * Fc / Fs;
            vector<fcomplex> baseband_input(input->size());
            for (size_t i=0; i<input->size(); i++)
                baseband_input[i] = (float)(*input)[i] * expj(float(s * (i + decoder_carrier_phase)));
            decoder_carrier_phase += input->size();

            // filter
            vector<fcomplex> baseband_filtered;
            decoder_lowpass_filter->apply(baseband_input, baseband_filtered);

            // downsample
            size_t first_sample = decoder_downsample_phase;
            size_t end_sample = ((baseband_filtered.size() - first_sample + upsample_factor - 1) / upsample_factor) * upsample_factor;
            decoder_downsample_phase = (end_sample - baseband_filtered.size()) % upsample_factor;
            vector<fcomplex> downsampled((end_sample-first_sample) / upsample_factor);
            for (size_t i=0,j=first_sample; i<downsampled.size(); i++, j+=upsample_factor)
                downsampled[i] = baseband_filtered[j];

            baseband_signal.push_back(downsampled);
        }

        // see if there's anything we can decode
        while (baseband_signal.length >= trigger) {
            //lock_io(cerr) << "trigger" << endl;
            baseband_signal.pop(try_decode());
        }

        this_thread::sleep_for(chrono::milliseconds(100));
    }
    lock_io(cerr) << "decoder finished" << endl;
}

audioFIFO::audioFIFO(double Fs_, double Fc_, size_t upsample_factor_, const WiFi80211 & wifi_)
    : Fs(Fs_), Fc(Fc_), upsample_factor(upsample_factor_), wifi(wifi_) {
    nanny_thread = thread(&audioFIFO::nanny_thread_func, this);
    encoder_thread = thread(&audioFIFO::encoder_thread_func, this);
    decoder_thread = thread(&audioFIFO::decoder_thread_func, this);

    string cutoff_str = to_string(.8/upsample_factor);
    const char *args[] = {"", "-Bu", "-Lp", "-o", "6", "-a", cutoff_str.c_str(), 0};
    decoder_lowpass_filter = new IIRFilter<fcomplex>(args);

    call_once(PaInit, initPortAudio);
    Pa_OpenDefaultStream(&stream, 2, 2, paFloat32, Fs, framesPerBuffer, myCallback, this);
    Pa_StartStream(stream);
}

void audioFIFO::push_back(const frame & f) {
    lock_io(cerr) << "<";
    unique_lock<mutex> sl(send_mutex);
    output_frames.push_back(f);
    send_condition.notify_one();
}

void audioFIFO::pop_front(DecodeResult & r) {
    lock_io(cerr) << "fetching received frame" << endl;
    unique_lock<mutex> rl(recv_mutex);
    while (input_frames.size() == 0)
        recv_condition.wait(rl);
    r = input_frames.back();
    input_frames.pop_back();
}

audioFIFO::~audioFIFO() {
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    shutdown = true;
    {
        unique_lock<mutex> sl(send_mutex);
        send_condition.notify_one();
    }
    encoder_thread.join();
    decoder_thread.join();
    nanny_thread.join();
    vector<stereo> * x;
    while (input_audio.get(x)) delete x;
    while (input_audio_ready.get(x)) delete x;
    while (output_audio.get(x)) delete x;
    while (output_audio_finished.get(x)) delete x;
    delete decoder_lowpass_filter;
}
