#ifndef audioloopback_h
#define audioloopback_h

#include "blurt.h"
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "iir.h"
#include "wifi80211.h"
#include "portaudio.h"
#include "pa_ringbuffer.h"
#include <cassert>

struct stereo {
    float l, r;
    stereo() : l(0), r(0) {}
    stereo(float lr) : l(lr), r(lr) {}
    stereo(float l_, float r_) : l(l_), r(r_) {}
    const stereo & operator += (const stereo & b) { l+=b.l; r+=b.r; return *this; }
    operator float() const { return .5f*(l+r); }
};

inline stereo operator *(const stereo & a, const float & b) { return stereo(a.l*b, a.r*b); }
inline stereo operator *(const float & b, const stereo & a) { return stereo(a.l*b, a.r*b); }

using std::vector;
using std::string;
using std::endl;
using std::mutex;
using std::condition_variable;
using std::deque;
using std::thread;
using std::min;
using std::unique_lock;
using std::cerr;
using std::cout;

extern mutex cerr_mutex;

template <class S>
struct lock_ios {
    S & s;
    unique_lock<mutex> l = unique_lock<mutex>(cerr_mutex);
    lock_ios(S & s_) : s(s_) {}
    template <typename U>
    lock_ios & operator<<(U && u) { s << u; return *this; }

    typedef std::ostream& (*ostream_manipulator)(std::ostream&);
    lock_ios & operator<<(ostream_manipulator manip) { s << manip; return *this; }
};

template <class S>
lock_ios<S> lock_io(S & s) { return lock_ios<S>(s); }

void processOutput(const vector<fcomplex> &input,
                   double Fs,
                   double Fc,
                   size_t upsample_factor,
                   const vector<float> &mask_noise,
                   vector<stereo> &output);

void processInput(const vector<stereo> &input,
                  double Fs,
                  double Fc,
                  size_t upsample_factor,
                  vector<fcomplex> &output);

struct frame {
    string contents;
    size_t rate;
    double gain_dB;
};

class packetProducer {
    public:
        const size_t length = 16;
        size_t i = 0;
        size_t N_per_step = 100;
        size_t steps = 8 * 9;
        vector<size_t> permutation = vector<size_t>(N_per_step * steps);
        packetProducer();
        frame nextPacket();
};

template <class T>
struct circularBuffer {
    size_t maximum = 16384;
    size_t read_idx = 0;
    size_t write_idx = 0;
    size_t length = 0;
    vector<T> buffer = vector<T>(maximum);

    void push_back(const vector<T> & input) {
        size_t N = min(input.size(), maximum - length);
        if (N < input.size())
            lock_io(cerr) << "AudioBuffer overrun" << endl;
        if (N) {
            size_t M = min(N, maximum - write_idx);
            if (M)
                copy(input.begin(), input.begin() + (ssize_t)M, buffer.begin() + (ssize_t)write_idx);
            if (N-M)
                copy(input.begin() + (ssize_t)M, input.begin() + (ssize_t)N, buffer.begin());
            write_idx = (write_idx + N) % maximum;
            length += N;
        }
    }

    void peek(size_t count, vector<T> & output) {
        size_t N = min(length, count);
        output.resize(N);
        if (N) {
            size_t M = min(N, maximum - read_idx);
            if (M)
                copy(buffer.begin() + (ssize_t)read_idx, buffer.begin() + (ssize_t)(read_idx + M), output.begin());
            if (N-M)
                copy(buffer.begin(), buffer.begin() + (ssize_t)(N - M), output.begin() + (ssize_t)M);
        }
    }

    void pop(size_t count) {
        size_t N = min(length, count);
        read_idx = (read_idx + N) % maximum;
        length -= N;
    }

    circularBuffer(size_t maximum_) : maximum(maximum_) {}
};

template <class T>
class lockFreeRingBuffer {
    int cap;
    T * buf = new T[cap];
    PaUtilRingBuffer r;
public:
    lockFreeRingBuffer(int cap_=256) : cap(cap_) {
        PaUtil_InitializeRingBuffer(&r, sizeof(T), cap, buf);
    }
    ~lockFreeRingBuffer() { delete [] buf; }
    ssize_t read_available() { return PaUtil_GetRingBufferReadAvailable(&r); }
    ssize_t write_available() { return PaUtil_GetRingBufferWriteAvailable(&r); }
    bool put(const T & t) { return PaUtil_WriteRingBuffer(&r, &t, 1); }
    bool get(T & t) { return PaUtil_ReadRingBuffer(&r, &t, 1); }
};

class packetTransmitter {
    public:
        const int channels = 2;
        double Fs, Fc;
        size_t upsample_factor;
        double cutoff;
        IIRFilter<stereo> *hp;
        const WiFi80211 & wifi;
        vector<float> mask_noise;
        packetTransmitter(double Fs, double Fc, size_t upsample_factor, const WiFi80211 & wifi);
        ~packetTransmitter();
        void encode(const frame & f, vector<stereo> & output);
};


class audioFIFO {
private:
    static int myCallback(const void *inputBuffer, void *outputBuffer, size_t
        framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags, void *userData);
    int callback( const void *inputBuffer, void *outputBuffer, size_t
        framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags);

    mutex send_mutex, recv_mutex;
    condition_variable send_condition, recv_condition;

    deque<frame> output_frames;
    deque<DecodeResult> input_frames;
    lockFreeRingBuffer<vector<stereo> *> output_audio, output_audio_finished;
    lockFreeRingBuffer<vector<stereo> *> input_audio, input_audio_ready;

    double Fs, Fc;
    size_t upsample_factor;
    const size_t framesPerBuffer = 512;
    const WiFi80211 & wifi;
    packetTransmitter transmitter = packetTransmitter(Fs, Fc, upsample_factor, wifi);
    thread nanny_thread, encoder_thread, decoder_thread;
    size_t decoder_carrier_phase = 0, decoder_downsample_phase = 0;
    IIRFilter<fcomplex> *decoder_lowpass_filter;

    circularBuffer<fcomplex> baseband_signal = circularBuffer<fcomplex>((size_t)(Fs * 10));

    const size_t trigger = 4096;

    bool shutdown = false;

    PaStream *stream;

    void nanny_thread_func();
    void encoder_thread_func();
    size_t try_decode();
    void decoder_thread_func();

    size_t current_output_index, current_input_index;
    vector<stereo> *current_output_buffer = nullptr, *current_input_buffer = nullptr;
    int callbacks = 0;

public:
    audioFIFO(double Fs, double Fc, size_t upsample_factor, const WiFi80211 & wifi);
    void push_back(const frame & f);
    void pop_front(DecodeResult & f);
    ~audioFIFO();
};

#endif
