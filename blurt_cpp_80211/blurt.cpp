#include "blurt.h"
#include "wave.h"
#include "upsample.h"
#include "wifi80211.h"
#include <string>
#include <cstdlib>
#include <ctime>

#include "ofdm.h"
#include "crc.h"
#include "util.h"
#include "iir.h"

int Fs = 48000;
float Fc = 19000;
float upsample_factor = 16;
std::vector<float> mask_noise;
int rate = 0;
int length = 16;

WiFi80211 wifi;

void tx(const std::vector<int> &input_octets, std::vector<float> &output) {
    std::vector<complex> modulated_signal;
    wifi.encode(input_octets, 0, modulated_signal);
    std::vector<complex> upsampled_output;
    upsample(modulated_signal, upsample_factor, upsampled_output);
    output.resize(upsampled_output.size());
    for (int i=0; i<upsampled_output.size(); i++)
        output[i] = real(upsampled_output[i] * exp(complex(0, (2*pi*Fc*i)/Fs)))*1e-1;
}

void rx(const std::vector<float> &input, std::vector<DecodeResult> &output) {
    std::vector<complex> baseband_signal(input.size());
    for (int i=0; i<input.size(); i++)
        baseband_signal[i] = input[i] * exp(complex(0, -(2*pi*Fc*i)/Fs));
    std::ostringstream order, freq;
    order << 6;
    freq << (.8 / upsample_factor);
    const char *args[] = {"", "-Bu", "-Lp", "-o", order.str().c_str(), "-a", freq.str().c_str(), 0};
    IIRFilter<complex> lp(args);
    std::vector<complex> filtered_signal;
    lp.apply(baseband_signal, filtered_signal);
    std::vector<complex> downsampled_signal(int(filtered_signal.size()/upsample_factor));
    for (int i=0; i<downsampled_signal.size(); i++)
        downsampled_signal[i] = filtered_signal[int(i*upsample_factor)];
    wifi.decode(downsampled_signal, output);
}

int main(int argc, char **argp, char **envp) {
    srand(time(NULL));
    std::vector<int> input(1500);
    std::vector<float> output;
    int trials = 20;
    for (int i=0; i<input.size(); i++)
        input[i] = rand() % 256;
    clock_t t0 = clock();
    for (int i=0; i<trials; i++) {
        tx(input, output);
    }
    clock_t t1 = clock();
    double per_trial = (double(t1 - t0)/CLOCKS_PER_SEC/trials);
    int samples_encoded = output.size();
    std::cout << (samples_encoded / per_trial) << std::endl;
    t0 = clock();
    std::vector<DecodeResult> decoded;
    for (int i=0; i<trials; i++)
        rx(output, decoded);
    t1 = clock();
    per_trial = (double(t1 - t0)/CLOCKS_PER_SEC/trials);
    std::cout << (samples_encoded / per_trial) << std::endl;
    //writewave("test.wav", output, Fs, 3, 1);
    //std::cout << decoded.size() << std::endl;
    //for (int i=0; i<decoded.size(); i++) {
    //    std::cout << decoded[i].lsnr << std::endl;
    //    std::vector<int> &payload = decoded[i].payload;
    //    std::cout << "'";
    //    for (int j=0; j<payload.size(); j++)
    //        std::cout << (char)payload[j];
    //    std::cout << "'" << std::endl;
    //}
    return 0;
}
