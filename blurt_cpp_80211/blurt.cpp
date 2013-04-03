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

void processOutput(const std::vector<complex> &input, std::vector<float> &output) {
    std::vector<complex> upsampled_output;
    upsample(input, upsample_factor, upsampled_output);
    output.resize(upsampled_output.size());
    for (int i=0; i<upsampled_output.size(); i++)
        output[i] = real(upsampled_output[i] * exp(complex(0, (2*pi*Fc*i)/Fs)))*1e-1;
}

void processInput(const std::vector<float> &input, std::vector<complex> &output) {
    std::vector<complex> baseband_signal(input.size());
    for (int i=0; i<input.size(); i++)
        baseband_signal[i] = input[i] * exp(complex(0, -(2*pi*Fc*i)/Fs));
    std::ostringstream order, freq;
    order << 6;
    freq << (.8 / upsample_factor);
    std::string order_str(order.str()), freq_str(freq.str());
    const char *args[] = {"", "-Bu", "-Lp", "-o", order_str.c_str(), "-a", freq_str.c_str(), 0};
    IIRFilter<complex> lp(args);
    std::vector<complex> filtered_signal;
    lp.apply(baseband_signal, filtered_signal);
    output.resize(int(filtered_signal.size()/upsample_factor));
    for (int i=0; i<output.size(); i++)
        output[i] = filtered_signal[int(i*upsample_factor)];
}

int main(int argc, char **argp, char **envp) {
    try {
        srand(time(NULL));
        std::vector<int> input(1500);
        std::vector<complex> output;
        int trials = 20;
        for (int i=0; i<input.size(); i++)
            input[i] = rand() % 256;
        clock_t t0 = clock();
        for (int i=0; i<trials; i++) {
            wifi.encode(input, 0, output);
        }
        clock_t t1 = clock();
        double per_trial = (double(t1 - t0)/CLOCKS_PER_SEC/trials);
        int samples_encoded = output.size();
        std::cout << (samples_encoded / per_trial) << std::endl;
        std::vector<DecodeResult> decoded;
        t0 = clock();
        for (int i=0; i<trials; i++) {
            wifi.decode(output, decoded);
        }
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
    } catch (const char *s) {
        std::cerr << s << std::endl;
    }
    return 0;
}
