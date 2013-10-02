#import <Foundation/Foundation.h>

#include "blurt.h"
#include "wave.h"
#include "upsample.h"
#include "wifi80211.h"
#include <string>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include "ofdm.h"
#include "crc.h"
#include "util.h"
#include "iir.h"

#include "dsp.h"

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

// produces 16-bit mono lPCM @ 48000 Hz
NSData *encodeBlurtWaveform(NSData *inputData, int rate) {
    std::vector<int> input((uint8_t *)inputData.bytes, (uint8_t *)inputData.bytes + inputData.length);
    std::vector<float> output;
    std::vector<complex> iq;
    
    wifi.encode(input, rate, iq);
    processOutput(iq, output);
    
    std::vector<float> mag(output);
    for (int i=0; i<mag.size(); i++)
        mag[i] = fabsf(mag[i]);
    
    int n = mag.size()*98/100;
    std::nth_element(mag.begin(), mag.begin() + n, mag.end());
    float scale = 0x7fff / mag[n] * .5f;
    
    int16_t *samples = (int16_t *)malloc(output.size() * sizeof(uint16_t));
    for (int i=0; i<output.size(); i++)
        samples[i] = output[i] * scale;
    
    return [NSData dataWithBytesNoCopy:samples length:output.size() * sizeof(uint16_t) freeWhenDone:YES];
}