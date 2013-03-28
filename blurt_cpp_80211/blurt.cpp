#include "blurt.h"
#include "wave.h"
#include "wifi80211.h"
#include <string>
#include <iostream>

//WiFi80211 wifi;

int Fs = 48000;
float Fc = 19000;
float upsample_factor = 16;
std::vector<float> mask_noise;
int rate = 0;
int length = 16;

int main(int argc, char **argp, char **envp) {
    std::string infilename = "../blurt_py_80211/35631__reinsamba__crystal-glass.wav";
    std::string outfilename = "../blurt_py_80211/test.wav";
    std::vector<float> samples;
    float Fs;
    readwave(infilename, samples, Fs);
    float scale = pow(2, -16);
    //for (int i=samples.size()-100; i<samples.size(); i++)
    //    std::cout << samples[i] << " ";
    for (int i=0; i<samples.size(); i++)
        samples[i] *= scale;
    std::cout << std::endl;
    writewave(outfilename, samples, Fs, 2, 1);
    return 0;
}
