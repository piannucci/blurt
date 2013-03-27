#ifndef WAVE_H
#define WAVE_H
#include "blurt.h"

void readwave(const std::string &filename, std::vector<float> &output, float &Fs);
void writewave(const std::string &filename, const std::vector<float> &input, int Fs, int bytesPerSample, int nchannels);

#endif
