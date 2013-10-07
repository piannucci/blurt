#ifndef WAVE_H
#define WAVE_H
#include "blurt.h"

bool readwave(const std::string &filename, std::vector<float> &output, float &Fs);
bool writewave(const std::string &filename, const std::vector<float> &input, size_t Fs, size_t bytesPerSample, size_t nchannels);

#endif
