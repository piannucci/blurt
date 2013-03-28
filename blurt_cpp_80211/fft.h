#ifndef FFT_H
#define FFT_H
#include "blurt.h"

void fft(complex *x, size_t n);
void ifft(complex *x, size_t n);
#endif
