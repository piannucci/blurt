#ifndef BLURT_H
#define BLURT_H
#include <complex>
#include <vector>
#include <string>
#include <stdint.h>
#include <iostream>
#include <cmath>

#define pi 3.141592653589793238462f
#define dpi 3.141592653589793238462

typedef std::vector<uint8_t> bitvector;
typedef std::complex<float> complex;
typedef complex fcomplex;
typedef std::complex<double> dcomplex;

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#if __has_attribute(noreturn) // || defined(__GNUC__)
#define NORETURN [[noreturn]]
//#define NORETURN __attribute__((noreturn))
#else
#define NORETURN
#endif

#endif
