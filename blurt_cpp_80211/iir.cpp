#include "iir.h"

float bar_a[1] = {};
IIRFilter<complex> foo(0, bar_a, NULL, 0.f);
IIRFilter<float> bar(0, bar_a, NULL, 0.f);
