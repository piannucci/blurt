#include "iir.h"

static float bar_a[1] = {};
static IIRFilter<complex> foo(0, bar_a, NULL, 0.f);
static IIRFilter<float> bar(0, bar_a, NULL, 0.f);
