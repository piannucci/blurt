#include "util.h"

uint32_t rev(uint32_t x, size_t n) {
    uint32_t y = 0;
    for (size_t i=0; i<n; i++) {
        y <<= 1;
        y |= x&1;
        x >>= 1;
    }
    return y;
}

uint32_t mul(uint32_t a, uint32_t b) {
    uint32_t c = 0;
    uint32_t i = 0;
    while (b) {
        if (b&1)
            c ^= a << i;
        b >>= 1;
        i += 1;
    }
    return c;
}
