#include "blurt.h"

uint32_t rev(uint32_t x, size_t n);
uint32_t mul(uint32_t a, uint32_t b);

static inline complex expj(float theta) {
    return complex(cosf(theta), sinf(theta));
}

template <class T>
static inline void shiftin(const bitvector &input, size_t noutput, std::vector<T> &output) {
    output.resize(input.size()/noutput);
    for (size_t i=0; i<output.size(); i++) {
        T x = 0;
        for (size_t j=0; j<noutput; j++)
            x += (T)input[i*noutput+j] << j;
        output[i] = x;
    }
}

template <class T>
static inline void shiftout(const std::vector<T> &input, size_t ninput, bitvector &output) {
    output.resize(ninput * input.size());
    for (size_t i=0; i<input.size(); i++)
        for (size_t j=0; j<ninput; j++)
            output[i*ninput+j] = (input[i] >> j) & 1;
}
