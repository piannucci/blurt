#include "util.h"

int rev(int x, int n) {
    int y = 0;
    for (int i=0; i<n; i++) {
        y <<= 1;
        y |= x&1;
        x >>= 1;
    }
    return y;
}

int mul(int a, int b) {
    int c = 0;
    int i = 0;
    while (b) {
        if (b&1)
            c ^= a << i;
        b >>= 1;
        i += 1;
    }
    return c;
}

void shiftin(const bitvector &input, int noutput, std::vector<int> &output) {
    output.resize(input.size()/noutput);
    for (int i=0; i<output.size(); i++) {
        int x = 0;
        for (int j=0; j<noutput; j++)
            x += input[i*noutput+j] << j;
        output[i] = x;
    }
}

void shiftout(const std::vector<int> &input, int ninput, bitvector &output) {
    output.resize(ninput * input.size());
    for (int i=0; i<input.size(); i++)
        for (int j=0; j<ninput; j++)
            output[i*ninput+j] = (input[i] >> j) & 1;
}
