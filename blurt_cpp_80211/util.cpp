#include "util.h"

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
