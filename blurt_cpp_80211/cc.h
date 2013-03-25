#ifndef CC_H
#define CC_H
#include "blurt.h"
#include <stdint.h>

class ConvolutionalCode {
private:
    int nu, g0, g1;
    bitvector output_map_1, output_map_2;
    std::vector<int> state_inv_map[2], output_map_1_soft, output_map_2_soft;
public:
    ConvolutionalCode(int nu, int g0, int g1);
    void encode(const bitvector &input, bitvector &output);
    void decode(const std::vector<int> &input, int length, bitvector &output);
};
#endif
