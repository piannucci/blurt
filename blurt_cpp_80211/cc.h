#ifndef CC_H
#define CC_H
#include "blurt.h"
#include <stdint.h>

class ConvolutionalCode {
private:
    size_t nu;
    bitvector output_map_1, output_map_2;
    std::vector<size_t> state_inv_map[2];
    std::vector<int> output_map_1_soft, output_map_2_soft;
public:
    ConvolutionalCode(size_t nu, uint32_t g0, uint32_t g1);
    void encode(const bitvector &input, bitvector &output) const;
    void decode(const std::vector<int> &input, size_t length, bitvector &output) const;
};
#endif
