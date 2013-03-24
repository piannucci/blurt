#include "blurt.h"
#include <stdint.h>

class ConvolutionalCode {
private:
    int nu, g0, g1;
    std::vector<bool> output_map_1, output_map_2;
    std::vector<int> state_inv_map[2], output_map_1_soft, output_map_2_soft;
public:
    ConvolutionalCode(int nu, int g0, int g1);
    void encode(const std::vector<bool> &input, std::vector<bool> &output);
    void decode(const std::vector<int> &input, int length, std::vector<bool> &output);
};
