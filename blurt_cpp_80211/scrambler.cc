#include "scrambler.h"

Scrambler::Scrambler(uint32_t state_) : state(state_) {
}

int Scrambler::next() {
    state = (0xffff & (state << 1)) ^ (1 & ((state >> 3) ^ (state >> 6)));
    return state & 1;
}

void Scrambler::scramble(const bitvector &input, size_t multipleOf, bitvector &output, uint32_t scramblerState) {
    Scrambler sc(scramblerState);
    size_t pad_bits = 6;
    if (multipleOf != 0) {
        size_t stragglers = (input.size() + 6) % multipleOf;
        if (stragglers)
            pad_bits += multipleOf - stragglers;
    }
    output.assign(input.begin(), input.end());
    output.resize(input.size() + pad_bits);
    for (size_t i=0; i<output.size(); i++)
        output[i] = (uint8_t)(output[i] ^ sc.next());
    for (size_t i=input.size(); i<input.size()+6; i++)
        output[i] = 0;
}
