#include "scrambler.h"

Scrambler::Scrambler(int state) : state(state) {
}

int Scrambler::next() {
    state = (0xffff & (state << 1)) ^ (1 & ((state >> 3) ^ (state >> 6)));
    return state & 1;
}

void Scrambler::scramble(const bitvector &input, int multipleOf, bitvector &output, int scramblerState) {
    Scrambler sc(scramblerState);
    int pad_bits = 6;
    if (multipleOf != 0) {
        int stragglers = (input.size() + 6) % multipleOf;
        if (stragglers)
            pad_bits += multipleOf - stragglers;
    }
    output.assign(input.begin(), input.end());
    output.resize(input.size() + pad_bits);
    for (int i=0; i<output.size(); i++)
        output[i] = output[i] ^ sc.next();
    for (int i=input.size(); i<input.size()+6; i++)
        output[i] = 0;
}
