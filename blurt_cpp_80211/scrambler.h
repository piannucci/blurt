#ifndef SCRAMBLER_H
#define SCRAMBLER_H
#include "blurt.h"

class Scrambler {
private:
    uint32_t state;
public:
    Scrambler(uint32_t state);
    int next();
    static void scramble(const bitvector &input, size_t multipleOf, bitvector &output, uint32_t scramblerState=0x7f);
};
#endif
