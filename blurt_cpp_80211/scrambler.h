#include "blurt.h"

class Scrambler {
private:
    int state;
public:
    Scrambler(int state);
    int next();
    static void scramble(const bitvector &input, int multipleOf, bitvector &output, int scramblerState=0x7f);
};
