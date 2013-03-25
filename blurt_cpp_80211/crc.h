#ifndef CRC_H
#define CRC_H
#include "blurt.h"

class CRC {
private:
    int L, s, M;
    uint32_t m;
    std::vector<uint32_t> lut;
    bitvector b;
    bitvector correct_remainder;
    void remainder_slow(const bitvector &a, bitvector &output);
    uint32_t remainder_fast(const bitvector &a, bitvector &output);
    void lut_bootstrap(const bitvector &new_b, int new_L);
public:
    CRC();
    void FCS(const bitvector &calculationFields, bitvector &output);
    bool checkFCS(const bitvector &frame);
};
#endif
