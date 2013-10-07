#ifndef CRC_H
#define CRC_H
#include "blurt.h"

class CRC {
private:
    size_t L, s, M;
    uint32_t m;
    std::vector<uint32_t> lut;
    bitvector b;
    bitvector correct_remainder;
    void remainder_slow(const bitvector &a, bitvector &output);
    uint32_t remainder_fast(const bitvector &a, bitvector &output) const;
    void lut_bootstrap(const bitvector &new_b, size_t new_L);
public:
    CRC();
    void FCS(const bitvector &calculationFields, bitvector &output) const;
    bool checkFCS(const bitvector &frame) const;
};
#endif
