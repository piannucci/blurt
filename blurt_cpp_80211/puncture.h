#include "blurt.h"

class PuncturingMask {
private:
    std::vector<bool> mask;
public:
    size_t numerator, denominator;
    PuncturingMask(size_t numerator, size_t denominator);
    void puncture(const bitvector &input, bitvector &output) const;
    void depuncture(const std::vector<int> &input, std::vector<int> &output) const;
};
