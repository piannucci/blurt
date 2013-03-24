#include "blurt.h"

class PuncturingMask {
private:
    std::vector<bool> mask;
public:
    int numerator, denominator;
    PuncturingMask(int numerator, int denominator);
    void puncture(const std::vector<bool> &input, std::vector<bool> &output);
    void depuncture(const std::vector<int> &input, std::vector<int> &output);
};
