#include "blurt.h"

int rev(int x, int n);
int mul(int a, int b);
void shiftin(const bitvector &input, int noutput, std::vector<int> &output);
void shiftout(const std::vector<int> &input, int ninput, bitvector &output);

static inline complex expj(float theta) {
    return complex(cos(theta), sin(theta));
}
