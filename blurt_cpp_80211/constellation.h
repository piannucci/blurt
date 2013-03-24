#include "blurt.h"
#include <vector>

class Constellation {
protected:
    Constellation(int Nbpsc);
public:
    int Nbpsc;
    std::vector<complex> symbols;
    void map(std::vector<bool> &input, std::vector<complex> &output);
    void demap(std::vector<complex> &input, float dispersion, std::vector<int64_t> &output);
};
