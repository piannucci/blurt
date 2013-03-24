#include "blurt.h"
#include <vector>

class Constellation {
protected:
    Constellation(int Nbpsc);
public:
    int Nbpsc;
    std::vector<complex> symbols;
    void map(const std::vector<bool> &input, std::vector<complex> &output);
    void demap(const std::vector<complex> &input, float dispersion, std::vector<int> &output);
};
