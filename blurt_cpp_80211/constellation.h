#ifndef CONSTELLATION_H
#define CONSTELLATION_H
#include "blurt.h"
#include <vector>

class Constellation {
protected:
    Constellation(size_t Nbpsc);
public:
    size_t Nbpsc;
    std::vector<complex> symbols;
    void map(const bitvector &input, std::vector<complex> &output) const;
    void demap(const std::vector<complex> &input, float dispersion, std::vector<int> &output) const;
};
#endif
