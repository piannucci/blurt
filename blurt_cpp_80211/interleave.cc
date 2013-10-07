#include "interleave.h"

void interleave_permutation(size_t Ncbps, size_t Nbpsc, std::vector<size_t> &j) {
    size_t s = Nbpsc/2;
    if (s < 1) s = 1;
    j.resize(Ncbps);
    for (size_t k=0; k<Ncbps; k++) {
        size_t i = (Ncbps/16) * (k % 16) + (k/16);
        j[k] = s * (i/s) + (i + Ncbps - (16 * i / Ncbps)) % s;
    }
}

void interleave_inverse_permutation(size_t Ncbps, size_t Nbpsc, std::vector<size_t> &k) {
    size_t s = Nbpsc/2;
    if (s < 1) s = 1;
    k.resize(Ncbps);
    for (size_t j=0; j<Ncbps; j++) {
        size_t i = s * (j/s) + (j + (16*j/Ncbps)) % s;
        k[j] = 16 * i - (Ncbps - 1) * (16 * i / Ncbps);
    }
}
