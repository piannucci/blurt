#include "interleave.h"

void interleave_permutation(int Ncbps, int Nbpsc, std::vector<int> &j) {
    int s = Nbpsc/2;
    if (s < 1) s = 1;
    j.resize(Ncbps);
    for (int k=0; k<Ncbps; k++) {
        int i = (Ncbps/16) * (k % 16) + (k/16);
        j[k] = s * (i/s) + (i + Ncbps - (16 * i / Ncbps)) % s;
    }
}

void interleave_inverse_permutation(int Ncbps, int Nbpsc, std::vector<int> &k) {
    int s = Nbpsc/2;
    if (s < 1) s = 1;
    k.resize(Ncbps);
    for (int j=0; j<Ncbps; j++) {
        int i = s * (j/s) + (j + (16*j/Ncbps)) % s;
        k[j] = 16 * i - (Ncbps - 1) * (16 * i / Ncbps);
    }
}
