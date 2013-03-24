#include "blurt.h"

void interleave_permutation(int Ncbps, int Nbpsc, std::vector<int> &j);
void interleave_inverse_permutation(int Ncbps, int Nbpsc, std::vector<int> &k);

template <class T>
void interleave(const std::vector<T> &input, int Ncbps, int Nbpsc, bool reverse, std::vector<T> &output) {
    std::vector<int> p;
    if (reverse)
        interleave_permutation(Ncbps, Nbpsc, p);
    else
        interleave_inverse_permutation(Ncbps, Nbpsc, p);
    int N = input.size();
    output.resize(input.size);
    for (int i=0; i<N; i+=Ncbps)
        for (int j=0; j<Ncbps; j++)
            output[i+j] = input[i+p[j]];
}
