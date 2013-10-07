#include "blurt.h"

void interleave_permutation(size_t Ncbps, size_t Nbpsc, std::vector<size_t> &j);
void interleave_inverse_permutation(size_t Ncbps, size_t Nbpsc, std::vector<size_t> &k);

template <class T>
void interleave(const std::vector<T> &input, size_t Ncbps, size_t Nbpsc, bool reverse, std::vector<T> &output) {
    std::vector<size_t> p;
    if (reverse)
        interleave_permutation(Ncbps, Nbpsc, p);
    else
        interleave_inverse_permutation(Ncbps, Nbpsc, p);
    size_t N = input.size();
    output.resize(input.size());
    for (size_t i=0; i<N; i+=Ncbps)
        for (size_t j=0; j<Ncbps; j++)
            output[i+j] = input[i+p[j]];
}
