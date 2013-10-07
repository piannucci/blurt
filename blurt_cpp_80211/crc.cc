#include "crc.h"
#include <algorithm>
#include "util.h"
#include <stdint.h>

void CRC::remainder_slow(const bitvector &a, bitvector &output) {
    bitvector c(a);
    size_t C = c.size(), B = b.size();
    for (size_t i=0; i<C-B+1; i++)
        if (c[i])
            for (size_t j=0; j<B; j++)
                c[i+j] = c[i+j] ^ b[j];
    output.assign(c.rbegin(), c.rbegin()+(ssize_t)(B-1));
}

uint32_t CRC::remainder_fast(const bitvector &a, bitvector &output) const {
    size_t N = a.size(), A = (N+L-1)/L;
    std::vector<uint32_t> a_words(A);
    for (size_t i=0; i<A; i++) {
        uint32_t x = 0;
        size_t k = N-1-i*L;
        for (size_t j=0; j<L && j <= k; j++)
            x += (uint32_t)a[k-j] << j;
        a_words[A-1-i] = x;
    }
    uint32_t r = 0;
    for (size_t i=0; i<A; i++)
        r = ((r << L) & m) ^ a_words[i] ^ lut[r >> s];
    output.resize(M);
    for (size_t i=0; i<M; i++)
        output[i] = (r >> (M-1-i)) & 1;
    return r;
}

void CRC::lut_bootstrap(const bitvector &new_b, size_t new_L) {
    b.assign(new_b.begin(), new_b.end());
    size_t new_M = b.size()-1;
    std::vector<uint32_t> new_lut(1<<new_L);
    if (L == 0) {
        // no look-up table; fall back on slow code
        std::vector<uint32_t> results;
        for (size_t i=0; i<1<<new_L; i++) {
            std::vector<uint32_t> num(1);
            num[0] = (uint32_t)i;
            bitvector a;
            shiftout(num, new_L, a);
            std::reverse(a.begin(), a.end());
            a.resize(a.size() + new_M);
            bitvector rem;
            remainder_slow(a, rem);
            shiftin(rem, new_M, results);
            new_lut[i] = results[0];
        }
    } else {
        bitvector bits;
        for (size_t i=0; i<1<<new_L; i++) {
            std::vector<uint32_t> num(1);
            num[0] = (uint32_t)i;
            bitvector a;
            shiftout(num, new_L, a);
            std::reverse(a.begin(), a.end());
            a.resize(a.size() + new_M);
            new_lut[i] = remainder_fast(a, bits);
        }
    }
    lut.assign(new_lut.begin(), new_lut.end());
    M = new_M;
    L = new_L;
    s = M-L;
    m = (uint32_t)((1ull<<M)-1);
}

void CRC::FCS(const bitvector &calculationFields, bitvector &output) const {
    bitvector a;
    a.assign(calculationFields.begin(), calculationFields.end());
    a.resize(a.size() + 32);
    for (size_t i=0; i<32; i++)
        a[i] = !a[i];
    remainder_fast(a, output);
    for (size_t i=0; i<output.size(); i++)
        output[i] = !output[i];
}

bool CRC::checkFCS(const bitvector &frame) const {
    bitvector a;
    a.assign(frame.begin(), frame.end());
    a.resize(a.size() + 32);
    for (size_t i=0; i<32; i++)
        a[i] = !a[i];
    bitvector output;
    remainder_fast(a, output);
    return std::equal(output.begin(), output.end(), correct_remainder.begin());
}

static const bool CRC32_802_11_FCS_G[] = {1,0,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,1,1,1,0,1,1,0,1,1,0,1,1,1};
static const bool CRC32_802_11_FCS_remainder[] = {1,1,0,0,0,1,1,1,0,0,0,0,0,1,0,0,1,1,0,1,1,1,0,1,0,1,1,1,1,0,1,1};

CRC::CRC()
    : L(0), correct_remainder(CRC32_802_11_FCS_remainder, CRC32_802_11_FCS_remainder+32) {
    bitvector G(CRC32_802_11_FCS_G, CRC32_802_11_FCS_G+33);
    lut_bootstrap(G, 8);
    lut_bootstrap(G, 16);
}
