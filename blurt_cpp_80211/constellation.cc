#include "constellation.h"
#include <limits>
#include <cmath>

float log1p(float x)
{
    const float u = 1.f + x;
    const float d = u - 1.f;
    return fabsf(d)>0 ? logf(u) * x / d : x;
}

float logaddexp(float x, float y)
{
    const float tmp = x - y;
    if (tmp > 0)
        return x + log1p(expf(-tmp));
    else if (tmp <= 0)
        return y + log1p(expf(tmp));
    else
        return x + y;
}

Constellation::Constellation(size_t Nbpsc_) : Nbpsc(Nbpsc_), symbols(1<<Nbpsc_) {
}

void Constellation::map(const bitvector &input, std::vector<complex> &output) const {
    output.resize(input.size()/Nbpsc);
    size_t j=0;
    for (size_t i=0; i<output.size(); i++) {
        size_t data = 0;
        for (size_t k=0; k<Nbpsc; k++)
            data |= (size_t)input[j++] << k;
        output[i] = symbols[data];
    }
}

void Constellation::demap(const std::vector<complex> &input, float dispersion, std::vector<int> &output) const {
    output.resize(input.size() * Nbpsc);
    std::vector<float> ll(1<<Nbpsc);
    const float minus_log_pi_dispersion = -logf((float)M_PI * dispersion);
    float neginf = -std::numeric_limits<float>::infinity();
    for (size_t k=0; k<input.size(); k++) {
        float ll_norm = neginf;
        for (size_t j=0; j<(1<<Nbpsc); j++) {
            float squared_distance = std::norm(input[k] - symbols[j]);
            float ll_val = minus_log_pi_dispersion - squared_distance / dispersion;
            ll_norm = logaddexp(ll_norm, ll_val);
            ll[j] = ll_val;
        }
        for (size_t j=0; j<(1<<Nbpsc); j++)
            ll[j] -= ll_norm;
        for (size_t i=0; i<Nbpsc; i++) {
            float ll0 = neginf, ll1 = neginf;
            for (size_t j=0; j<(1<<Nbpsc); j++) {
                if ((j >> i) & 1)
                    ll1 = logaddexp(ll1, ll[j]);
                else
                    ll0 = logaddexp(ll0, ll[j]);
            }
            float x = 10.f * (ll1 - ll0);
            if (x > 1e4) x = 1e4;
            if (x < -1e4) x = -1e4;
            output[k*Nbpsc+i] = (int)x;
        }
    }
}
