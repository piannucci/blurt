#include "fft.h"
#include <stdint.h>
#include <cmath>
#include <map>

#include <float.h>
typedef float real;

class FFT {
private:
    size_t n;
    bool forward;
    int *rev_idx, *rev_idx_end;
    complex *factors;
    real scale;
    inline void rev_in_place(complex *x);
public:
    FFT(size_t n, bool forward);
    ~FFT();
    void transform(complex *x);
};


static uint32_t rev_byte(uint8_t b)
{
    return ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
}

static uint32_t rev_word(uint32_t w)
{
    return (rev_byte((w >> 24) & 0xFF) <<  0) | (rev_byte((w >> 16) & 0xFF) <<  8) |
           (rev_byte((w >>  8) & 0xFF) << 16) | (rev_byte((w >>  0) & 0xFF) << 24);
}

static uint32_t rev(uint32_t w, size_t logn)
{
    return rev_word(w) >> (32 - logn);
}

inline void FFT::rev_in_place(complex *x)
{
    int *idx = rev_idx, *end = rev_idx_end;
    while (idx != end)
    {
        int i = *idx++;
        int j = *idx++;
        complex temp = x[j];
        x[j] = x[i];
        x[i] = temp;
    }
}

static inline complex omega(int e, size_t logn, bool forward)
{
    if (e == 0)
        return 1.f;
    if (!forward)
        e = -e;
    double theta = -2*M_PI*e/(1<<logn);
    return complex(float(cos(theta)), float(sin(theta)));
}

FFT::FFT(size_t n_, bool forward_) : n(n_), forward(forward_)
{
    rev_idx = new int [2 * (1<<n)];
    rev_idx_end = rev_idx;
    for (uint32_t i=0; i<(1<<n); i++)
    {
        uint32_t j = rev(i, n);
        if (j>i)
        {
            *rev_idx_end++ = int(i);
            *rev_idx_end++ = int(j);
        }
    }
    factors = new complex [1<<(n-1)];

    for (uint32_t i=0; i<(1<<(n-1)); i++)
        factors[i] = omega(int(rev(i, n-1)), n, forward);

    // 1/N scaling of IFFT
    scale = forward ? 1.f : powf(2.f, -float(n));
}

FFT::~FFT() {
    delete [] rev_idx;
    delete [] factors;
}

void FFT::transform(complex *x)
{
    rev_in_place(x);

    for (size_t m=0; m<n-1; m++)
    {
        size_t jmax = 1<<m, imax = (1<<(n-m-1));
        complex *x0 = x, *x1 = x;
        for (size_t i=0; i<imax; i++)
        {
            x1 += jmax;
            complex factor = factors[i];
            for (size_t j=0; j<jmax; j++)
            {
                complex _x1 = *x1, y1 = (*x0-_x1) * factor;
                *x0 += _x1; *x1 = y1;
                x0 += 1; x1 += 1;
            }
            x0 = x1;
        }
    }

    int jmax = 1<<(n-1);
    complex *x0 = x, *x1 = x + jmax;
    if (!forward)
    {
        for (int j=jmax; j; j--)
        {
            complex _x0 = *x0, _x1 = *x1;
            *x0++ = (_x0+_x1) * scale; *x1++ = (_x0-_x1) * scale;
        }
    }
    else
    {
        for (int j=jmax; j; j--)
        {
            complex _x1 = *x1, y1 = (*x0-_x1);
            *x0++ += _x1; *x1++ = y1;
        }
    }
}

int log2(size_t n) {
    int logn;
    for (logn=0; n>1; n>>=1, logn++);
    return logn;
}

void fft(complex *x, size_t n) {
    static std::map<size_t, FFT *> fft_objects;
    if (fft_objects.find(n) == fft_objects.end())
        fft_objects[n] = new FFT(size_t(log2(n)), true);
    fft_objects[n]->transform(x);
}

void ifft(complex *x, size_t n) {
    static std::map<size_t, FFT *> ifft_objects;
    if (ifft_objects.find(n) == ifft_objects.end())
        ifft_objects[n] = new FFT(size_t(log2(n)), false);
    ifft_objects[n]->transform(x);
}
