#include "fft.h"
#include <stdint.h>
#include <cmath>
#include <map>

#include <float.h>
typedef float real;
#define IFFT_GET_SCALE(_scale_) real _scale_ = scale;
#define IFFT_SCALE(_x_, _n_, _scale_) { _x_ *= scale; }
#define TWIDDLE_SCALE(_x_)

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


static uint8_t rev_byte(uint8_t b)
{
    return ((b * 0x80200802ULL) & 0x0884422110ULL) * 0x0101010101ULL >> 32;
}

static uint32_t rev_word(uint32_t w)
{
    return (rev_byte((w >> 24) & 0xFF) <<  0) | (rev_byte((w >> 16) & 0xFF) <<  8) |
           (rev_byte((w >>  8) & 0xFF) << 16) | (rev_byte((w >>  0) & 0xFF) << 24);
}

static uint32_t rev(uint32_t w, int logn)
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

static inline complex omega(int e, int logn, bool forward)
{
    if (e == 0)
        return 1.f;
    if (!forward)
        e = -e;
    double theta = -2*M_PI*e/(1<<logn);
    return complex(cos(theta), sin(theta));
}

FFT::FFT(size_t n, bool forward) : n(n), forward(forward)
{
    rev_idx = new int [2 * (1<<n)];
    rev_idx_end = rev_idx;
    for (int i=0; i<(1<<n); i++)
    {
        int j = rev(i, n);
        if (j>i)
        {
            *rev_idx_end++ = i;
            *rev_idx_end++ = j;
        }
    }
    factors = new complex [1<<(n-1)];

    for (int i=0; i<(1<<(n-1)); i++)
        factors[i] = omega(rev(i, n-1), n, forward);

    // 1/N scaling of IFFT
    scale = forward ? 1.f : pow(2, -n);
}

FFT::~FFT() {
    delete [] rev_idx;
    delete [] factors;
}

void FFT::transform(complex *x)
{
    rev_in_place(x);

    int n = n;
    complex *factors = factors;
    for (int m=0; m<n-1; m++)
    {
        int jmax = 1<<m, imax = (1<<(n-m-1));
        complex *x0 = x, *x1 = x;
        for (int i=0; i<imax; i++)
        {
            x1 += jmax;
            complex factor = factors[i];
            for (int j=0; j<jmax; j++)
            {
                complex _x1 = *x1, y1 = (*x0-_x1) * factor;
                TWIDDLE_SCALE(y1);
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
        IFFT_GET_SCALE(scale);
        for (int j=jmax; j; j--)
        {
            complex _x1 = *x1, y0 = (*x0+_x1), y1 = (*x0-_x1);
            IFFT_SCALE(y0, n, scale); IFFT_SCALE(y1, n, scale);
            *x0 = y0; *x1 = y1;
            x0 += 1; x1 += 1;
        }
    }
    else
    {
        for (int j=jmax; j; j--)
        {
            complex _x1 = *x1, y1 = (*x0-_x1);
            *x0 += _x1; *x1 = y1;
            x0 += 1; x1 += 1;
        }
    }
}

int log2(size_t n) {
    int logn;
    for (logn=0; n>1; n>>=1, logn++);
    return logn;
}

std::map<size_t, FFT *> fft_objects;

void fft(complex *x, size_t n) {
    if (fft_objects.find(n) == fft_objects.end())
        fft_objects[n] = new FFT(log2(n), true);
    fft_objects[n]->transform(x);
}

std::map<size_t, FFT *> ifft_objects;

void ifft(complex *x, size_t n) {
    if (ifft_objects.find(n) == ifft_objects.end())
        ifft_objects[n] = new FFT(log2(n), false);
    ifft_objects[n]->transform(x);
}
