#include <cstdint>
#include <complex>
#include <map>
#include <string>
#include <blitz/array.h>
#include <complex>
#include <iostream>
#include <queue>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreAudio/AudioHardware.h>

using namespace std;
using namespace std::literals;
using namespace blitz;

/*
 * ====================================================
 * logaddexp function is
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

double logaddexp(double x, double y)
{
    if (x == y)
        return x + 0.693147180559945309417232121458176568;
    double a = x - y;
    if (a > 0)
    {
        a = -a;
        y = x;
    }
    a = exp(a);
    if (__builtin_isinf(a))
        return y + a;
    const double u = 1. + a;
    const double d = u - 1.;
    return y + (d ? log(u) * a / d : a);
}

BZ_DECLARE_FUNCTION2(logaddexp)

const int mtu = 150;
const double Fs = 96e3, Fc = 20e3, upsample_factor = 32;
const double stereoDelay = .005;
const int preemphasisOrder = 13;

const int audioInputFrameSize = 2048;

const int nfft = 64;
const int ncp = 16;
const complex<double> sts_freq[64] = {};
const complex<double> lts_freq[64] = {
    0, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,
    1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,
    1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1
};
const int ts_reps = 2;
const int dataSubcarriers[] = {
    -26,-25,-24,-23,-22,-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-6,-5,-4,-3,-2,-1,
      1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,18,19,20,22,23,24,25,26};
const int pilotSubcarriers[] = {-21,-7,7,21};
const int pilotTemplate[] = {1,1,1,-1};

Array<uint8_t,2> scrambler(128,127);
void init_scrambler()
{
    Array<uint8_t,1> s(128);
    s = firstIndex();
    for (int j=0; j<127; j++)
        scrambler(Range::all(),j) = (s = (s << 1) ^ (1 & ((s >> 3) ^ (s >> 6)))) & 1;
}

Array<uint32_t,1> lut(1<<16);
void init_crc()
{
    Array<uint64_t,1> lut_(1<<16);
    lut_ = cast<uint64_t>(firstIndex()) << 32;
    for (int j=47; j!=31; j--)
        lut_ ^= where((lut_ >> j) & 1, uint64_t(0x104c11db7ull << (j-32)), 0);
    lut = lut_;
}

uint32_t CRC(const Array<bool,1> &x)
{
    int x_size = x.extent(0);
    Array<bool,1> a(32 + x_size + (16 - x_size % 16));
    int n = a.extent(0)/16;
    a = 0;
    a(Range(32+x_size-1, 32, -1)) = x;
    a(Range(x_size, x_size+32-1)) ^= 1;
    uint32_t r = 0;
    for (int i=0; i<n; i++)
    {
        r = (r << 16) ^ lut(r >> 16);
        for (int j=0; j<16; j++)
            r ^= uint16_t(a((n-1-i)*16+j)) << j;
    }
    return r;
}

Array<uint8_t,2> output_map(2,128);
Array<int,2> output_map_soft(2,128);

void init_cc()
{
    Array<uint8_t,1> a(2);
    a = 109, 79;
    Array<uint8_t,1> b(128);
    b = firstIndex();
    output_map = 0;
    for (int i=0; i<7; i++)
        output_map ^= a(firstIndex()) * (b(secondIndex()) & (1<<i));
    output_map = 1 & (output_map >> 6);
    output_map_soft = cast<int>(output_map) * 2 - 1;
}

Array<uint8_t,1> encode(const Array<uint8_t,1> &y)
{
    int sh = 0, N = y.extent(0);
    Array<uint8_t,1> output(N*2);
    for (int i=0; i<N; i++) {
        sh = (sh>>1) ^ ((int)y(i) << 6);
        output(2*i+0) = output_map(0,sh);
        output(2*i+1) = output_map(1,sh);
    }
    return output;
}

Array<uint8_t,1> decode(const Array<int,1> &llr)
{
    int N = llr.extent(0)/2;
    Array<int,2> x(N,128);
    x = llr(Range(0,N*2-1,2))(firstIndex()) * output_map_soft(0,Range::all())(secondIndex()) +
        llr(Range(1,N*2-1,2))(firstIndex()) * output_map_soft(1,Range::all())(secondIndex());
    Array<uint8_t,1> msg(N);
    const int M = 128;
    int64_t cost[M*2], scores[M] = {/* zero-initialized */};
    uint8_t bt[N][M];
    for (int k=0; k<N; k++) {
        for (int i=0; i<M; i++) {
            cost[2*i+0] = scores[((i<<1) & 127) | 0] + x(k, i);
            cost[2*i+1] = scores[((i<<1) & 127) | 1] + x(k, i);
        }
        for (int i=0; i<M; i++) {
            int a = cost[2*i+0];
            int b = cost[2*i+1];
            bt[k][i] = (a<b) ? 1 : 0;
            scores[i] = (a<b) ? b : a;
        }
    }
    int i = (scores[0] < scores[1]) ? 1 : 0;
    for (int k=N-1; k>=0; k--) {
        int j = bt[k][i];
        msg(k) = i >> 6;
        i = ((i<<1)&127) + j;
    }
    return msg;
}

template<typename P_sourcetype, typename P_resulttype = BZ_SUMTYPE(P_sourcetype)>
class ReduceLogsumexp {
public:

    typedef P_sourcetype T_sourcetype;
    typedef P_resulttype T_resulttype;
    typedef T_resulttype T_numtype;

    static const bool needIndex = false, needInit = false;

    ReduceLogsumexp() { }

    bool operator()(const T_sourcetype& x,const int=0) const { 
        sum_ = logaddexp(x, sum_); 
        return true;
    }

    T_resulttype result(const int) const { return sum_; }

    void reset() const { sum_ = -infinity(T_resulttype()); }
 
    static const char* name() { return "logsumexp"; }
 
protected:

    mutable T_resulttype sum_;
};

BZ_DECL_ARRAY_PARTIAL_REDUCE(logsumexp, ReduceLogsumexp)
BZ_DECL_ARRAY_FULL_REDUCE(logsumexp, ReduceLogsumexp)

class Rate {
    public:
        int Nbpsc;
        Array<complex<double>,1> symbols;
        Array<bool,1> puncturingMatrix;
        pair<int,int> ratio;
        Rate(int Nbpsc, pair<int,int> ratio) :
            Nbpsc(Nbpsc), symbols(1<<Nbpsc), puncturingMatrix(ratio.first*2),
            ratio(ratio)
        {
            if (Nbpsc == 1)
                symbols = -1, 1;
            else
            {
                int n = Nbpsc/2;
                Array<int,1> grayRevCode(1<<n);
                grayRevCode = 0;
                for (int i=0; i<n; i++)
                    grayRevCode += ((firstIndex() >> i) & 1) << (n-1-i);
                grayRevCode ^= grayRevCode >> 1;
                grayRevCode ^= grayRevCode >> 2;
                Array<double,1> symbols_(1<<n);
                symbols_ = (2*grayRevCode+1-(1<<n)) * sqrt(1.5 / ((1<<Nbpsc) - 1));
                for (int i=0; i<1<<n; i++)
                    symbols(Range((1<<n)*i,(1<<n)*(i+1)-1)) = symbols_;
                for (int j=0; j<1<<n; j++)
                    symbols(Range(j,(1<<Nbpsc)-1,1<<n)) = symbols_ * 1i;
            }
            switch (ratio.first) {
                case 1://2
                    puncturingMatrix = 1,1;
                    break;
                case 2://3
                    puncturingMatrix = 1,1,1,0;
                    break;
                case 3://4
                    puncturingMatrix = 1,1,1,0,0,1;
                    break;
                case 5://6
                    puncturingMatrix = 1,1,1,0,0,1,1,0,0,1;
                    break;
            }
        }
            
        template <class T>
        Array<T,1> depuncture(const Array<T,1> &y)
        {
            int output_size = (y.size() + ratio.second-1) / ratio.second * ratio.first * 2;
            Array<T,1> output(output_size);
            int j=0;
            for (int i=0; i<output_size; i++)
            {
                bool mask = puncturingMatrix(i % puncturingMatrix.extent(0));
                output(i) = mask ? y(j++) : 0;
            }
            return output;
        }

        Array<int,2> demap(const Array<complex<double>,1> &y, double dispersion)
        {
            Array<double,2> squared_distance(y.extent(0), 1<<Nbpsc);
            squared_distance = sqr(abs(symbols(secondIndex()) - y(firstIndex())));
            Array<double,2> ll(y.extent(0), 1<<Nbpsc);
            ll = -log(M_PI * dispersion) - squared_distance / dispersion;
            Array<double,1> llsum(y.extent(0));
            llsum = logsumexp(ll, secondIndex());
            ll -= llsum(firstIndex());
            Array<int,2> llr(y.extent(0), Nbpsc);
            llr = 0;
            for (int i=0; i<Nbpsc; i++)
            {
                Array<double,1> llsum1(y.extent(0)), llsum0(y.extent(0));
                llsum1 = -infinity(double());
                llsum0 = -infinity(double());
                for (int j=0; j<1<<Nbpsc; j++)
                    if (0 != (j & (1<<i)))
                        llsum1 = logaddexp(llsum1, ll(Range::all(),j));
                    else
                        llsum0 = logaddexp(llsum0, ll(Range::all(),j));
                llr(Range::all(),i) = 10 * (llsum1 - llsum0);
            }
            llr = where(llr > 1e4, 1e4, where(llr < -1e4, -1e4, llr));
            return llr;
        }
};

map<int, Rate> rates = {
    {0xb, Rate(1, {1,2})}, {0xf, Rate(2, {1,2})}, {0xa, Rate(2, {3,4})}, {0xe, Rate(4, {1,2})},
    {0x9, Rate(4, {3,4})}, {0xd, Rate(6, {2,3})}, {0x8, Rate(6, {3,4})}, {0xc, Rate(6, {5,6})}
};

const int Nsc = sizeof(dataSubcarriers)/sizeof(dataSubcarriers[0]);
const int Nsc_used = Nsc + sizeof(pilotSubcarriers)/sizeof(pilotSubcarriers[0]);
const int N_sts_period = nfft / 4;
const int N_sts_samples = ts_reps * (ncp + nfft);
const int N_sts_reps = N_sts_samples / N_sts_period;
const int N_training_samples = N_sts_samples + ts_reps * (ncp + nfft) + 8;

Array<int,1> interleave(int Ncbps, int Nbpsc, bool reverse=false)
{
    int s = max(Nbpsc/2, 1);
    Array<int,1> j(Ncbps), i(Ncbps), p(Ncbps);
    j = firstIndex();
    if (reverse)
    {
        i = (Ncbps/16) * (j%16) + (j/16);
        p = s*(i/s) + (i + Ncbps - (16*i/Ncbps)) % s;
    }
    else
    {
        i = s*(j/s) + (j + (16*j/Ncbps)) % s;
        p = 16*i - (Ncbps - 1) * (16*i/Ncbps);
    }
    return p;
}

class Autocorrelator {
    public:
        Array<complex<double>,1> y_hist;
        queue<Array<double,1>> results;
        Autocorrelator() : y_hist(0) {}
        void feed(const Array<complex<double>,1> &sequence)
        {
            Array<complex<double>,1> y(sequence.extent(0) + y_hist.extent(0));
            y(Range(y_hist.extent(0), toEnd)) = sequence;
            y(Range(0, y_hist.extent(0)-1)) = y_hist;
            int count_needed = y.extent(0) / N_sts_period * N_sts_period;
            int count_consumed = count_needed - N_sts_reps * N_sts_period;
            if (count_consumed <= 0)
            {
                y_hist.resize(y.extent(0));
                y_hist = y;
            }
            else
            {
                y_hist.resize(y.extent(0)-count_consumed);
                y_hist = y(Range(count_consumed,toEnd));
                Array<complex<double>,2> y2(count_needed/N_sts_period, N_sts_period);
                for (int i=0; i<y2.extent(0); i++)
                    y2(i,Range::all()) = y(Range(i*N_sts_period,(i+1)*N_sts_period-1));
                Array<complex<double>,2> corr_sum2(y2.extent(0)-1, y2.extent(1));
                corr_sum2 = conj(y2(Range(0,y.extent(0)-2),Range::all())) * y2(Range(1,toEnd),Range::all());
                Array<double,1> corr_sum(y2.extent(0)-1);
                corr_sum = abs(sum(corr_sum2, secondIndex()));
                double acc=0;
                for (int i=0; i<corr_sum.extent(0); i++)
                    corr_sum(i) = (acc += corr_sum(i));
                Array<double,1> corr(corr_sum.extent(0) - (N_sts_reps-1));
                corr = corr_sum(Range(N_sts_reps-1,toEnd)) - corr_sum(Range(fromStart,corr_sum.extent(0)-N_sts_reps));
                results.push(corr);
            }
        }

        bool hasnext()
        {
            return !results.empty();
        }

        Array<double,1> next()
        {
            Array<double,1> x = results.front();
            results.pop();
            return x;
        }
};

class PeakDetector
{
    public:
        Array<double,1> y_hist;
        int l;
        int i;
        queue<int> results;
        PeakDetector(int l, int i) : y_hist(0), l(l), i(i) {}

        void feed(const Array<double,1> &sequence)
        {
            Array<double,1> y(sequence.extent(0) + y_hist.extent(0));
            y(Range(y_hist.extent(0), toEnd)) = sequence;
            y(Range(0, y_hist.extent(0)-1)) = y_hist;
            int count_needed = y.extent(0);
            int count_consumed = count_needed - 2*l;
            if (count_consumed <= 0)
            {
                y_hist.resize(y.extent(0));
                y_hist = y;
            }
            else
            {
                y_hist.resize(y.extent(0)-count_consumed);
                y_hist = y(Range(count_consumed,toEnd));
                y_hist = y;
                Array<double,2> stripes(count_needed-2*l, 2*l+1);
                for (int j=0; j<stripes.extent(0); j++)
                    for (int k=0; k<stripes.extent(1); k++)
                        stripes(j,k) = y(j+k);
                Array<bool,1> mask(stripes.extent(0));
                mask = maxIndex(stripes, secondIndex()) == l;
                for (int j=0; j<mask.extent(0); j++)
                    if (mask(j))
                        results.push(i+j);
                i += count_consumed;
            }
        }

        bool hasnext()
        {
            return !results.empty();
        }

        int next()
        {
            int x = results.front();
            results.pop();
            return x;
        }
};

int main(void)
{
    return 0;
}
