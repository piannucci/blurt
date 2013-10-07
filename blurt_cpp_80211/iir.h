#ifndef IIR_H
#define IIR_H
#include "blurt.h"
#include "mkfilter.h"

template <class T>
class IIRFilter {
private:
    size_t order;
    float *alpha, *beta;
    float gamma;
    T *x_hist;
    T *y_hist;
public:
    IIRFilter(size_t order_, float *alpha_, float *beta_, float gamma_) :
        order(order_), alpha(new float [order+1]), beta(new float [order]), gamma(gamma_),
        x_hist(new T [order]), y_hist(new T [order])
    {
        for (size_t i=0; i<order; i++) {
            this->x_hist[i] = 0.f;
            this->y_hist[i] = 0.f;
            this->alpha[i] = alpha_[i];
            this->beta[i] = beta_[i];
        }
        this->alpha[order] = alpha_[order];
    }

    IIRFilter(const char *argv[])
    {
        float mkfilter_alpha[MAXPZ], mkfilter_beta[MAXPZ];
        mkfilter(argv, order, mkfilter_alpha, mkfilter_beta, gamma);
        this->alpha = new float[order+1];
        this->beta = new float[order];
        x_hist = new T [order];
        y_hist = new T [order];
        for (size_t i=0; i<order; i++) {
            this->x_hist[i] = 0.f;
            this->y_hist[i] = 0.f;
            this->alpha[i] = mkfilter_alpha[i];
            this->beta[i] = mkfilter_beta[i];
        }
        this->alpha[order] = mkfilter_alpha[order];
    }

    ~IIRFilter() {
        delete [] x_hist;
        delete [] y_hist;
        delete [] alpha;
        delete [] beta;
    }

    void apply(const std::vector<T> &input, std::vector<T> &output) {
        size_t N = input.size();
        output.resize(N);
        T *x = new T [order + N];
        T *y = new T [order + N];
        for (size_t i=0; i<order; i++) {
            x[i] = x_hist[i];
            y[i] = y_hist[i];
        }
        for (size_t i=order; i<N+order; i++) {
            x[i] = input[i-order];
            T acc = 0;
            for (size_t j=0; j<order; j++) {
                acc += alpha[j]*x[i+j-order];
                acc += beta[j]*y[i+j-order];
            }
            acc += alpha[order]*x[i];
            y[i] = acc;
        }
        for (size_t i=0; i<order; i++) {
            x_hist[i] = x[N+i];
            y_hist[i] = y[N+i];
        }
        for (size_t i=order; i<N+order; i++)
            output[i-order] = y[i] * gamma;
        delete [] x;
        delete [] y;
    }
};
#endif
