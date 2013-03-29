#ifndef IIR_H
#define IIR_H
#import "blurt.h"
#import "mkfilter.h"

template <class T>
class IIRFilter {
private:
    int order;
    float *alpha, *beta;
    float gamma;
    T *x_hist;
    T *y_hist;
public:
    IIRFilter(int order, float *alpha, float *beta, float gamma) :
        order(order), alpha(new float [order+1]), beta(new float [order]), gamma(gamma),
        x_hist(new T [order]), y_hist(new T [order])
    {
        for (int i=0; i<order; i++) {
            this->x_hist[i] = 0.f;
            this->y_hist[i] = 0.f;
            this->alpha[i] = alpha[i];
            this->beta[i] = beta[i];
        }
        this->alpha[order] = alpha[order];
    }

    IIRFilter(const char *argv[])
    {
        float alpha[MAXPZ], beta[MAXPZ];
        mkfilter(argv, order, alpha, beta, gamma);
        this->alpha = new float[order+1];
        this->beta = new float[order];
        x_hist = new T [order];
        y_hist = new T [order];
        for (int i=0; i<order; i++) {
            this->x_hist[i] = 0.f;
            this->y_hist[i] = 0.f;
            this->alpha[i] = alpha[i];
            this->beta[i] = beta[i];
        }
        this->alpha[order] = alpha[order];
    };

    ~IIRFilter() {
        delete [] x_hist;
        delete [] y_hist;
        delete [] alpha;
        delete [] beta;
    };

    void apply(const std::vector<T> &input, std::vector<T> &output) {
        size_t N = input.size();
        output.resize(N);
        T *x = new T [order + N];
        T *y = new T [order + N];
        for (int i=0; i<order; i++) {
            x[i] = x_hist[i];
            y[i] = y_hist[i];
        }
        for (int i=order; i<N+order; i++) {
            x[i] = input[i-order];
            T acc = 0;
            for (int j=0; j<order; j++) {
                acc += alpha[j]*x[i+j-order];
                acc += beta[j]*y[i+j-order];
            }
            acc += alpha[order]*x[i];
            y[i] = acc;
        }
        for (int i=0; i<order; i++) {
            x_hist[i] = x[N+i];
            y_hist[i] = y[N+i];
        }
        for (int i=order; i<N+order; i++)
            output[i-order] = y[i] * gamma;
        delete [] x;
        delete [] y;
    }
};
#endif
