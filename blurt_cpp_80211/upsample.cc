#include "upsample.h"
#include <cstdlib>

void upsample(const std::vector<complex> &input, float n, std::vector<complex> &output) {
    int M = input.size();
    int N = 0;
    for (int i=0; i<24; i++) {
        if (M <= (1<<i)) {
            N = 1<<i;
            break;
        }
    }
    if (!N)
        abort();
    int L = N + int((n-1)*N);
    std::vector<complex> x(input);
    x.resize(N);
    fft(&x[0], N);
    x.resize(L);
    for (int i=N-1; i>=N/2; i--)
        x[L-N+i] = x[i];
    for (int i=N/2; i<L-N/2; i++)
        x[i] = 0;
    ifft(&x[0], L);
    output.assign(x.begin(), x.begin() + int(M*n));
}
