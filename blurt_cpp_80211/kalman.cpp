#include "kalman.h"

KalmanPilotTracker::KalmanPilotTracker(float uncertainty, float var_n) {
    const float std_theta = uncertainty;
    const float sigma_noise = 4*var_n*.5;
    const float sin_std_theta = sin(std_theta);
    const float sin_std_theta_sq = sin_std_theta*sin_std_theta;
    const float sigma_re = sigma_noise + 4*sin_std_theta_sq;
    const float sigma_im = sigma_noise + 4*sin_std_theta_sq;
    const float sigma_theta = std_theta*std_theta;
    P.resize(9, 0); P[0] = sigma_re; P[4] = sigma_im; P[8] = sigma_theta;
    x.resize(3, 0); x[0] = 4;
    Q.resize(9, 0); Q[0] = P[0] * .1f; Q[4] = P[4] * .1f; Q[8] = P[8] * .1f;
    R.resize(4); R[0] = sigma_noise; R[3] = sigma_noise;
}

template <class T>
inline void dot(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &c, int n, int m, int l) {
    // multiply an n x m matrix by an m x l matrix to get an n x l matrix
    c.resize(n*l);
    for (int i=0; i<n; i++) {
        for (int j=0; j<l; j++) {
            T acc = 0;
            for (int k=0; k<m; k++)
                acc += a[i*m+k] * b[k*l+j];
            c[i*l+j] = acc;
        }
    }
}

template <class T>
inline void trans(const std::vector<T> &a, std::vector<T> &b, int n, int m) {
    // transpose an n x m matrix
    b.resize(n*m);
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            b[j*n+i] = a[i*m+j];
}

template <class T>
void solve(const std::vector<T> &a, const std::vector<T> &b, std::vector<T> &x, int n, int m, int l) {
    // solve a x = b for x
    // a is n x m, b is n x l, x is m x l
    x.resize(m*l);
    // ... ?
    // profit
    if (n == 2 && m == 2) {
        std::vector<T> ai(4);
        T det = a[0]*a[3] - a[1]*a[2];
        ai[0] = a[3]/det; ai[1] = -a[1]/det; ai[2] = -a[2]/det; ai[3] = a[0]/det;
        dot(ai, b, x, n, m, l);
    } else {
        // do some fancy LU stuff using the code in lu.cpp
    }
}

void KalmanPilotTracker::update(complex pilot, complex &u) {
    const float re = x[0], im = x[1], theta = x[2];
    const float c = cos(theta);
    const float s = sin(theta);
    std::vector<float> F(9);
    F[0] = c; F[1] = -s; F[2] = -s*re - c*im;
    F[3] = s; F[4] =  c; F[5] =  c*re - s*im;
    F[6] = 0; F[7] =  0; F[8] = 1;
    x[0] = c*re - s*im;
    x[1] = c*im + s*re;
    std::vector<float> tmp;
    dot(F, P, tmp, 3, 3, 3);
    std::vector<float> FT;
    trans(F, FT, 3, 3);
    dot(tmp, FT, P, 3, 3, 3);
    for (int i=0; i<9; i++)
        P[i] += Q[i];
    std::vector<float> y(2);
    y[0] = real(pilot) - x[0];
    y[1] = imag(pilot) - x[1];
    std::vector<float> S(4);
    for (int i=0; i<2; i++)
        for (int j=0; j<2; j++)
            S[2*i+j] = P[3*i+j] + R[2*i+j];
    std::vector<float> K, KT;
    solve(S, P, KT, 2, 2, 3);
    trans(KT, K, 2, 3);
    dot(K, y, tmp, 3, 2, 1);
    for (int i=0; i<3; i++)
        x[i] += tmp[i];
    dot(K, P, tmp, 3, 2, 3);
    for (int i=0; i<9; i++)
        P[i] -= tmp[i];
    u = complex(x[0], -x[1]);
    u /= abs(u);
}
