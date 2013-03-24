#include "cc.h"

int rev(int x, int n) {
    int y = 0;
    for (int i=0; i<n; i++) {
        y <<= 1;
        y |= x&1;
        x >>= 1;
    }
    return y;
}

int mul(int a, int b) {
    int c = 0;
    int i = 0;
    while (b) {
        if (b&1)
            c ^= a << i;
        b >>= 1;
        i += 1;
    }
    return c;
}

ConvolutionalCode::ConvolutionalCode(int nu, int g0, int g1) : nu(nu) {
    g0 = rev(g0, nu);
    g1 = rev(g1, nu);
    output_map_1.resize(1<<nu);
    output_map_2.resize(1<<nu);
    state_inv_map[0].resize(1<<nu); state_inv_map[1].resize(1<<nu);
    output_map_1_soft.resize(1<<nu);
    output_map_2_soft.resize(1<<nu);
    for (int i=0; i<1<<nu; i++) {
        output_map_1[i] = 1 & (mul(g0, i) >> (nu-1));
        output_map_2[i] = 1 & (mul(g1, i) >> (nu-1));
        state_inv_map[0][i] = ((i << 1) & ((1<<nu)-1));
        state_inv_map[1][i] = ((i << 1) & ((1<<nu)-1)) + 1;
        output_map_1_soft[i] = output_map_1[i] * 2 - 1;
        output_map_2_soft[i] = output_map_2[i] * 2 - 1;
    }
}

void ConvolutionalCode::encode(const bitvector &input, bitvector &output) {
    int N = input.size();
    output.resize(N*2);
    uint32_t sh = 0;
    for (int i=0; i<N; i++) {
        sh = (sh>>1) ^ ((int)input[i] << (nu-1));
        output[2*i+0] = output_map_1[sh];
        output[2*i+1] = output_map_2[sh];
    }
}

void ConvolutionalCode::decode(const std::vector<int> &input, int length, bitvector &output) {
    int N = length+nu-1;
    int M = 1<<nu;
    if (input.size() < N*2) {
        output.resize(0);
        return;
    }
    bitvector bt(N*M);
    output.resize(N);
    std::vector<int64_t> cost(M*2);
    std::vector<int64_t> scores(M, 0);
    for (int k=0; k<N; k++) {
        for (int i=0; i<M; i++) {
            int edgeCost = input[2*k+0]*output_map_1_soft[i] + input[2*k+1]*output_map_2_soft[i];
            cost[2*i+0] = scores[state_inv_map[0][i]] + edgeCost;
            cost[2*i+1] = scores[state_inv_map[1][i]] + edgeCost;
        }
        for (int i=0; i<M; i++) {
            int a = cost[2*i+0], b = cost[2*i+1];
            bt[(k<<nu)+i] = (a<b) ? 1 : 0;
            scores[i] = (a<b) ? b : a;
        }
    }
    int i = (scores[0] < scores[1]) ? 1 : 0;
    for (int k=N-1; k>=0; k--) {
        int j = bt[(k<<nu)+i];
        output[k] = i >> (nu-1);
        i = state_inv_map[j][i];
    }
}
