#include "cc.h"
#include "util.h"

ConvolutionalCode::ConvolutionalCode(size_t nu_, uint32_t g0, uint32_t g1) : nu(nu_) {
    g0 = rev(g0, nu);
    g1 = rev(g1, nu);
    output_map_1.resize(1<<nu);
    output_map_2.resize(1<<nu);
    state_inv_map[0].resize(1<<nu); state_inv_map[1].resize(1<<nu);
    output_map_1_soft.resize(1<<nu);
    output_map_2_soft.resize(1<<nu);
    for (size_t i=0; i<1<<nu; i++) {
        output_map_1[i] = 1 & (mul(g0, (uint32_t)i) >> (nu-1));
        output_map_2[i] = 1 & (mul(g1, (uint32_t)i) >> (nu-1));
        state_inv_map[0][i] = ((i << 1) & ((1<<nu)-1));
        state_inv_map[1][i] = ((i << 1) & ((1<<nu)-1)) + 1;
        output_map_1_soft[i] = output_map_1[i] * 2 - 1;
        output_map_2_soft[i] = output_map_2[i] * 2 - 1;
    }
}

void ConvolutionalCode::encode(const bitvector &input, bitvector &output) const {
    const size_t N = input.size();
    output.resize(N*2);
    uint32_t sh = 0;
    for (size_t i=0; i<N; i++) {
        sh = (sh>>1) ^ ((uint32_t)input[i] << (nu-1));
        output[2*i+0] = output_map_1[sh];
        output[2*i+1] = output_map_2[sh];
    }
}

void ConvolutionalCode::decode(const std::vector<int> &input, size_t length, bitvector &output) const {
    const size_t N = length+nu-1;
    const size_t M = 1<<nu;
    if (input.size() < N*2) {
        output.resize(0);
        return;
    }
    bitvector bt(N*M);
    output.resize(N);
    std::vector<int32_t> cost(M*2);
    std::vector<int32_t> scores(M, 0);
    for (size_t k=0; k<N; k++) {
        for (size_t i=0; i<M; i++) {
            int edgeCost = input[2*k+0]*output_map_1_soft[i] + input[2*k+1]*output_map_2_soft[i];
            cost[2*i+0] = scores[state_inv_map[0][i]] + edgeCost;
            cost[2*i+1] = scores[state_inv_map[1][i]] + edgeCost;
        }
        for (size_t i=0; i<M; i++) {
            int a = cost[2*i+0], b = cost[2*i+1];
            bt[(k<<nu)+i] = (a<b) ? 1 : 0;
            scores[i] = (a<b) ? b : a;
        }
    }
    size_t i = (scores[0] < scores[1]) ? 1 : 0;
    for (size_t k=N; k>=1; k--) {
        size_t j = bt[((k-1)<<nu)+i];
        output[k-1] = (uint8_t)(i >> (nu-1));
        i = state_inv_map[j][i];
    }
}
