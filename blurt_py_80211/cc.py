#!/usr/bin/env python
import numpy as np
from scipy import weave
from scipy.weave import converters
import util

# This file implements the 802.11 convolutional code.  The convolutional encoder
# and Viterbi decoder are optimized in terms of look-up tables and in-line C++.

standard_cc = (7,0133,0171)

# Puncturing schedules
matrix_1_2 = np.array([1,1])
matrix_2_3 = np.array([1,1,1,0])
matrix_3_4 = np.array([1,1,1,0,0,1])
matrix_5_6 = np.array([1,1,1,0,0,1,1,0,0,1])
matrix_7_8 = np.array([1,1,1,0,1,0,1,0,0,1,1,0,0,1])
matrix_1_1 = np.array([0,1,1,0])
matrix_3_2 = np.array([0,0,1,0,1,0])
matrix_2_1 = np.array([0,0,0,1,0,0,1,0])
matrices = [matrix_1_2, matrix_2_3, matrix_3_4, matrix_5_6, matrix_7_8, matrix_1_1, matrix_3_2, matrix_2_1]
rates = [(1,2), (2,3), (3,4), (5,6), (7,8), (1,1), (3,2), (2,1)]
puncturingSchedule = dict(zip(rates, zip(rates, matrices)))

class ConvolutionalCode:
    def __init__(self, code=standard_cc):
        self.nu = code[0]
        self.g0 = util.rev(code[1], self.nu)
        self.g1 = util.rev(code[2], self.nu)
        self.output_map_1      = np.uint8(1 & (util.mul(self.g0, np.arange(1<<self.nu)) >> (self.nu-1)))
        self.output_map_2      = np.uint8(1 & (util.mul(self.g1, np.arange(1<<self.nu)) >> (self.nu-1)))
        self.states            = np.arange(1<<self.nu)
        self.state_inv_map     = ((np.arange(1<<self.nu) << 1) & ((1<<self.nu)-1))[:,np.newaxis] + np.array([0,1])[np.newaxis,:]
        self.state_inv_map_tag = np.tile(np.arange(1<<self.nu)[:,np.newaxis] >> (self.nu-1), (1,2))
        self.output_map_1_soft = self.output_map_1.astype(int)[np.newaxis,:] * 2 - 1
        self.output_map_2_soft = self.output_map_2.astype(int)[np.newaxis,:] * 2 - 1
    def encode(self, input):
        """Encode a sequence of input bits."""
        # Equivalent to the following Python code:
        #    output = empty((input.size,2), uint8)
        #    sh = int64(0)
        #    x = int64(input << (nu-1))
        #    for i in xrange(x.size):
        #        sh = (sh>>1) ^ x[i]
        #        output[i,0] = output_map_1[sh]
        #        output[i,1] = output_map_2[sh]
        #    return output.flatten()
        nu, output_map_1, output_map_2 = self.nu, self.output_map_1, self.output_map_2
        output = np.empty(input.size*2, np.uint8)
        N = input.size
        code = """
        uint8_t sh;
        for (int i=0; i<N; i++) {
            sh = (sh>>1) ^ ((int)input(i) << (nu-1));
            output(2*i+0) = output_map_1(sh);
            output(2*i+1) = output_map_2(sh);
        }
        """
        weave.inline(code, ['N','nu','input','output', 'output_map_1','output_map_2'], type_converters=converters.blitz)
        return output.flatten()
    def decode(self, input, length):
        """Decode a sequence of log likelihood ratios.  Length does not include
        tail or padding."""
        # Equivalent to the following Python code:
        #    N = length+nu-1
        #    if input.size < N*2:
        #        return zeros(length, int)
        #    scores = zeros(1<<nu, int64)
        #    bt = [None] * N
        #    x = input[0:2*N:2,newaxis]*output_map_1_soft + input[1:2*N:2,newaxis]*output_map_2_soft
        #    for k in xrange(N):
        #        cost = scores[state_inv_map] + x[k]
        #        idxs = cost.argmax(1)
        #        scores = cost[states,idxs]
        #        bt[k] = idxs
        #    msg = empty(N, uint8)
        #    i = scores[:2].argmax()
        #    for k in xrange(N-1, -1, -1):
        #        msg[k] = state_inv_map_tag[i,bt[k][i]]
        #        i = state_inv_map[i,bt[k][i]]
        #    return msg
        nu, state_inv_map, state_inv_map_tag = self.nu, self.state_inv_map, self.state_inv_map_tag
        N = int(length+nu-1)
        M = 1<<nu
        if input.size < N*2:
            return np.zeros(length, int)
        bt = np.empty((N, M), np.uint8);
        x = input[0:2*N:2,np.newaxis]*self.output_map_1_soft + \
            input[1:2*N:2,np.newaxis]*self.output_map_2_soft
        msg = np.empty(N, np.uint8)
        code = """
        int64_t *cost = new int64_t [M*2];
        int64_t *scores = new int64_t [M];
        for (int i=0; i<M; i++)
            scores[i] = 0;
        for (int k=0; k<N; k++) {
            for (int i=0; i<M; i++) {
                cost[2*i+0] = scores[state_inv_map(i, 0)] + x(k, i);
                cost[2*i+1] = scores[state_inv_map(i, 1)] + x(k, i);
            }
            for (int i=0; i<M; i++) {
                int a, b;
                a = cost[2*i+0];
                b = cost[2*i+1];
                bt(k, i) = (a<b) ? 1 : 0;
                scores[i] = (a<b) ? b : a;
            }
        }
        int i = (scores[0] < scores[1]) ? 1 : 0;
        for (int k=N-1; k>=0; k--) {
            int j = bt(k, i);
            msg(k) = state_inv_map_tag(i,j);
            i = state_inv_map(i,j);
        }
        delete [] cost;
        delete [] scores;
        """
        weave.inline(code, ['N','M','state_inv_map', 'x','bt','state_inv_map_tag', 'msg'], type_converters=converters.blitz)
        return msg
    def puncture(self, input, m):
        return input[np.where(np.tile(m, (input.size+m.size-1)/m.size)[:input.size])]
    def depuncture(self, input, m):
        output = np.zeros(((input.size + m.sum() - 1) / m.sum()) * m.size, input.dtype)
        i = np.where(np.tile(m, (output.size+m.size-1)/m.size)[:output.size])
        output[i] = input
        return output

__all__ = ['ConvolutionalCode', 'puncturingSchedule']
