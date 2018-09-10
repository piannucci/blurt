#!/usr/bin/env python
import numpy as np

scrambler = np.zeros((128,127), np.uint8)
s = np.arange(128, dtype=np.uint8)
for i in range(127):
    s = np.uint8((s << 1) ^ (1 & ((s >> 3) ^ (s >> 6))))
    scrambler[:,i] = s & 1

scrambler_state_lookup = (scrambler[:,:7] << np.arange(6,-1,-1)).sum(1).argsort()

pilot_sequence = scrambler[0x7f]

def scramble(bits, state):
    return bits ^ np.resize(scrambler[state], bits.size)

def descramble(bits):
    return scramble(bits, scrambler_state_lookup[(bits[:7] << np.arange(6,-1,-1)).sum()])
