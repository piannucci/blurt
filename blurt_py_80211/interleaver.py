#!/usr/bin/env python
import numpy as np

def interleave_permutation(Ncbps, Nbpsc):
    s = max(Nbpsc/2, 1)
    k = np.arange(Ncbps)
    i = (Ncbps/16) * (k % 16) + (k/16)
    j = s * (i/s) + (i + Ncbps - (16 * i / Ncbps)) % s
    return j

def interleave_inverse_permutation(Ncbps, Nbpsc):
    s = max(Nbpsc/2, 1)
    j = np.arange(Ncbps)
    i = s * (j/s) + (j + (16*j/Ncbps)) % s
    k = 16 * i - (Ncbps - 1) * (16 * i / Ncbps)
    return k

def interleave(input, Ncbps, Nbpsc, reverse=False):
    symbols = input.size / Ncbps
    input = input[:symbols*Ncbps].reshape(symbols, Ncbps)
    if reverse:
        p = interleave_permutation(Ncbps, Nbpsc)
    else:
        p = interleave_inverse_permutation(Ncbps, Nbpsc)
    return input[:,p].flatten()

