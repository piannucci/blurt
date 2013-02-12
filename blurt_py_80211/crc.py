#!/usr/bin/env python
import numpy as np
import util

def remainder(a, b):
    # clobbers input!
    for i in xrange(a.size-b.size+1):
        if a[i]:
            a[i:i+b.size] ^= b
    return a[1-b.size:]

# The FCS field is transmitted commencing with the coefficient of the highest-order term.
# polynomial from x^32 down to x^0: 0b100000100110000010001110110110111
CRC32_802_11_FCS_G = 0b100000100110000010001110110110111
CRC32_802_11_FCS_remainder = 0b11000111000001001101110101111011

def FCS(calculationFields):
    k = calculationFields.size
    G = util.shiftout(np.r_[CRC32_802_11_FCS_G], 33)[::-1]
    return 1 & ~remainder(np.r_[np.ones(32, int), np.zeros(k, int)] ^ np.r_[calculationFields, np.zeros(32, int)], G)

def checkFCS(frame):
    k = frame.size
    G = util.shiftout(np.r_[CRC32_802_11_FCS_G], 33)[::-1]
    x = remainder(np.r_[np.ones(32, int), np.zeros(k, int)] ^ np.r_[frame, np.zeros(32, int)], G)
    return all(x == util.shiftout(np.r_[CRC32_802_11_FCS_remainder], 32)[::-1])
