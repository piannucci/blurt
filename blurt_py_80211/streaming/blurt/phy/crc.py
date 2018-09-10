#!/usr/bin/env python
import numpy as np
from . import kernels

class CRC:
    def __init__(self, M, G, crc_pass, L=16):
        self.M = M
        self.L = L
        self.crc_pass = crc_pass
        lut = np.arange(1<<L, dtype=np.uint64) << M
        for i in range(47,31,-1): # uh...
            lut ^= (G << (i-M)) * ((lut >> i) & 1)
        self.lut = lut.astype(np.uint32)

    def crc(self, x):
        a = np.r_[np.ones(self.M, int), np.zeros(x.size, int)] ^ np.r_[x, np.zeros(self.M, int)]
        a = np.r_[np.zeros(-a.size % self.L, int), a]
        a = (a.reshape(-1, self.L) << np.arange(self.L)[::-1]).sum(1)
        return kernels.crc(a, self.lut)

    def compute(self, x):
        return (~self.crc(x) >> np.arange(self.M)[::-1]) & 1

    def check(self, x):
        return self.crc(x) == self.crc_pass

# The FCS field is transmitted commencing with the coefficient of the highest-order term.
CRC32_802_11_FCS = CRC(32, 0x104c11db7, 0xc704dd7b)
CRC8_HT_SIG      = CRC(8, 0x107, 0)
CRC16_16_2_3_7   = CRC(16, 0x11021, 0)
