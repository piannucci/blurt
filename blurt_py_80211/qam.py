#!/usr/bin/env python
import numpy as np
import util

def grayRevToBinary(x, n):
    y = 0*x
    for i in range(n):
        y <<= 1
        y |= x&1
        x >>= 1
    shift = 1
    while shift<n:
        y ^= y >> shift
        shift<<=1
    return y

class Constellation:
    def __init__(self, Nbpsc, symbols):
        self.Nbpsc = Nbpsc
        self.symbols = symbols

def qam_constellation(Nbpsc):
    scale = 1.
    n = 1
    if Nbpsc != 1:
        # want variance = .5 per channel
        # in fact, var{ 2 range(2^{Nbpsc/2}) - const } = (2^Nbpsc - 1) / 3
        scale = ( 1.5 / ((1<<Nbpsc) - 1) ) ** .5
        n = Nbpsc >> 1
    symbols = (2*grayRevToBinary(np.arange(1<<n), n) + 1 - (1<<n)) * scale
    if Nbpsc != 1:
        symbols = np.tile(symbols, 1<<n) + 1j*np.repeat(symbols, 1<<n)
    return Constellation(Nbpsc, symbols)

bpsk = qam_constellation(1)
qpsk = qam_constellation(2)
qam16 = qam_constellation(4)
qam64 = qam_constellation(6)

def encode(interleaved_bits, rate):
    return rate.constellation[util.shiftin(interleaved_bits, rate.Nbpsc)]

def demapper(data, constellation, dispersion, n):
    squared_distance = np.abs(constellation[np.newaxis,:] - data[:,np.newaxis])**2
    ll = -np.log(np.pi * dispersion) - squared_distance / dispersion
    ll -= np.logaddexp.reduce(ll, 1)[:,np.newaxis]
    j = np.arange(1<<n)
    ll0 = np.zeros((data.shape[0], n), float)
    ll1 = np.zeros((data.shape[0], n), float)
    for i in xrange(n):
        idx0 = np.where(0 == (j & (1<<i)))[0]
        idx1 = np.where(j & (1<<i))[0]
        ll0[:,i] = np.logaddexp.reduce(ll[:,idx0], 1)
        ll1[:,i] = np.logaddexp.reduce(ll[:,idx1], 1)
    return np.int64(np.clip(10.*(ll1-ll0), -1e4, 1e4)).flatten()
