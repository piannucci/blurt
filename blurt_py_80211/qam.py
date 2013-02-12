#!/usr/bin/env python
import numpy as np
import util

bpsk  = np.array([-1, 1]) * 1.
qpsk  = np.array([-1, 1]) / 2.**.5
qam16 = np.array([-3, -1, 3, 1]) / 10.**.5
qam64 = np.array([-7, -5, -1, -3, 7, 5, 1, 3]) / 42.**.5

qpsk  = qpsk [np.arange( 4) & 1] * 1j + qpsk [np.arange( 4) >> 1]
qam16 = qam16[np.arange(16) & 3] * 1j + qam16[np.arange(16) >> 2]
qam64 = qam64[np.arange(64) & 7] * 1j + qam64[np.arange(64) >> 3]

bpsk  = bpsk [util.rev(np.arange(1<<1), 1)]
qpsk  = qpsk [util.rev(np.arange(1<<2), 2)]
qam16 = qam16[util.rev(np.arange(1<<4), 4)]
qam64 = qam64[util.rev(np.arange(1<<6), 6)]

bpsk = (1, bpsk)
qpsk = (2, qpsk)
qam16 = (4, qam16)
qam64 = (6, qam64)

def encode(interleaved_bits, rate, Nsc):
    result = rate.constellation[util.shiftin(interleaved_bits, rate.Nbpsc)]
    return result.reshape(result.size/Nsc, Nsc)

def demapper(data, constellation, min_dist, dispersion, n):
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
    return np.int64(np.clip(10.*(ll1-ll0), -1e4, 1e4))
