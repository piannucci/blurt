#!/usr/bin/env python
import numpy as np

def upsample(x, n):
    M = x.size
    N = 1 << np.where(M <= (1<<np.arange(24)))[0][0]
    pad = N - M
    X = np.fft.fft(np.r_[x, np.zeros(pad)])
    X = np.r_[X[:N/2], np.zeros(int((n-1)*N)), X[N/2:]]
    return np.fft.ifft(X)[:int(M*n)]

def rev(x,n):
    """Reverse the bits in the n-bit number x."""
    y = 0
    for i in xrange(n):
        y <<= 1
        y |= 1 * np.array(x&1, bool)
        x >>= 1
    return y

def mul(a, b):
    """Polynomial multiplication in Z2[x]."""
    b = np.copy(b)
    c = 0
    i = 0
    while np.any(b):
        c ^= (a * np.array(b&1, bool)) << i
        b >>= 1
        i += 1
    return c

def shiftin(input, noutput):
    """Convert an array of bits into an array of noutput-bit numbers."""
    return (input.reshape(input.size/noutput, noutput) << np.arange(noutput)[np.newaxis,:]).sum(1)

def shiftout(input, ninput):
    """Convert an array of ninput-bit numbers into an array of bits."""
    return ((input.flatten()[:,np.newaxis] >> np.arange(ninput)[np.newaxis,:]) & 1).flatten()

def truncate(x, i):
    while i>0:
        i -= 1
        yield x.next()
