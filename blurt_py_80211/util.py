#!/usr/bin/env python
import numpy as np
import wave

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
    for i in range(n):
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
        yield next(x)

def papr(x):
    m = np.abs(x)**2
    return m.max()/m.mean()

def readwave(fn):
    f = wave.open(fn)
    Fs = f.getframerate()
    assert f.getcompname() == 'not compressed' and f.getcomptype() == 'NONE', 'Bad WAV type'
    channels = f.getnchannels()
    nframes = f.getnframes()
    dtype = [None, np.uint8, np.int16, None, np.int32][f.getsampwidth()]
    input = f.readframes(nframes)
    input = np.fromstring(input, np.uint8)
    if f.getsampwidth() == 3:
        input = np.hstack((0x80 + np.zeros((input.size//3, 1), dtype=np.uint8), input.reshape(input.size//3, 3))).flatten()
        dtype = np.int32
    input = input.view(dtype=dtype).astype(float)
    nframes = input.size // channels
    input = input.reshape(nframes, channels).mean(1)
    f.close()
    return input, Fs

def writewave(fn, x, Fs, bytesPerSample=3):
    nframes, nchannels = x.shape
    x = x.flatten()
    if x.dtype.kind == 'f':
        x = (x * ((1<<31) - 2)).round()
    x = x.astype(np.int32)
    x = x.view(dtype=np.uint8)
    x = x.reshape(nframes*nchannels, 4)[:,-bytesPerSample:].flatten().tostring()
    f = wave.open(fn, 'w')
    f.setnchannels(nchannels)
    f.setsampwidth(bytesPerSample)
    f.setframerate(Fs)
    f.setnframes(nframes)
    f.setcomptype('NONE', 'not compressed')
    f.writeframes(x)
    f.close()
