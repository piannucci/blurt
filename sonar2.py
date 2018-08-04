#!/usr/bin/env python
import audio
import numpy as np
import pylab as pl
import sys
import time

# http://poincare.matf.bg.ac.rs/~ezivkovm/publications/primpol1.pdf
polys = {
    2: (1,),
    3: (1,),
    4: (1,),
    5: (2,),
    6: (1,),
    7: (1,),
    8: (1, 2, 7),
    9: (4,),
    10: (3,),
    11: (2,),
    12: (1, 2, 10),
    13: (3, 5, 8),
    14: (1, 11, 12),
    15: (1,),
    16: (10, 12, 15),
    17: (3,),
}

def mls(n=13):
    assert n in polys, "No primitive polynomial for n=%d" % n
    poly = (0, n) + polys[n]
    poly = np.array(poly)
    r = 1
    m = np.empty((1<<n)-1, np.uint8)
    for i in range((1<<n)-1):
        m[i] = r & 1
        r = (r>>1) | (np.bitwise_xor.reduce(r >> poly) & 1)<<(n-1)
    return m

class SonarTransciever:
    def __init__(self, m):
        self.period = m.size*2
        m = m.astype(np.float32) * 2 - 1
        m = np.concatenate((np.r_[m, m][:,None], np.r_[m, -m][:,None]), 1)
        m *= 1e-3
        self.inBufSize = 2048
        self.outBufSize = 2048
        targetLength = self.outBufSize+self.period-1
        self.output = np.tile(m, ((targetLength+m.shape[0]-1) // m.shape[0], 1))
        self.input_fragment = np.zeros(0, np.complex64)
    def __len__(self):
        return sys.maxsize
    def __getitem__(self, sl):
        start = sl.start % self.period
        count = sl.stop - sl.start
        return self.output[start:start+count]
    def append(self, sequence):
        stream = np.r_[self.input_fragment, sequence]
        lookahead = self.pulse_duration-1 # needed for convolution
        n = max(0, (stream.size-lookahead) // self.period)
        advance = n*self.period
        if n > 0:
            y = stream[:(n+1)*advance+lookahead]
        self.input_fragment = stream[advance:]
        self.i += advance

n = 11
Fs = 96e3
print('Period: %.1f ms' % (1000*2*((1<<n)-1)/Fs,))
xcvr = SonarTransciever(mls(n))
ap = audio.AudioInterface()
try:
    ap.play(xcvr, Fs)
    ap.record(xcvr, Fs)
    while True:
        time.sleep(1.)
except KeyboardInterrupt:
    ap.stop()
