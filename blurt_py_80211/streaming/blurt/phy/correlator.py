import numpy as np
from . import ofdm

class Autocorrelator:
    def __init__(self, nChannelsPerFrame, quantum, width):
        self.y_hist = np.zeros((0, nChannelsPerFrame))
        self.nChannelsPerFrame = nChannelsPerFrame
        self.quantum = quantum
        self.width = width
    def process(self, y):
        y = np.r_[self.y_hist, y]
        count_needed = y.shape[0] // self.quantum * self.quantum
        count_consumed = count_needed - self.quantum * self.width
        if count_consumed <= 0:
            self.y_hist = y
        else:
            self.y_hist = y[count_consumed:]
            y = y[:count_needed].reshape(-1, self.quantum, self.nChannelsPerFrame)
            corr_sum = np.abs((y[:-1].conj() * y[1:]).sum(1)).cumsum(0)
            yield (corr_sum[self.width-1:] - corr_sum[:-self.width+1]).mean(-1)

class PeakDetector:
    def __init__(self, l, quantum):
        self.y_hist = np.zeros(l)
        self.l = l
        self.i = 1
        self.quantum = quantum
    def process(self, y):
        y = np.r_[self.y_hist, y]
        count_needed = y.size
        count_consumed = count_needed - 2*self.l
        if count_consumed <= 0:
            self.y_hist = y
        else:
            self.y_hist = y[count_consumed:]
            stripes_shape = (2*self.l+1, count_needed-2*self.l)
            stripes_strides = (y.strides[0],)*2
            stripes = np.lib.stride_tricks.as_strided(y, stripes_shape, stripes_strides)
            yield from ((stripes.argmax(0) == self.l).nonzero()[0] + self.i) * self.quantum
            self.i += count_consumed

class Clause18Detector(PeakDetector):
    def __init__(self, nChannelsPerFrame, oversample):
        N_sts_period = ofdm.L.nfft // 4
        N_sts_samples = ofdm.L.ts_reps * (ofdm.L.ncp + ofdm.L.nfft)
        quantum = N_sts_period * oversample
        width = N_sts_samples // N_sts_period
        self.ac = Autocorrelator(nChannelsPerFrame, quantum, width)
        super().__init__(9, quantum)
    def process(self, y):
        for a in self.ac.process(y):
            yield from super().process(a)
