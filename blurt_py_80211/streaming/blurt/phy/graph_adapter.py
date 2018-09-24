from typing import Tuple
import collections
import numpy as np
from ..graph import Port, Block
from ..graph.typing import Array
from . import iir

class GenericDecoderBlock(Block):
    """
    A graph block that performs downconversion, downsampling, decoder
    instantiation, and lookback.
    """
    inputs = [Port(Tuple[Array[[None, 'nChannelsPerFrame'], np.float32], float, float])]
    outputs = [Port(Tuple[Array[[None], np.uint8], float])]

    def __init__(self, channel, detector_class, decoder_class):
        """
        class Detector:
            def __init__(self, nChannelsPerFrame, channel):
                ...
            def process(self, y):
                yield from (synchronization indices in input stream)
        
        class Decoder:
            def __init__(self, start_index, nChannelsPerFrame, oversample):
                ...
            def process(y, k):
                # Return None for "keep going", false-ish for "decoding
                # failed", or (datagram, snr).  Once the result is non-None,
                # return the same result if called again.
        """
        super().__init__()
        self.channel = channel
        self.intermediate_upsample = 1
        self.detector_class = detector_class
        self.decoder_class = decoder_class

    def start(self):
        super().start()
        self.detector = self.detector_class(self.nChannelsPerFrame, self.intermediate_upsample)
        self.decoders = []
        self.lookback = collections.deque()
        self.k_current = 0
        self.k_lookback = 0
        self.lp = iir.IIRFilter.lowpass(.45/self.channel.upsample_factor, shape=(None, self.nChannelsPerFrame), axis=0)
        self.i = 0

    def process(self):
        channel = self.channel
        Omega = 2*np.pi*channel.Fc/channel.Fs
        upsample_factor = channel.upsample_factor
        for (y, inputTime, now), in self.iterinput():
            nFrames = y.shape[0]
            y = self.lp(y * np.exp(-1j*Omega * np.r_[self.i:self.i+nFrames])[:,None])
            y = y[-self.i%upsample_factor::upsample_factor]
            self.i += nFrames
            for peak in self.detector.process(y):
                d = self.decoder_class(peak, self.nChannelsPerFrame, self.intermediate_upsample)
                k = self.k_lookback
                for y_old in self.lookback:
                    d.process(y_old, k)
                    k += y_old.shape[0]
                self.decoders.append(d)
            self.lookback.append(y)
            for d in list(self.decoders):
                result = d.process(y, self.k_current)
                if result:
                    print('recv psdu (%3d bytes, %2.5f dB)' % (len(result[0]), result[1]))
                    self.output((result,))
                if result is not None:
                    self.decoders.remove(d)
            self.k_current += y.shape[0]
            while self.lookback:
                if self.k_lookback + self.lookback[0].shape[0] >= self.k_current - 1024:
                    break
                self.k_lookback += self.lookback.popleft().shape[0]
