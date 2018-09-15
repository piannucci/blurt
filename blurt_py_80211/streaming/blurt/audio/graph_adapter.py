# adapter from audio stream to asynchronous graph
import warnings
import numpy as np
import typing
import queue
from . import AudioHardware as AH
from ..graph import Output, Input, Block, UnderrunWarning
from .stream import IOStream

class InStream_SourceBlock(IOStream, Block):
    inputs = []
    outputs = [Output(typing.Tuple[np.ndarray, int, int], ('nChannelsPerFrame',))]

    def __init__(self, ios):
        self.ios = ios
        self.nChannelsPerFrame = self.ios.nChannelsPerFrame(AH.kAudioObjectPropertyScopeInput)
        super().__init__()

    # IOStream methods

    def write(self, frames, inputTime, now):
        if self.output_queues[0].closed:
            return
        self.output(((frames, inputTime, now),))
        self.notify()

    def inDone(self):
        return self.output_queues[0].closed

class OutStream_SinkBlock(IOStream, Block):
    inputs = [Input(('nChannelsPerFrame',))]
    outputs = []

    def __init__(self):
        super().__init__()
        self.outFragment = None
        self.warnOnUnderrun = True

    # IOStream methods

    def read(self, nFrames, outputTime, now):
        result = np.empty((nFrames, self.nChannelsPerFrame), np.float32)
        i = 0
        if self.outFragment is not None:
            n = min(self.outFragment.shape[0], nFrames)
            result[:n] = self.outFragment[:n]
            i += n
            if n < self.outFragment.shape[0]:
                self.outFragment = self.outFragment[n:]
            else:
                self.outFragment = None
        while i < nFrames:
            try:
                fragment = self.input_queues[0].get_nowait()
            except queue.Empty:
                result[i:] = 0
                if self.warnOnUnderrun:
                    warnings.warn('%s underrun' % self.__class__.__name__, UnderrunWarning)
                break
            if fragment.ndim != 2 or fragment.shape[1] != self.nChannelsPerFrame:
                raise ValueError('shape mismatch')
            n = min(nFrames-i, fragment.shape[0])
            result[i:i+n] = fragment[:n]
            i += n
            if fragment.shape[0] > n:
                self.outFragment = fragment[n:]
        return result

    def outDone(self):
        return self.input_queues[0].closed

class IOSession_Block(Block):
    inputs = []
    outputs = []

    def __init__(self, ios):
        super().__init__()
        self.ios = ios

    def start(self):
        super().start()
        self.ios.start()

    def stop(self):
        self.ios.stop()
        super().stop()

# software loopback
class AudioBypass_Block(Block):
    inputs = [Input(('nChannelsPerFrame',))]
    outputs = [Output(typing.Tuple[np.ndarray, int, int], ('nChannelsPerFrame',))]

    def process(self):
        for frames, in self.input():
            self.output(((frames, 0, 0),))
