# adapter from audio stream to asynchronous graph
import warnings
import numpy as np
import typing
from ..graph import Output, Input, Block, OverrunWarning, UnderrunWarning

class InStream_SourceBlock(IOStream, Block):
    inputs = []
    outputs = [Output(typing.Tuple[np.ndarray, int, int], ('nChannelsPerFrame',))]

    # IOStream methods

    def write(self, frames, inputTime, now):
        if self.output_queues[0].closed:
            return
        try:
            self.output_queues[0].put_nowait((frames, inputTime, now))
            self.notify()
        except queue.Full:
            warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)

    def inDone(self):
        return self.output_queues[0].closed

    def stop(self):
        self.closeOutput()

    # Block methods

    def process(self):
        pass

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

    def stop(self):
        self.closeInput()

    # Block methods

    def process(self):
        pass
