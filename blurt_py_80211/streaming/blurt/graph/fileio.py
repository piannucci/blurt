import os
import warnings
import time
import pickle
from .graph import Input, Output, Block

class FileSink(Block):
    inputs = [Input('shape')]
    outputs = []

    def __init__(self, fn):
        super().__init__()
        self.f = open(fn, 'wb')

    def process(self):
        for item, in self.input():
            pickle.dump(item, self.f, -1)

class FileSource(Block):
    # TODO don't load entire file into memory when rate is unlimited
    def __init__(self, fn, dtype, shape, ratelimit=None, timeGranularity=.01):
        self.inputs = []
        self.outputs = [Output(dtype, shape)]
        super().__init__()
        if ratelimit is None:
            st = os.stat(fn)
            if st.st_size > 1e9:
                warnings.warn('On graph startup, FileSource will load %d bytes of pickled data into memory' % st.st_size)
        else:
            self.maxQueue = int(ratelimit * timeGranularity * 1.5 + 1)
        self.f = open(fn, 'rb')
        self.ratelimit = ratelimit
        self.timeGranularity = timeGranularity

    def start(self):
        super().start()
        if self.ratelimit is not None:
            self.tokens = 0
            self.lastFired = self.startTime = time.monotonic()
            self._timerFired()
        else:
            while True:
                try:
                    self.output_queues[0].put(pickle.load(self.f))
                except EOFError:
                    break

    def stop(self):
        if self.ratelimit is not None:
            self.runloop.removeTimer(self.timer)
        super().stop()

    def _timerFired(self):
        now = time.monotonic()
        dt = now - self.lastFired
        self.lastFired = now
        self.tokens = min(
            self.tokens + dt * self.ratelimit,
            self.maxQueue - self.out_queues[0].qsize(),
        )
        while self.tokens >= 1:
            try:
                self.output_queues[0].put(pickle.load(self.f))
            except EOFError:
                self.output_queues[0].closed = True
                return
            self.tokens -= 1
        self.timer = self.runloop.addTimer(self._timerFired, delay=self.timeGranularity)
