import os
import warnings
import time
import pickle
import typing
from .graph import Port, Block

class FileSink(Block):
    inputs = [Port(typing.Any)]
    outputs = []

    def __init__(self, fn):
        super().__init__()
        self.f = open(fn, 'wb')

    def process(self):
        for item, in self.iterinput():
            pickle.dump(item, self.f, -1)

class FileSource(Block):
    # TODO don't load entire file into memory when rate is unlimited
    def __init__(self, fn, itemtype, ratelimit=None, timeGranularity=.01):
        self.inputs = []
        self.outputs = [Port(itemtype)]
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
                    self.output1(0, pickle.load(self.f))
                except EOFError:
                    break

    def stopped(self):
        if self.ratelimit is not None:
            self.runloop.removeTimer(self.timer)
        super().stopped()

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
                self.output1(0, pickle.load(self.f))
            except EOFError:
                self.closeOutput()
                return
            self.tokens -= 1
        self.timer = self.runloop.addTimer(self._timerFired, delay=self.timeGranularity)
