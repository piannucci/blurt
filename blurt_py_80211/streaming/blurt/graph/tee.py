import queue
import warnings
import typing
from .graph import Port, Block, OverrunWarning

class Tee(Block):
    _T = typing.TypeVar('T')
    inputs = [Port(_T)]

    def __init__(self, n, dtype=None):
        self.outputs = [Port(self._T)] * n
        super().__init__()

    def process(self):
        for item, in self.iterinput():
            self.output((item,) * len(self.output_queues))

class Arbiter(Block):
    _T = typing.TypeVar('T')
    outputs = [Port(_T)]

    def __init__(self, n):
        self.inputs = [Port(self._T) for i in range(n)]
        super().__init__()

    def process(self):
        while True:
            for iq in self.input_queues:
                try:
                    item = iq.get_nowait()
                except queue.Empty:
                    continue
                self.output((item,))
                break
            else:
                return

class LambdaBlock(Block):
    _T = typing.TypeVar('T')
    _S = typing.TypeVar('S')
    inputs = [Port(_T)]
    outputs = [Port(_S)]

    def __init__(self, itype, otype, func):
        self.T = itype
        self.S = otype
        self.func = func
        super().__init__()

    def process(self):
        for item, in self.iterinput():
            self.output((self.func(item),))
