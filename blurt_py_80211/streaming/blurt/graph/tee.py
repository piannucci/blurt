import queue
import typing
from .graph import Port, Block, OverrunWarning

class Tee(Block):
    inputs = [Port('T')]

    def __init__(self, n, dtype=None):
        self.outputs = [Port('T')] * n
        super().__init__()

    def process(self):
        for item, in self.iterinput():
            self.output((item,) * len(self.outputs))

class Arbiter(Block):
    outputs = [Port('T')]

    def __init__(self, n):
        self.inputs = [Port('T') for i in range(n)]
        super().__init__()

    def process(self):
        while True:
            for ip in range(len(self.inputs)):
                try:
                    item = self.input1(ip)
                except queue.Empty:
                    continue
                self.output((item,))
                break
            else:
                return

class LambdaBlock(Block):
    inputs = [Port('T')]
    outputs = [Port('S')]

    def __init__(self, itype, otype, func):
        self.T = itype
        self.S = otype
        self.func = func
        super().__init__()

    def process(self):
        for item, in self.iterinput():
            self.output((self.func(item),))
