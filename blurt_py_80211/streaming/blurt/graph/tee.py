import queue
import warnings
from .graph import Output, Input, Block, OverrunWarning

class Tee(Block):
    inputs = [Input('shape')]

    def __init__(self, n, dtype=None):
        self.outputs = [Output(dtype, 'shape')] * n
        super().__init__()

    def process(self):
        for item, in self.input():
            self.output((item,) * len(self.output_queues))

class Arbiter(Block):
    def __init__(self, n, dtype=None):
        self.inputs = [Input('shape') for i in range(n)]
        self.outputs = [Output(dtype, 'shape')]
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
    def __init__(self, idtype, ishape, odtype, oshape, fn):
        self.inputs = [Input(ishape)]
        self.outputs = [Output(odtype, oshape)]
        self.fn = fn
        super().__init__()

    def process(self):
        for item, in self.input():
            self.output((self.fn(item),))
