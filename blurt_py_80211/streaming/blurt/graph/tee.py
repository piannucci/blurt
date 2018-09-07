import queue
import warnings
from .graph import Output, Input, Block, OverrunWarning

class Tee(Block):
    def __init__(self, n, dtype=None):
        self.inputs = [Input('shape')]
        self.outputs = [Output(dtype, 'shape')]
        super().__init__()

    def process(self):
        for item, in self.input():
            self.output([item for oq in self.output_queues])

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
