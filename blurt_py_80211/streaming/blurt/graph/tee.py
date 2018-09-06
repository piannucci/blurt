import queue
import warnings
from .graph import Output, Input, Block, OverrunWarning

class Tee(Block):
    def __init__(self, n, dtype=None):
        self.inputs = [Input('shape')]
        self.outputs = [Output(dtype, 'shape')]
        super().__init__()

    def process(self):
        while True:
            try:
                item = self.input_queues[0].get_nowait()
            except queue.Empty:
                break
            for oq in self.output_queues:
                try:
                    oq.put_nowait(item)
                except queue.Full:
                    warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)

class Arbiter(Block):
    def __init__(self, n, dtype=None):
        self.inputs = [Input('shape') for i in range(n)]
        self.outputs = [Output(dtype, 'shape')]
        super().__init__()

    def process(self):
        done = True
        while not done:
            done = True
            for iq in self.input_queues:
                try:
                    item = iq.get_nowait()
                except queue.Empty:
                    continue
                done = False
                try:
                    self.output_queues[0].put_nowait(item)
                except queue.Full:
                    warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)
