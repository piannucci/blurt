from .graph import Input, Output, Block
import pickle

class FileSink(Block):
    inputs = [Input('shape')]
    outputs = []

    def __init__(self, fn):
        super().__init__()
        self.f = open(fn, 'wb')

    def process(self):
        for item, in self.input():
            pickle.dump(item, self.f, -1)

# import numpy as np
# import threading
# from .threaded import SelectorBlock
#
# class FileSource(SelectorBlock):
#     def __init__(self, fn, dtype, itemshape=(1,)):
#         self.inputs = []
#         self.outputs = [Output(dtype, itemshape)]
#         itemsize = np.product(itemshape)
#         self.f = np.memmap(fn, dtype, 'r')
#         self.f = iter(self.f[:self.f.size//itemsize*itemsize].reshape((-1,) + itemshape))
#         self.stopping = False
#         self.running = False
#         super().__init__()
#
#     def start(self):
#         self.output_queues[0].maxsize = 64
#         with self.cv
#
#     def process(self):
#         while not self.output_queues[0].full():
#             try:
#                 item = next(self.f)
#             except StopIteration:
#                 self.stop()
#                 break
#             self.output_queues[0].put_nowait(item)
