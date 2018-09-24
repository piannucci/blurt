import typing
import numpy as np
from ..graph import Port, Block

class PHYLoopbackBlock(Block): # given an encoder input, produce a fake decoder output
    inputs = [Port(Array[[None], np.uint8])]
    outputs = [Port(Tuple[Array[[None], np.uint8], float])]

    def process(self):
        for item, in self.input():
            self.output(((item, 0),))
