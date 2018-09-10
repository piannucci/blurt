import typing
import numpy as np
from ..graph import Output, Input, Block

class PHYLoopbackBlock(Block): # given an encoder input, produce a fake decoder output
    inputs = [Input(())]
    outputs = [Output(typing.Tuple[np.ndarray, float], ())]

    def process(self):
        for item, in self.input():
            self.output(((item, 0),))
