from ..graph import Output, Input, Block
from ..graph.selector import Selector
from ..mac.lowpan import Packet

class TunnelSource(Block, Selector):
    inputs = []
    outputs = [Output(Packet, ())]

    def __init__(self, utun, ll_da):
        super().__init__()
        self.utun = utun
        self.ll_da = ll_da
        self.rlist.append(utun)

    def readReady(self, fd):
        if fd is self.utun:
            datagram = fd.read()
            print('%s -> %d B' % (fd, len(datagram)))
            packet = Packet(fd.ll_addr, self.ll_da, datagram)
            if self.output((packet,)):
                self.notify()

class TunnelSink(Block):
    inputs = [Input(())]
    outputs = []

    def __init__(self, utun):
        super().__init__()
        self.utun = utun

    def process(self):
        for packet, in self.input():
            datagram = packet.tail()
            print('%5d B -> %s' % (len(datagram), self.utun))
            self.utun.write(datagram)
