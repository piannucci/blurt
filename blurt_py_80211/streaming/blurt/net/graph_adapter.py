import sys
from ..graph import Port, Block
from ..mac.lowpan import Packet

class TunnelSource(Block):
    inputs = []
    outputs = [Port(Packet)]

    def __init__(self, utun, ll_da):
        super().__init__()
        self.utun = utun
        self.ll_da = ll_da

    def start(self):
        super().start()
        self.runloop.rlist[self.utun] = self._readHandler

    def stop(self):
        del self.runloop.rlist[self.utun]
        super().stop()

    def _readHandler(self):
        datagram = self.utun.read()
        print('%s -> %d B' % (self.utun, len(datagram)), file=sys.stderr)
        packet = Packet(self.utun.ll_addr, self.ll_da, datagram)
        if self.output((packet,)):
            self.notify()

class TunnelSink(Block):
    inputs = [Port(Packet)]
    outputs = []

    def __init__(self, utun):
        super().__init__()
        self.utun = utun

    def process(self):
        for packet, in self.iterinput():
            datagram = packet.tail()
            print('%5d B -> %s' % (len(datagram), self.utun), file=sys.stderr)
            self.utun.write(datagram)
