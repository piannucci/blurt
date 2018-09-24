import warnings
import queue
import numpy as np
from typing import Tuple
from ..graph.typing import Array
from ..graph import Port, Block, OverrunWarning
from .lowpan import Packet

class FragmentationBlock(Block):
    inputs = [Port(Packet)]
    outputs = [Port(Array[[None], np.uint8])]

    def __init__(self, pdb):
        super().__init__()
        self.pdb = pdb
        self.pdb.sendMPDU = self._sendMPDU

    def process(self):
        for packet, in self.iterinput():
            self.pdb.recvMSDU(packet)

    def _sendMPDU(self, mpdu : np.ndarray):
        self.output((mpdu,))
        self.notify()

class PacketDispatchBlock(Block):
    inputs = [Port(Tuple[Array[[None], np.uint8], float])]

    def connectAll(self, block_by_ll_addr):
        self.outputs = [Port(Tuple[Packet, float]) for _ in block_by_ll_addr]
        self.op_by_ll_addr = {}
        for i, (ll_addr, block) in enumerate(block_by_ll_addr.items()):
            self.connect(i, block, 0)
            self.op_by_ll_addr[ll_addr] = i

    def process(self):
        for (datagram, lsnr), in self.iterinput():
            ll_sa = datagram[0:6]
            ll_da = datagram[6:12]
            packet = Packet(ll_sa, ll_da, datagram[12:])
            try:
                self.output_queues[self.op_by_ll_addr[packet.ll_da]].put_nowait((packet, lsnr))
            except queue.Full:
                warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)

class ReassemblyBlock(Block):
    inputs = [Port(Tuple[Packet, float])]
    outputs = [Port(Packet)]

    def __init__(self, pdb):
        super().__init__()
        self.pdb = pdb
        pdb.sendMSDU = self._sendMSDU

    def process(self):
        for (packet, lsnr), in self.iterinput():
            print('audio -> lowpan (%d bytes) (%10f dB)' % (12+len(packet), lsnr))
            self.pdb.recvMPDU(packet)

    def _sendMSDU(self, packet):
        self.output((packet,))
        self.notify()
