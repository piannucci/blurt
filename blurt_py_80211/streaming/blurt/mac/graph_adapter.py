import warnings
import queue
import numpy as np
import typing
from ..graph import Output, Input, Block, OverrunWarning
from .lowpan import Packet

class FragmentationBlock(Block):
    inputs = [Input(())]
    outputs = [Output(np.uint8, ())]

    def __init__(self, pdb):
        super().__init__()
        self.pdb = pdb
        self.pdb.injectMPDU = self._injectMPDU

    def process(self):
        for packet, in self.input():
            fragments = self.pdb.compressIPv6Datagram(packet)
            for f in fragments:
                print('lowpan -> %d B' % (12+len(f),))
                self.output((np.frombuffer(packet.ll_sa + packet.ll_da + f, np.uint8),))

    def _injectMPDU(self, mpdu):
        self.output((mpdu,))
        self.notify()

class PacketDispatchBlock(Block):
    inputs = [Input(())]

    def connectAll(self, block_by_ll_addr):
        self.outputs = [Output(typing.Tuple[Packet, float], ()) for _ in block_by_ll_addr]
        self.op_by_ll_addr = {}
        for i, (ll_addr, block) in enumerate(block_by_ll_addr.items()):
            self.connect(i, block, 0)
            self.op_by_ll_addr[ll_addr] = i

    def process(self):
        for (datagram, lsnr), in self.input():
            ll_sa = datagram[0:6]
            ll_da = datagram[6:12]
            packet = Packet(ll_sa, ll_da, datagram[12:])
            try:
                self.output_queues[self.op_by_ll_addr[packet.ll_da]].put_nowait((packet, lsnr))
            except queue.Full:
                warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)

class ReassemblyBlock(Block):
    inputs = [Input(())]
    outputs = [Output(typing.Tuple[Packet, float], ())]

    def __init__(self, pdb):
        super().__init__()
        self.pdb = pdb
        pdb.dispatchIPv6PDU = self._dispatchIPv6PDU

    def process(self):
        for (packet, lsnr), in self.input():
            print('audio -> lowpan (%d bytes) (%10f dB)' % (12+len(packet), lsnr))
            self.pdb.dispatchFragmentedPDU(packet)

    def _dispatchIPv6PDU(self, packet):
        self.output((packet,))
