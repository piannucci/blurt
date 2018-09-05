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
        self.pdb = pdb

    def process(self):
        while True:
            try:
                packet = self.input_queues[0].get_nowait()
            except queue.Empty:
                break
            fragments = self.pdb.compressIPv6Datagram(packet)
            try:
                for f in fragments:
                    print('lowpan -> %d B' % (12+len(f),))
                    self.output_queues[0].put_nowait(np.frombuffer(ll_sa + ll_da + f, np.uint8))
            except queue.Full:
                warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)

class PacketDispatchBlock(Block):
    inputs = [Input(())]

    def connectAll(self, block_by_ll_addr):
        self.outputs = [Output(typing.Tuple[Packet, float], ()) for _ in block_by_ll_addr]
        self.op_by_ll_addr = {}
        for i, (ll_addr, block) in enumerate(block_by_ll_addr.items()):
            self.connect(i, block, 0)
            self.op_by_ll_addr[ll_addr] = i

    def process(self):
        while True:
            try:
                datagram, lsnr = self.input_queues.get_nowait()
            except queue.Empty:
                break
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
        self.pdb = pdb
        pdb.dispatchIPv6PDU = self.dispatchIPv6PDU

    def process(self):
        while True:
            try:
                packet, lsnr = self.input_queues[0].get_nowait()
            except queue.Empty:
                break
            print('audio -> lowpan (%d bytes) (%10f dB)' % (len(packet), lsnr))
            self.pdb.dispatchFragmentedPDU(packet)

    def dispatchIPv6PDU(self, packet):
        try:
            self.output_queues[0].put_nowait(packet)
        except queue.Full:
            warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)
