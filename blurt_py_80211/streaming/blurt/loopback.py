#!/usr/bin/env python3.7
import os
import queue
import itertools
import time
import numpy as np
import binascii
from .net import utun
from .mac import lowpan
from .graph import Graph
from .graph.tee import Arbiter
from .audio import IOSession, play_and_record, MicrophoneAGCAdapter, CSMAOutStreamAdapter
from .audio import InStream_SourceBlock, OutStream_SinkBlock, IOSession_Block
from .audio import AudioHardware as AH
from .net.graph_adapter import TunnelSink, TunnelSource
from .mac.graph_adapter import PacketDispatchBlock, FragmentationBlock, ReassemblyBlock
from .phy.ieee80211a import Channel, IEEE80211aDecoderBlock, IEEE80211aEncoderBlock
from .phy.loopback import PHYLoopbackBlock
from .audio.graph_adapter import AudioBypass_Block

############################ Parameters ############################

mtu = 76
_channel = Channel(48e3, 17.0e3, 8)
audioFrameSize = 256
vuThresh = 0

############################ Audio ############################

class BlurtTransceiver:
    def __init__(self, utun_by_ll_addr, txchannel, rxchannel):
        super().__init__()
        self.txchannel = txchannel
        self.rxchannel = rxchannel

        tunnels = list(utun_by_ll_addr.values())
        utun_other = dict(zip(tunnels, tunnels[::-1]))
        self.decoder_b = IEEE80211aDecoderBlock(rxchannel)
        self.encoder_b = IEEE80211aEncoderBlock(txchannel)
        self.dispatch_b = PacketDispatchBlock()
        self.arbiter_b = Arbiter(len(tunnels))
        self.reassemblers = [ReassemblyBlock(utun.pdb) for utun in tunnels]
        self.fragmenters = [FragmentationBlock(utun.pdb) for utun in tunnels]
        self.tunnelSinks = [TunnelSink(utun) for utun in utun_by_ll_addr.values()]
        self.tunnelSources = [TunnelSource(utun, utun_other[utun].ll_addr) for utun in utun_by_ll_addr.values()]
        self.loopback_b = AudioBypass_Block()
        # source_b -> fragmentation_b -> arbiter_b -> encoder_b -> decoder_b -> dispatch_b -> reassembly_b -> sink_b
        for i, (source_b, fragmentation_b) in enumerate(zip(self.tunnelSources, self.fragmenters)):
            source_b.connect(0, fragmentation_b, 0)
            fragmentation_b.connect(0, self.arbiter_b, i)
        self.arbiter_b.connect(0, self.encoder_b, 0)
        self.encoder_b.connect(0, self.loopback_b, 0)
        self.loopback_b.connect(0, self.decoder_b, 0)
        self.decoder_b.connect(0, self.dispatch_b, 0)
        self.dispatch_b.connectAll({utun.ll_addr:reassembly_b for utun, reassembly_b in zip(tunnels, self.reassemblers)})
        for reassembly_b, sink_b in zip(self.reassemblers, self.tunnelSinks):
            reassembly_b.connect(0, sink_b, 0)
        self.g = Graph(self.tunnelSources)

    def start(self):
        self.g.start()

    def stop(self):
        self.g.stop()

if __name__ == '__main__':
    u1 = utun.utun(mtu=1280, ll_addr=binascii.unhexlify('0200c0f000d1'))
    u2 = utun.utun(mtu=1280, ll_addr=binascii.unhexlify('0200c0f000d2'))
    class BlurtPDB(lowpan.PDB):
        def __init__(self, utun: utun.utun):
            super().__init__()
            self.utun = utun
            self.ll_mtu = utun.mtu - 12 # room for src, dst link-layer address
    u1.pdb = BlurtPDB(u1)
    u2.pdb = BlurtPDB(u2)
    xcvr = BlurtTransceiver({
        u1.ll_addr:u1,
        u2.ll_addr:u2
    }, _channel, _channel)
    xcvr.start()
    try:
        while True:
            time.sleep(.05)
            xcvr.loopback_b.input_queues[0].put(np.zeros((1000,2)))
            xcvr.g.notify()
    except KeyboardInterrupt:
        pass
    xcvr.stop()
