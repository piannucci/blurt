#!/usr/bin/env python3.7
import time
import numpy as np
import binascii
import sys
from . import io
from .net import utun
from .mac import lowpan
from .graph import Graph
from .graph.tee import Tee, Arbiter
from .graph.fileio import FileSink
from .graph.selector import Selector
from .audio import IOSession, MicrophoneAGCAdapter, CSMAOutStreamAdapter
from .audio import InStream_SourceBlock, OutStream_SinkBlock, IOSession_Block
from .audio import AudioHardware as AH
from .net.graph_adapter import TunnelSink, TunnelSource
from .mac.graph_adapter import PacketDispatchBlock, FragmentationBlock, ReassemblyBlock
from .phy.ieee80211a import Channel, IEEE80211aDecoderBlock, IEEE80211aEncoderBlock
from .audio.graph_adapter import AudioBypass_Block

############################ Parameters ############################

mtu = 76
_channel = Channel(48e3, 17.0e3, 8)
audioFrameSize = 256
vuThresh = 25
recordToFile = False
bypassAudio = False
vuMeter = not bypassAudio

############################ Audio ############################

class BlurtTransceiver:
    def __init__(self, utun_by_ll_addr, txchannel, rxchannel, *, runloop=None):
        super().__init__()
        self.txchannel = txchannel
        self.rxchannel = rxchannel

        if 1:
            # create blocks and stream processors
            sources = []
            if not bypassAudio:
                self.ios = IOSession()
                self.ios.addDefaultInputDevice()
                self.ios.addDefaultOutputDevice()
                self.ios.negotiateFormat(AH.kAudioObjectPropertyScopeOutput,
                    minimumSampleRate=txchannel.Fs, maximumSampleRate=txchannel.Fs, outBufSize=audioFrameSize)
                self.ios.negotiateFormat(AH.kAudioObjectPropertyScopeInput,
                    minimumSampleRate=rxchannel.Fs, maximumSampleRate=rxchannel.Fs, inBufSize=audioFrameSize)
                self.ios_b = IOSession_Block(self.ios)
                self.inputChannels = self.ios.nChannelsPerFrame(AH.kAudioObjectPropertyScopeInput)
                self.outputChannels = self.ios.nChannelsPerFrame(AH.kAudioObjectPropertyScopeOutput)
                self.agc = MicrophoneAGCAdapter()
                self.csma = CSMAOutStreamAdapter(self.agc, vuThresh, self.outputChannels)
                self.is_b = InStream_SourceBlock(self.ios)
                self.os_b = OutStream_SinkBlock()
                sources.append(self.is_b)
                sources.append(self.ios_b)
            else:
                self.audio_loopback_b = AudioBypass_Block()
                self.is_b = self.os_b = self.audio_loopback_b
            tunnels = list(utun_by_ll_addr.values())
            utun_other = dict(zip(tunnels, tunnels[::-1]))
            if recordToFile:
                self.tee_b = Tee(2)
                self.filesink_b = FileSink('in_stream.s16_%d' % self.inputChannels)
            self.decoder_b = IEEE80211aDecoderBlock(rxchannel)
            self.encoder_b = IEEE80211aEncoderBlock(txchannel)
            self.encoder_b.preferredRate = 6
            self.dispatch_b = PacketDispatchBlock()
            self.arbiter_b = Arbiter(len(tunnels))
            self.reassemblers = [ReassemblyBlock(utun.pdb) for utun in tunnels]
            self.fragmenters = [FragmentationBlock(utun.pdb) for utun in tunnels]
            self.tunnelSinks = [TunnelSink(utun) for utun in utun_by_ll_addr.values()]
            self.tunnelSources = [TunnelSource(utun, utun_other[utun].ll_addr) for utun in utun_by_ll_addr.values()]
            sources.extend(self.tunnelSources)

        if 1:
            # assemble graph
            if 1:
                # is_b -> decoder_b -> dispatch_b -> reassembly_b -> sink_b
                if not bypassAudio:
                    # ios -> agc -> is_b
                    self.ios.inStream = self.agc
                    self.agc.stream = self.is_b
                if recordToFile:
                    self.is_b.connect(0, self.tee_b, 0)
                    self.tee_b.connect(0, self.decoder_b, 0)
                    self.tee_b.connect(1, self.filesink_b, 0)
                else:
                    self.is_b.connect(0, self.decoder_b, 0)
                self.decoder_b.connect(0, self.dispatch_b, 0)
                self.dispatch_b.connectAll({utun.ll_addr:reassembly_b for utun, reassembly_b in zip(tunnels, self.reassemblers)})
                for reassembly_b, sink_b in zip(self.reassemblers, self.tunnelSinks):
                    reassembly_b.connect(0, sink_b, 0)
            if 1:
                # source_b -> fragmentation_b -> arbiter_b -> encoder_b -> os_b
                for i, (source_b, fragmentation_b) in enumerate(zip(self.tunnelSources, self.fragmenters)):
                    source_b.connect(0, fragmentation_b, 0)
                    fragmentation_b.connect(0, self.arbiter_b, i)
                self.arbiter_b.connect(0, self.encoder_b, 0)
                self.encoder_b.connect(0, self.os_b, 0)
                if not bypassAudio:
                    # os_b -> csma -> ios
                    self.csma.stream = self.os_b
                    self.ios.outStream = self.csma
            self.graph = Graph(sources, runloop=runloop)

        self.start = self.graph.start
        self.stop = self.graph.stop

        # TODO
        # latency = ios.inLatency+ios.outLatency
        # move IIR filters to end of transmit chain

if __name__ == '__main__':
    runloop = Selector()
    u1 = utun.utun(mtu=1280, ll_addr=binascii.unhexlify('0200c0f000d1'))
    u2 = utun.utun(mtu=1280, ll_addr=binascii.unhexlify('0200c0f000d2'))
    class BlurtPDB(lowpan.PDB):
        def __init__(self, utun: utun.utun, *, runloop=None):
            super().__init__(runloop=runloop)
            self.utun = utun
            self.ll_mtu = mtu - 12 # room for src, dst link-layer address
    u1.pdb = BlurtPDB(u1)
    u2.pdb = BlurtPDB(u2)
    xcvr = BlurtTransceiver({
            u1.ll_addr:u1,
            u2.ll_addr:u2
        },
        _channel,
        _channel,
        runloop=runloop)
    xcvr.start()
    try:
        while True:
            time.sleep(.05)
            if bypassAudio:
                # keep a steady stream of zeros flowing when there isn't a packet
                xcvr.audio_loopback_b.input_queues[0].put(np.zeros((1000,2)))
                xcvr.graph.notify()
            if vuMeter:
                vu = int(max(0, 80 + 10*np.log10(xcvr.agc.vu)))
                bar = [' '] * 150
                bar[:vu] = ['.'] * vu
                bar[vuThresh+80] = '|'
                sys.stderr.status(''.join(bar) + ' %3d' % vu)
    except KeyboardInterrupt:
        pass
    xcvr.stop()
