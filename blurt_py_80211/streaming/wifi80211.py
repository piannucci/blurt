#!/usr/bin/env python3.7
from os import pipe, fdopen
from queue import Queue, Empty
from itertools import count
from threading import Thread, Condition
import numpy as np
from graph import Graph
from graph.tee import Arbiter
from audio.session import play_and_record, MicrophoneAGCAdapter, IOSession
from audio.agc import MicrophoneAGCAdapter, CSMAOutStreamAdapter
from audio.graph_adapter import InStream_SourceBlock, OutStream_SinkBlock
from net.graph_adapter import TunnelSink, TunnelSource
from mac.graph_adapter import PacketDispatchBlock
from phy import Channel, IEEE80211aDecoderBlock, IEEE80211aEncoderBlock

############################ Parameters ############################

bypassAudio = False
bypassPHY = False
bypassLoWPAN = False

mtu = 76
_channel = phy.Channel(48e3, 17.0e3, 8)
audioFrameSize = 256
vuThresh = 0

############################ Audio ############################

class PollableQueue(Queue):
    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        rd, wr = pipe()
        self.pipe = (fdopen(rd, 'rb'), fdopen(wr, 'wb'))
    def fileno(self):
        return self.pipe[0].fileno()
    def put(self, item, block=True, timeout=None):
        super().put(item, block, timeout)
        self.pipe[1].write(b'\0')
        self.pipe[1].flush()
    def get_nowait(self):
        item = self.get_nowait()
        self.pipe[0].read(1)
        return item

class PollableTransceiver:
    def __init__(self):
        self.packet_in_queue = PollableQueue()
    def start(self):
        pass
    def stop(self):
        pass
    def read(self):
        try:
            return self.packet_in_queue.get_nowait()
        except Empty:
            return None
    def fileno(self):
        return self.packet_in_queue.fileno()
    def write(self, buf):
        self.packet_in_queue.put(buf)

class BlurtTransceiver(PollableTransceiver):
    def __init__(self, txchannel, rxchannel, rate=0):
        super().__init__()
        self.packet_out_queue = Queue(1)
        self.txchannel = txchannel
        self.rxchannel = rxchannel
        self.rate = rate

        if 1:
            # create blocks and stream processors
            self.ios = IOSession()
            self.ios.addDefaultInputDevice()
            self.ios.addDefaultOutputDevice()
            self.ios.negotiateFormat(kAudioObjectPropertyScopeOutput,
                minimumSampleRate=txchannel.Fs, maximumSampleRate=txchannel.Fs, outBufSize=audioFrameSize)
            self.ios.negotiateFormat(kAudioObjectPropertyScopeInput,
                minimumSampleRate=rxchannel.Fs, maximumSampleRate=rxchannel.Fs, inBufSize=audioFrameSize)
            self.inputChannels = self.ios.nChannelsPerFrame(kAudioObjectPropertyScopeInput)
            self.outputChannels = self.ios.nChannelsPerFrame(kAudioObjectPropertyScopeOutput)
            tunnels = list(utun_by_ll_addr.values())
            self.agc = MicrophoneAGCAdapter()
            self.csma = CSMAOutStreamAdapter(self.agc, vuThresh, self.outputChannels)
            self.is_b = InStream_SourceBlock()
            self.os_b = OutStream_SinkBlock()
            self.decoder_b = IEEE80211aDecoderBlock(rxchannel)
            self.encoder_b = IEEE80211aEncoderBlock(txchannel)
            self.dispatch_b = PacketDispatchBlock()
            self.arbiter_b = Arbiter(len(tunnels))
            self.reassemblers = [ReassemblyBlock(utun.pdb) for utun in tunnels]
            self.fragmenters = [FragmentationBlock(utun.pdb) for utun in tunnels]
            self.tunnelSinks = [TunnelSink(utun) for utun in utun_by_ll_addr.values()]
            self.tunnelSources = [TunnelSource(utun, utun_other[utun].ll_addr) for utun in utun_by_ll_addr.values()]

        if 1:
            # assemble graph
            sources = [self.is_b] + self.tunnelSources
            if 1:
                # ios -> agc -> is_b -> decoder_b -> dispatch_b -> reassembly_b -> sink_b
                self.ios.inStream = self.agc
                self.agc.stream = self.is_b
                self.is_b.connect(0, self.decoder_b, 0)
                self.decoder_b.connect(0, self.dispatch_b, 0)
                self.dispatch_b.connectAll({utun.ll_addr:reassembly_b for utun, reassembly_b in zip(tunnels, self.reassemblers)})
                for reassembly_b, sink_b in zip(self.reassemblers, self.tunnelSinks):
                    reassembly_b.connect(0, sink_b, 0)
            if 1:
                # source_b -> fragmentation_b -> arbiter_b -> encoder_b -> os_b -> csma -> ios
                for i, (source_b, fragmentation_b) in enumerate(zip(self.tunnelSources, self.fragmenters)):
                    source_b.connect(0, fragmentation_b, 0)
                    fragmentation_b.connect(0, self.arbiter_b, i)
                self.arbiter_b.connect(0, self.encoder_b, 0)
                self.encoder_b.connect(0, self.os_b, 0)
                self.csma.stream = self.os_b
                self.ios.outStream = self.csma
            self.g = Graph(sources)

        # TODO
        # IEEE80211aDecoderBlock, IEEE80211aEncoderBlock
        phy.Decoder(self.inQueue, rxchannel)
        phy.Encoder(self.packet_out_queue, txchannel)
        # latency_us = (ios.inLatency+ios.outLatency) * ios.nsPerAbsoluteTick*1e-3
        # move IIR filters to end of transmit chain

    def start(self):
        self.ios.start()
        self.g.run()

    def stop(self):
        self.ios.stop()
        self.ios = None

if __name__ == '__main__':
    import sys
    xcvr = BlurtTransceiver(_channel, _channel) if not bypassPHY else PollableTransceiver()
    rlist = [xcvr]
    realTunnel = '--utun' in sys.argv
    if not realTunnel:
        def packetSource():
            for i in count():
                xcvr.write(np.r_[np.random.randint(ord('A'),ord('Z'),26),
                                 np.frombuffer(('%06d' % i).encode(), np.uint8)])
        def packetSink(packet, lsnr, latency_us):
            random_letters = packet.readOctets(26).decode()
            sequence_number = packet.readOctets(6).decode()
            print(clearLine + 'audio -> /dev/null (%d bytes) (seqno %d) (%10f dB) (%.3f us)' % (
                len(packet), int(sequence_number, 10), lsnr, latency_us))
        Thread(target=packetSource, daemon=True).start()
        tunnels = ()
    else:
        import binascii
        import utun
        import lowpan
        u1 = utun.utun(mtu=1280, ll_addr=binascii.unhexlify('0200c0f000d1'))
        u2 = utun.utun(mtu=1280, ll_addr=binascii.unhexlify('0200c0f000d2'))
        tunnels = u1, u2
        rlist.extend(tunnels)
        class BlurtPDB(lowpan.PDB):
            def __init__(self, utun: utun.utun):
                super().__init__()
                self.utun = utun
                self.ll_mtu = utun.mtu - 12 # room for src, dst link-layer address
            def dispatchIPv6PDU(self, p: lowpan.Packet):
                datagram = p.tail()
                print('lowpan -> utun (%d bytes)' % (len(datagram),))
                self.utun.write(datagram)
        u1.pdb = BlurtPDB(u1)
        u2.pdb = BlurtPDB(u2)
        def packetSink(packet, lsnr, latency_us):
            print(clearLine + 'audio -> lowpan (%d bytes) (%10f dB) (%.3f us)' % (len(packet), lsnr, latency_us))
            utun_by_ll_addr[packet.ll_da].pdb.dispatchFragmentedPDU(packet)
        utun_other = {u1:u2, u2:u1}
        utun_by_ll_addr = {u1.ll_addr:u1, u2.ll_addr:u2}
    xcvr.start()
    clearLine = '\r\x1b[2K'
    try:
        while True:
            time.sleep(.05)
            vu = int(max(0, 80 + 10*np.log10(xcvr.stream.vu)))
            bar = [' '] * 100
            bar[:vu] = ['.'] * vu
            bar[vuThresh+80] = '|'
            print(clearLine + ''.join(bar) + ' %3d' % vu, end='')
    except KeyboardInterrupt:
        pass
    print(clearLine, end='')
    xcvr.stop()
