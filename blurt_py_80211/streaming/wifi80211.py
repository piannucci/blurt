#!/usr/bin/env python3.7
from os import pipe, fdopen
from queue import Queue
from itertools import count, islice
from collections import namedtuple, deque
from threading import Thread, Condition
import numpy as np
from audio import AudioInterface, stream
from iir import IIRFilter
import kernels

############################ Parameters ############################

bypassAudio = False
bypassPHY = False
bypassLoWPAN = False

mtu = 76 #150
_channel = phy.Channel(48e3, 17.0e3, 8) # 16.5e3

audioFrameSize = 256

############################ Audio ############################

class BlurtStream(stream.ThreadedStream):
    def __init__(self, encoder, sink, rxchannel):
        self.encoder = iter(encoder)
        self.vu = 1e-10
        self.cv = Condition()
        self.ready = False
        super().__init__(channels=2, inThread=True, outThread=True, out_queue_depth=1,
                         inBufSize=audioFrameSize, outBufSize=audioFrameSize)
        self.decoder = phy.Decoder(self.inQueue, rxchannel)
        self.sink = sink
        with self.cv:
            self.ready = True
            self.cv.notify()
    def write(self, frames): # called from IO proc
        self.vu = (frames**2).max()
        super().write(frames)
    def in_thread_loop(self):
        with self.cv:
            while not self.ready:
                self.cv.wait()
        for packet in self.decoder:
            self.sink(packet)
    def immediate_produce(self):
        carrierSense = (self.vu == 0 or self.vu > 10**(.1 * (vuThresh-80)))
        queueEmpty = self.outQueue.empty()
        if queueEmpty or carrierSense:
            return np.zeros((self.outBufSize, 2))
        else:
            return self.outQueue.get_nowait()
    def produce(self):
        return next(self.encoder)

class WaitableTransceiver:
    def __init__(self):
        rd, wr = pipe()
        self.readpipe = (fdopen(rd, 'rb'), fdopen(wr, 'wb'))
        self.packet_in_queue = deque()
    def start(self):
        pass
    def stop(self):
        pass
    def read(self):
        try:
            self.readpipe[0].read(1)
            return self.packet_in_queue.popleft()
        except:
            return None
    def write(self, buf):
        self.sink(buf)
    def fileno(self):
        return self.readpipe[0].fileno()
    def sink(self, buf):
        self.packet_in_queue.append(buf)
        self.readpipe[1].write(b'\0')
        self.readpipe[1].flush()

class BlurtTransceiver(WaitableTransceiver):
    def __init__(self, txchannel, rxchannel, rate=0):
        super().__init__()
        self.packet_out_queue = Queue(1)
        self.txchannel = txchannel
        self.rxchannel = rxchannel
        self.rate = rate
        self.stream = BlurtStream(
            phy.Encoder(self.packet_out_queue, txchannel),
            self.sink, rxchannel)
    def start(self):
        self.audioInterface = AudioInterface()
        self.audioInterface.record(self.stream, self.rxchannel.Fs)
        self.audioInterface.play(self.stream, self.txchannel.Fs)
    def stop(self):
        self.audioInterface.stop()
        self.audioInterface = None
    def write(self, buf):
        self.packet_out_queue.put(buf)

if __name__ == '__main__':
    import sys
    xcvr = BlurtTransceiver(_channel, _channel) if not bypassPHY else WaitableTransceiver()
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
        import select
        while True:
            for fd in select.select(rlist, [], [], .01)[0]:
                if fd in tunnels:
                    datagram = fd.read()
                    ll_sa = fd.ll_addr
                    ll_da = utun_other[fd].ll_addr
                    print(clearLine + 'utun -> lowpan (%d bytes)' % (len(datagram),))
                    fragments = fd.pdb.compressIPv6Datagram(lowpan.Packet(ll_sa, ll_da, datagram))
                    for f in fragments:
                        print(clearLine + 'lowpan -> audio (%d bytes)' % (12+len(f),))
                        xcvr.write(np.frombuffer(ll_sa + ll_da + f, np.uint8))
                elif fd is xcvr:
                    datagram, lsnr = xcvr.read()
                    ap = xcvr.audioInterface
                    latency_us = (ap.recordingLatency+ap.playbackLatency) * ap.nanosecondsPerAbsoluteTick*1e-3
                    ll_sa = datagram[0:6]
                    ll_da = datagram[6:12]
                    packet = lowpan.Packet(ll_sa, ll_da, datagram[12:])
                    packetSink(packet, lsnr, latency_us)
            vu = int(max(0, 80 + 10*np.log10(xcvr.stream.vu)))
            bar = [' '] * 100
            bar[:vu] = ['.'] * vu
            bar[vuThresh] = '|'
            print(clearLine + ''.join(bar) + ' %3d' % vu, end='')
    except KeyboardInterrupt:
        pass
    print(clearLine, end='')
    xcvr.stop()
