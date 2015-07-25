#!/usr/bin/env python
import numpy as np
import sys
import audioLoopback, wifi80211 as wifi
import audio
import iir
import traceback
import collections
import os

Channel = collections.namedtuple('Channel', ['Fs', 'Fc', 'upsample_factor'])

channels = [Channel(96e3, 20e3, 16), Channel(96e3, 14e3, 16)]

class ContinuousReceiver(audioLoopback.AudioBuffer):
    def init(self):
        self.mtu = self.kwargs['mtu']
        packetLength = wifi.encode(np.zeros(self.mtu, int), 0).size
        self.kwargs['maximum'] = int(packetLength*4)
        self.kwargs['trigger'] = int(packetLength*2)
        self.dtype = np.complex64
        super(ContinuousReceiver, self).init()
        self.channel = self.kwargs['channel']
        self.packetSink = self.kwargs['packetSink']
        self.inputProcessor = audioLoopback.InputProcessor(self.channel.Fs, self.channel.Fc, self.channel.upsample_factor)
    def trigger_received(self):
        input = self.peek(self.maximum)
        endIndex = 0
        try:
            results, _ = wifi.decode(input, False, False)
            for payload, startIndex, endIndex, lsnr_estimate in results:
                self.packetSink(payload.astype(np.uint8).tostring())
        except Exception, e:
            traceback.print_exc()
        if endIndex:
            return endIndex
        else:
            return self.trigger/2

class ContinuousTransmitter(audio.stream.ThreadedStream):
    def init(self):
        self.channels = 2
        self.channel = self.kwargs['channel']
        self.packetSource = self.kwargs['packetSource']
        cutoff = self.channel.Fc + self.channel.Fs/self.channel.upsample_factor
        self.hp = [iir.highpass(cutoff/self.channel.Fs, continuous=True, dtype=np.float64) for i in range(2)]
        super(ContinuousTransmitter, self).init()
    def thread_produce(self):
        packet = self.packetSource()
        if packet is not None:
            buf, rate = packet
            output = wifi.encode(np.fromstring(buf, dtype=np.uint8), rate)
            output = audioLoopback.processOutput(output, self.channel.Fs, self.channel.Fc, self.channel.upsample_factor, None)
        else:
            output = np.zeros((self.channel.Fs * .001,2))
        return np.hstack((self.hp[0](output[:,0])[:,np.newaxis],
                          self.hp[1](output[:,1])[:,np.newaxis]))

class AsynchronousTransciever(object):
    def __init__(self, txchannel, rxchannel, rate=0, mtu=150):
        self.inqueue = []
        self.outqueue = []
        self.txchannel = txchannel
        self.rxchannel = rxchannel
        self.rate = rate
        self.transmitter = ContinuousTransmitter(channel=txchannel, packetSource=self.txPoll)
        self.receiver = ContinuousReceiver(channel=rxchannel, packetSink=self.rxPush, mtu=mtu)
        rd, wr = os.pipe()
        self.readpipe = (os.fdopen(rd, 'rb'), os.fdopen(wr, 'wb'))
    def start(self):
        self.rxAudioInterface = audio.AudioInterface()
        self.rxAudioInterface.record(self.receiver, self.rxchannel.Fs)
        self.txAudioInterface = audio.AudioInterface()
        self.txAudioInterface.play(self.transmitter, self.txchannel.Fs)
    def stop(self):
        self.rxAudioInterface.stop()
        self.rxAudioInterface = None
        self.txAudioInterface.stop()
        self.txAudioInterface = None
    def txPoll(self):
        try:
            return self.outqueue.pop(0), self.rate
        except:
            return None
    def rxPush(self, buf):
        self.inqueue.append(buf)
        self.readpipe[1].write('\0')
        self.readpipe[1].flush()
    def read(self):
        try:
            self.readpipe[0].read(1)
            return self.inqueue.pop(0)
        except:
            return None
    def write(self, buf):
        self.outqueue.append(buf)
    def fileno(self):
        return self.readpipe[0].fileno()

Fs = 96000.
Fc1 = 19000.
Fc2 = 13000.
upsample_factor = 32
