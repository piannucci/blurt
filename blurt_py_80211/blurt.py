#!/usr/bin/env python
import os
try:
    os.chdir(os.path.dirname(__file__))
except OSError:
    pass
except:
    exit()
import numpy as np
import sys
import audioLoopback, channelModel, maskNoise, wifi80211
import audio, util
import pylab as pl
import iir

wifi = wifi80211.WiFi_802_11()

fn = '35631__reinsamba__crystal-glass.wav'
Fs = 48000.
Fc = 19000. #Fs/4
upsample_factor = 16
mask_noise = maskNoise.prepareMaskNoise(fn, Fs, Fc, upsample_factor)
mask_noise = mask_noise[:int(Fs)]
mask_noise[int(Fs*.5):] *= 1-np.arange(int(Fs*.5))/float(Fs*.5)
lsnr = None
mask_noise = None

rate = 0
length = 16

def test(visualize=False):
    input_octets = np.random.random_integers(0,255,length)
    output = wifi.encode(input_octets, rate)
    if lsnr is not None:
        input = channelModel.channelModel(output, lsnr)
        bitrate = None
    elif Fs is not None:
        input = audioLoopback.audioLoopback(output, Fs, Fc, upsample_factor, mask_noise)
        bitrate = input_octets.size*8 * Fs / float(output.size) / upsample_factor
    try:
        results, _ = wifi.decode(input, visualize)
        return len(results), bitrate
    except Exception, e:
        print e
        return False, bitrate

def transmit(message):
    length = len(message)
    input_octets = np.array(map(ord, message))
    output = wifi.encode(input_octets, rate)
    audioLoopback.audioOut(output, Fs, Fc, upsample_factor, mask_noise)

def presentResults(results, drawFunc):
    _results = results
    _drawFunc = drawFunc
    def f():
        if len(_results):
            for result in _results:
                payload, _, _, lsnr_estimate = result
                print repr(''.join(map(chr, payload))) + (' @ %.3f dB' % lsnr_estimate)
                if typeModHex:
                    import keypress
                    mod = "cbdefghijklnrtuv"
                    otp = ''.join('%s%s' % (mod[x>>4], mod[x&15]) for x in payload)
                    keypress.type(otp+'\n')
        #else:
        #    decoderDiagnostics()
        #    pl.draw()
        if drawFunc is not None:
            pl.figure(1)
            drawFunc()
            pl.draw()
    return f

badPacketWaveforms = []

class ContinuousReceiver(audioLoopback.AudioBuffer):
    def init(self):
        packetLength = wifi.encode(np.zeros(length, int), rate).size
        self.kwargs['maximum'] = int(packetLength*4)
        self.kwargs['trigger'] = int(packetLength*2)
        self.dtype = np.complex64
        super(ContinuousReceiver, self).init()
        self.inputProcessor = audioLoopback.InputProcessor(Fs, Fc, upsample_factor)
    def trigger_received(self):
        input = self.peek(self.maximum)
        #print '%.0f%%' % ((float(self.length)/self.maximum) * 100)
        #print '\r\x1b[K' + ('.' * int(30 + 10*np.log10(np.var(input)))),
        endIndex = 0
        try:
            results, drawFunc = wifi.decode(input, visualize, visualize)
            for result in results:
                _, startIndex, endIndex, _ = result
            if not len(results):
                badPacketWaveforms.append(input)
                del badPacketWaveforms[:-10]
            self.onMainThread(presentResults(results, drawFunc))
        except Exception, e:
            badPacketWaveforms.append(input)
            del badPacketWaveforms[:-10]
            print repr(e)
        if endIndex:
            return endIndex
        else:
            return self.trigger/2

class ContinuousTransmitter(audio.stream.ThreadedStream):
    def init(self):
        self.channels = 2
        self.i = 0
        cutoff = Fc - Fs/upsample_factor
        self.hp = [iir.highpass(cutoff/Fs, continuous=True, dtype=np.float64) for i in xrange(2)]
        super(ContinuousTransmitter, self).init()
    def thread_produce(self):
        input_octets = ord('A') + np.random.random_integers(0,25,length)
        input_octets[:6] = map(ord, '%06d' % self.i)
        self.i += 1
        output = wifi.encode(input_octets, rate)
        output = audioLoopback.processOutput(output, Fs, Fc, upsample_factor, None)
        if False:
            return output
        else:
            return np.hstack((self.hp[0](output[:,0])[:,np.newaxis],
                              self.hp[1](output[:,1])[:,np.newaxis]))

def startListening():
    audio.record(ContinuousReceiver(), Fs)

def startTransmitting():
    audio.play(ContinuousTransmitter(), Fs)

def decoderDiagnostics(waveform=None):
    if waveform is None:
        waveform = badPacketWaveforms[-1]
    Fs_eff = Fs/upsample_factor
    ac = wifi.autocorrelate(waveform)
    ac_t = np.arange(ac.size)*16/Fs_eff
    synch = wifi.synchronize(waveform)/float(Fs_eff)
    pl.figure(2)
    pl.clf()
    pl.subplot(211)
    pl.specgram(waveform, NFFT=64, noverlap=64-1, Fc=Fc, Fs=Fs_eff, interpolation='nearest', window=lambda x:x)
    pl.xlim(0, waveform.size/Fs_eff)
    yl = pl.ylim(); pl.vlines(synch, *yl); pl.ylim(*yl)
    pl.subplot(212)
    pl.plot(ac_t, ac)
    yl = pl.ylim(); pl.vlines(synch, *yl); pl.ylim(*yl)
    pl.xlim(0, waveform.size/Fs_eff)

visualize = False
typeModHex = False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        while True:
            if args[0] == '--Fc':
                Fc = float(args[1])
                args = args[2:]
            elif args[0] == '--visualize':
                visualize = True
                args = args[1:]
            elif args[0] == '--length':
                length = int(args[1])
                args = args[2:]
            else:
                break
        if args[0] == '--rx':
            typeModHex = '--yubikey' in args
            doDiagnostics = ('--nodiags' not in args) and not typeModHex
            try:
                startListening()
            except KeyboardInterrupt:
                pass
            if doDiagnostics:
                decoderDiagnostics()
        elif args[0] == '--tx':
            startTransmitting()
        elif args[0] == '--wav-in' and len(args) > 1:
            input, Fs = util.readwave(args[1])
            input = audioLoopback.processInput(input, Fs, Fc, upsample_factor)
            for payload,_,_,lsnr_estimate in wifi.decode(input)[0]:
                print repr(''.join(map(chr, payload))) + (' @ %.1f dB' % lsnr_estimate)
        elif args[0] == '--wav-out' and len(args) > 1:
            fn = args[1]
            args = args[2:]
            packets = 1
            while len(args):
                if args[0] == '--packets':
                    packets = int(args[1])
                    args = args[2:]
            outputChunks = []
            for i in xrange(packets):
                input_octets = ord('A') + np.random.random_integers(0,25,length)
                input_octets[:6] = map(ord, '%06d' % i)
                output = audioLoopback.processOutput(wifi.encode(input_octets, rate), Fs, Fc, upsample_factor, mask_noise)
                output *= 1 / np.abs(output).max()
                outputChunks.append(output)
                outputChunks.append(np.zeros((Fs/10, outputChunks[0].shape[1])))
            outputChunks.append(np.zeros((Fs/2, outputChunks[0].shape[1])))
            output = np.vstack(outputChunks)
            bytesPerSample = 3
            util.writewave(fn, output, Fs, bytesPerSample)
