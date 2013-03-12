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
import audio
import pylab as pl

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
        visualize = True
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
        super(ContinuousTransmitter, self).init()
        self.i = 0
    def thread_produce(self):
        input_octets = ord('A') + np.random.random_integers(0,25,length)
        input_octets[:6] = map(ord, '%06d' % self.i)
        self.i += 1
        output = wifi.encode(input_octets, rate)
        output = audioLoopback.processOutput(output, Fs, Fc, upsample_factor, None)
        return output

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
    synch = wifi.synchronize(waveform, True)/float(Fs_eff)
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

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--rx':
            try:
                startListening()
            except KeyboardInterrupt:
                pass
            decoderDiagnostics()
        elif sys.argv[1] == '--tx':
            startTransmitting()
        elif sys.argv[1] == '--wav' and len(sys.argv) > 2:
            import wave
            f = wave.open(sys.argv[2])
            Fs = f.getframerate()
            assert f.getcompname() == 'not compressed' and f.getcomptype() == 'NONE', 'Bad WAV type'
            channels = f.getnchannels()
            nframes = f.getnframes()
            dtype = [None, np.uint8, np.int16, None, np.int32][f.getsampwidth()]
            input = f.readframes(nframes)
            if f.getsampwidth() == 3:
                frames = ''.join(frames[3*i:3*(i+1)] + ('\xff' if ord(frames[3*i+2]) & 0x80 else '\0') for i in xrange(len(frames)/3))
                dtype = np.int32
            input = np.fromstring(input, dtype).astype(float)
            nframes = input.size // channels
            input = input.reshape(nframes, channels).mean(1)
            f.close()
            input = audioLoopback.processInput(input, Fs, Fc, upsample_factor)
            for payload,_,_,lsnr_estimate in wifi.decode(input)[0]:
                print repr(''.join(map(chr, payload))) + (' @ %.1f dB' % lsnr_estimate)
