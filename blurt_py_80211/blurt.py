#!/usr/bin/env python
import numpy as np
import sys
import audioLoopback, channelModel, maskNoise, wifi80211
import audio, audio.stream

wifi = wifi80211.WiFi_802_11()

fn = '35631__reinsamba__crystal-glass.wav'
Fs = 48000.
Fc = 16000.
upsample_factor = 16
mask_noise = maskNoise.prepareMaskNoise(fn, Fs, Fc, upsample_factor)
mask_noise = mask_noise[:int(Fs)]
mask_noise[int(Fs*.5):] *= 1-np.arange(int(Fs*.5))/float(Fs*.5)
lsnr = 15.
rate = 1
length = 100

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
        return wifi.decode(input, lsnr, visualize) is not None, bitrate
    except Exception, e:
        print e
        return False, bitrate

def testOut(message):
    length = len(message)
    input_octets = np.array(map(ord, message))
    output = wifi.encode(input_octets, rate)
    audioLoopback.audioOut(output, Fs, Fc, upsample_factor, mask_noise)

def testIn(duration=5., visualize=False):
    input = audioLoopback.audioIn(Fs, Fc, upsample_factor, duration)
    result, used_samples = wifi.decode(input, lsnr, visualize)
    return ''.join(map(chr, result)) if result is not None else None

def testInStream():
    vumeter = audio.stream.VUMeter(Fs=float(Fs)/upsample_factor/16.)
    #oscilloscope = audio.stream.Oscilloscope(aggregate=32)
    autocorrelator = wifi80211.Autocorrelator(next=vumeter)
    downconverter = audioLoopback.downconverter(autocorrelator, Fs, Fc, upsample_factor)
    audio.record(downconverter, Fs)

class AudioConsumer(audioLoopback.AudioBuffer):
    def init(self):
        self.kwargs['maximum'] = int(Fs*3)
        self.kwargs['trigger'] = int(Fs)
        super(AudioConsumer, self).init()
    def trigger_received(self):
        input = self.peek(self.maximum)
        #print '%.0f%%' % ((float(self.length)/self.maximum) * 100)
        #print '\r\x1b[K' + ('.' * int(30 + 10*np.log10(np.var(input)))),
        used_samples = 0
        try:
            input = audioLoopback.processInput(input, Fs, Fc, upsample_factor)
            result = wifi.decode(input, lsnr)
            if result is not None:
                input, used_samples = result
                print repr(''.join(map(chr, input)))
        except Exception, e:
            print repr(e)
        if used_samples:
            return used_samples*upsample_factor
        else:
            return self.trigger/2

class Spammer(audio.stream.ThreadedStream):
    def thread_produce(self):
        input_octets = ord('A') + np.random.random_integers(0,25,length)
        output = wifi.encode(input_octets, rate)
        output = audioLoopback.processOutput(output, Fs, Fc, upsample_factor, None)
        return output[:,0]

def startListening():
    ac = AudioConsumer()
    audio.record(ac, Fs)

def startTransmitting():
    sp = Spammer()
    audio.play(sp, Fs)
