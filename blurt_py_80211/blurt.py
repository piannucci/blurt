#!/usr/bin/env python
import numpy as np
import sys
import audioLoopback, channelModel, maskNoise, wifi80211
import audio, audio.stream

wifi = wifi80211.WiFi_802_11()

fn = '35631__reinsamba__crystal-glass.wav'
Fs = 96000.
Fc = 16000.
upsample_factor = 16
mask_noise = maskNoise.prepareMaskNoise(fn, Fs, Fc, upsample_factor)
mask_noise = mask_noise[:int(Fs)]
mask_noise[int(Fs*.5):] *= 1-np.arange(int(Fs*.5))/float(Fs*.5)
lsnr = 15.
rate = 1
length = 100

def test():
    input_octets = np.random.random_integers(0,255,length)
    output = wifi.encode(input_octets, rate)
    if lsnr is not None:
        input = channelModel.channelModel(output, lsnr)
        bitrate = None
    elif Fs is not None:
        input = audioLoopback.audioLoopback(output, Fs, Fc, upsample_factor, mask_noise)
        bitrate = input_octets.size*8 * Fs / float(output.size) / upsample_factor
    try:
        return wifi.decode(input, lsnr) is not None, bitrate
    except Exception, e:
        print e
        return False, bitrate

def testOut(message):
    length = len(message)
    input_octets = np.array(map(ord, message))
    output = wifi.encode(input_octets, rate)
    input = audioLoopback.audioOut(output, Fs, Fc, upsample_factor, mask_noise)

def testIn(duration=5.):
    input = audioLoopback.audioIn(Fs, Fc, upsample_factor, duration)
    result = wifi.decode(input, lsnr)
    return ''.join(map(chr, result)) if result is not None else None

def testInStream():
    vumeter = audio.stream.VUMeter(Fs=float(Fs)/upsample_factor/16.)
    #oscilloscope = audio.stream.Oscilloscope(aggregate=32)
    autocorrelator = wifi80211.Autocorrelator(next=vumeter)
    downconverter = audioLoopback.downconverter(autocorrelator, Fs, Fc, upsample_factor)
    audio.record(downconverter, Fs)
