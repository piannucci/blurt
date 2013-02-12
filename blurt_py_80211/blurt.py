#!/usr/bin/env python
import numpy as np
import sys
import audioLoopback, channelModel, maskNoise, wifi80211

wifi = wifi80211.WiFi_802_11()

fn = '35631__reinsamba__crystal-glass.wav'
Fs = 96000.
Fc = 16000.
upsample_factor = 16
mask_noise = maskNoise.prepareMaskNoise(fn, Fs, Fc, upsample_factor)
mask_noise = mask_noise[:int(Fs)]
mask_noise[int(Fs*.5):] *= 1-np.arange(int(Fs*.5))/float(Fs*.5)
lsnr = None

def test():
    rate, length = 1, 1000
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
    except:
        return False, bitrate

def testOut(message):
    rate, length = 1, len(message)
    input_octets = np.array(map(ord, message))
    output = wifi.encode(input_octets, rate)
    input = audioLoopback.audioOut(output, Fs, Fc, upsample_factor, mask_noise)

def testIn():
    input = audioLoopback.audioIn(Fs, Fc, upsample_factor)
    try:
        return ''.join(map(chr, wifi.decode(input, lsnr)))
    except:
        return None
