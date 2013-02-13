#!/usr/bin/env python
import numpy as np
import util

def add_noise(x, v):
    return x + np.random.standard_normal(x.shape)*(.5*v)**.5 + 1j * np.random.standard_normal(x.shape)*(.5*v)**.5

def channelModel(output, lsnr):
    # received symbols
    input = np.copy(np.r_[output, np.zeros(32)])
    freq_offset = 232e3 * (2*np.random.uniform()-1)
    input *= np.exp(2*np.pi*1j*freq_offset*np.arange(input.size)/20e6)
    phase_offset = 2*np.pi*np.random.uniform()
    input *= np.exp(1j*phase_offset)
    # random causal 16-tap FIR filter with exponential tail
    h = (np.random.standard_normal(16) + 1j*np.random.standard_normal(16)) * np.exp(-np.arange(16)*5.4/16)
    h[0] += 4.
    h /= np.sum(np.abs(h)**2)**.5
    input = np.convolve(input, h, 'same')
    input = util.upsample(input, 16)
    time_offset = np.random.random_integers(0, 15)
    input = input[time_offset::16]
    snr = 10.**(.1*lsnr)
    input = add_noise(input, np.var(input)*64./52./snr)
    return input
