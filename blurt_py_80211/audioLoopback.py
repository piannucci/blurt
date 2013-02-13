#!/usr/bin/env python
import numpy as np
import audio
import util, iir

def processOutput(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise):
    output = util.upsample(output, upsample_factor)
    output = (output * np.exp(1j * 2 * np.pi * np.arange(output.size) * loopback_Fc / loopback_Fs)).real
    output *= 1.0 / np.percentile(np.abs(output), 95)
    output = np.r_[output, np.zeros(int(.1*loopback_Fs))]
    if mask_noise is not None:
        if mask_noise.size > output.size:
            output = np.r_[output, np.zeros(mask_noise.size-output.size)]
        output[:mask_noise.size] += mask_noise[:output.size]
    return output

def autocorrelate(input):
    autocorr = input[16:] * input[:-16].conj()
    autocorr = autocorr[:16*(autocorr.size//16)].reshape(autocorr.size//16, 16).sum(1)
    return np.convolve(np.abs(autocorr), np.ones(9), 'same')

def processInput(input, loopback_Fs, loopback_Fc, upsample_factor):
    input = input * np.exp(-1j * 2 * np.pi * np.arange(input.size) * loopback_Fc / loopback_Fs)
    input = iir.lowpass(.8/upsample_factor)(np.r_[np.zeros(6), input])[6::upsample_factor]
    if 1:
        # find STS by autocorrelation
        score = autocorrelate(input)
        input = input[16*np.argmax(score)-72:]
    else:
        # energy detector
        input_max = 1.5 * np.std(np.abs(input)) + np.mean(np.abs(input))
        input_mask = np.convolve((np.abs(input) > input_max*2e-1), np.ones(8), 'same') > 0
        input = input[np.where(input_mask)]
    return input

def audioLoopback(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise):
    output = processOutput(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise)
    input = audio.play_and_record(output, loopback_Fs)
    return processInput(input, loopback_Fs, loopback_Fc, upsample_factor)

def audioOut(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise):
    output = processOutput(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise)
    audio.play(output, loopback_Fs)

def audioIn(loopback_Fs, loopback_Fc, upsample_factor):
    input = audio.record(int(loopback_Fs*5.), loopback_Fs)
    return processInput(input, loopback_Fs, loopback_Fc, upsample_factor)
