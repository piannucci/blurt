#!/usr/bin/env python
import numpy as np
import audio
import util, iir

def processInput(input, loopback_Fs, loopback_Fc, upsample_factor):
    input = input * np.exp(-1j * 2 * np.pi * np.arange(input.size) * loopback_Fc / loopback_Fs)
    input = iir.lowpass(.8/upsample_factor)(input)
    input_max = 1.5 * np.std(np.abs(input)) + np.mean(np.abs(input))
    input_mask = np.convolve((np.abs(input) > input_max*2e-1), np.ones(upsample_factor*8), 'same') > 0
    input = input[np.where(input_mask)]
    input = input[::upsample_factor]
    return input

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
