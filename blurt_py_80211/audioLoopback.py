#!/usr/bin/env python
import numpy as np
import audio, audio.stream
import util, iir

delay = .005

def processOutput(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise):
    output = util.upsample(output, upsample_factor)
    output = (output * np.exp(1j * 2 * np.pi * np.arange(output.size) * loopback_Fc / loopback_Fs)).real
    output *= 1.0 / np.percentile(np.abs(output), 95)
    output = np.r_[output, np.zeros(int(.1*loopback_Fs))]
    if mask_noise is not None:
        if mask_noise.size > output.size:
            output = np.r_[output, np.zeros(mask_noise.size-output.size)]
        output[:mask_noise.size] += mask_noise[:output.size]
    delay_samples = int(delay*loopback_Fs)
    # delay one channel slightly relative to the other:
    # this breaks up the spatially-dependent frequency-correlated
    # nulls of our speaker array
    output = np.vstack((np.r_[np.zeros(delay_samples), output],
                        np.r_[output, np.zeros(delay_samples)])).T
    return output

def processInput(input, loopback_Fs, loopback_Fc, upsample_factor):
    input = input * np.exp(-1j * 2 * np.pi * np.arange(input.size) * loopback_Fc / loopback_Fs)
    input = iir.lowpass(.8/upsample_factor)(np.r_[np.zeros(6), input])[6::upsample_factor]
    return input

def audioLoopback(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise):
    output = processOutput(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise)
    input = audio.play_and_record(output, loopback_Fs)
    return processInput(input, loopback_Fs, loopback_Fc, upsample_factor)

def audioOut(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise):
    output = processOutput(output, loopback_Fs, loopback_Fc, upsample_factor, mask_noise)
    audio.play(output, loopback_Fs)

def audioIn(loopback_Fs, loopback_Fc, upsample_factor, duration=5.):
    input = audio.record(int(loopback_Fs*duration), loopback_Fs)
    return processInput(input, loopback_Fs, loopback_Fc, upsample_factor)

def downconverter(next, loopback_Fs, loopback_Fc, upsample_factor):
    class Downconverter(audio.stream.ThreadedStream):
        def init(self):
            super(Downconverter, self).init()
            self.next = self.kwargs['next']
            self.iir_state = np.zeros(6)
            self.iir = iir.lowpass(.8/upsample_factor)
            self.i = 0
            self.s = -1j * 2 * np.pi * loopback_Fc / loopback_Fs
        def thread_consume(self, input):
            input = input * np.exp(self.s * (self.i + np.arange(input.size)))
            self.i += input.size
            input = self.iir(np.r_[self.iir_state, input])
            self.iir_state = input[-6:]
            input = input[6::upsample_factor]
            self.next.consume(input)
    return Downconverter(next=next)
