#!/usr/bin/env python
import numpy as np
import wave, util, iir

def prepareMaskNoise(fn, Fs, Fc, upsample_factor):
    f = wave.open(fn)
    nframes = f.getnframes()
    dtype = [None, np.uint8, np.int16, None, np.int32][f.getsampwidth()]
    frames = np.fromstring(f.readframes(nframes), dtype).astype(float)
    frames = frames.reshape(nframes, f.getnchannels())
    frames = frames.mean(1)
    frames = util.upsample(frames, Fs/f.getframerate())
    frames /= np.amax(np.abs(frames))
    # band-stop filter for data
    frames *= np.exp(-1j * 2 * np.pi * np.arange(frames.size) * Fc / Fs)
    frames = iir.highpass(.8/upsample_factor)(frames)
    frames *= np.exp(2j * 2 * np.pi * np.arange(frames.size) * Fc / Fs)
    frames = iir.highpass(.8/upsample_factor)(frames)
    frames *= np.exp(-1j * 2 * np.pi * np.arange(frames.size) * Fc / Fs)
    frames = frames.real
    # look for beginning and end of noise
    envelope = iir.lowpass(.01)(np.r_[np.zeros(6), np.abs(frames)])[6:]
    start = np.where(envelope > np.amax(envelope)*.01)[0][0]
    end = np.where(envelope > np.amax(envelope)*1e-3)[0][-1]
    return frames[start:end]
