#!/usr/bin/env python
import numpy as np
import util
delay = .005
Fs = 48000
reps = 100
tail_samples = int(Fs*2.)
delay_samples = int(delay*Fs)
signal = np.tile(np.r_[np.random.standard_normal(Fs*2)*.1, np.zeros(Fs*2)], reps)
signal = np.vstack((np.r_[np.zeros(delay_samples), signal], np.r_[signal, np.zeros(delay_samples)])).T
signal = np.vstack((signal, np.zeros((tail_samples, 2))))
util.writewave('noise.wav', signal, Fs, 3)
