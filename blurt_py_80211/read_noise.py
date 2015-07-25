#!/usr/bin/env python
import numpy as np, pylab as pl, scipy.linalg
import util, iir
import sys

def processWave(fn):
    signal, Fs = util.readwave(fn)

    signal = np.r_[np.zeros(Fs/2), signal]

    alternation_interval = Fs*2
    steady_interval = Fs*1.5

    # we need to classify the sections of noise and silence, identify the
    # steady-state intervals, FFT them, and label them, then write out the results.
    signal *= 1./signal.std()

    def downsample(x, ratio):
        order = 6
        return iir.lowpass(1./(2*ratio), method='Ch -.2', order=order)(np.r_[np.zeros(order), x])[order::ratio]

    def peak_detect(input, window_size):
        # look for points outstanding in their neighborhood
        # by explicit comparison
        l = window_size
        M = scipy.linalg.toeplitz(np.r_[input, np.zeros(2*l)], np.zeros(2*l+1)).T
        ext_input = np.r_[np.zeros(l), input, np.zeros(l)]
        M[:l] = M[:l] < ext_input
        M[l+1:] = M[l+1:] < ext_input
        M[l] = True
        return np.where(M.all(0))[0] - l

    ratio = 32*15*25
    envelope = iir.lowpass(.125)(downsample(downsample(downsample(np.abs(signal), 32), 15), 25))
    peaks = peak_detect(envelope, 8) * ratio - 77840

    radius = steady_interval/2
    subintervals = 75
    intervals = ([], [])
    S_Y_samples = []
    S_N_samples = []
    for p in peaks:
        if p < radius or p + alternation_interval + radius >= signal.size: continue
        intervals[0].append((p-radius,p+radius))
        noise = signal[p-radius:p+radius]
        noise = np.fft.fft(noise.reshape(subintervals, steady_interval/subintervals), axis=1)
        S_Y_samples.append(noise.var(0))
        intervals[1].append((p+alternation_interval-radius,p+alternation_interval+radius))
        silence = signal[p+alternation_interval-radius:p+alternation_interval+radius]
        silence = np.fft.fft(silence.reshape(subintervals, steady_interval/subintervals), axis=1)
        S_N_samples.append(silence.var(0))

    bins = steady_interval/subintervals/2
    # note: S_Y = P + N, S_N = N, so S_Y/S_N = 1 + P/N
    capacity = np.array([np.log2(np.clip(S_Y/S_N, 1., np.inf)) for S_Y, S_N in zip(S_Y_samples, S_N_samples)]).mean(0)[:bins]
    S_X = bins * 2 * .1**2
    S_H = np.array([np.clip(S_Y-S_N,0,np.inf)/S_X for S_Y, S_N in zip(S_Y_samples, S_N_samples)])[:,:bins]
    H_mag = S_H.mean(0)**.5
    S_Y = np.array(S_Y_samples).mean(0)[:bins]
    S_N = np.array(S_N_samples).mean(0)[:bins]
    df = Fs/float(steady_interval/subintervals)
    f = df*np.arange(bins)
    visualize = False
    if visualize:
        pl.figure()
        pl.subplot(221)
        pl.plot(f, 10*np.log10(S_Y-S_N))
        pl.plot(f, 10*np.log10(S_N))
        pl.xlim(0, f.max())
        pl.subplot(223)
        pl.plot(f, capacity)
        pl.xlim(0, f.max())
        pl.xlabel('Frequency (Hz)')
        pl.ylabel('Capacity (bits/s/Hz)')
        pl.subplot(122)
        pl.plot(f, capacity.cumsum()*df)
        pl.xlim(0, f.max())

    test_synchronization = False
    if test_synchronization:
        pl.clf()
        lp = lambda x:iir.lowpass(.005)(np.r_[np.zeros(6), x])[6:]
        pl.plot(lp(np.abs(signal))*.5)
        pl.plot(np.arange(envelope.size) * ratio - 77840, envelope)
        for x in peaks:
            pl.gca().axvline(x, color='k')
        noise_interval = np.zeros(signal.size, bool)
        for start, end in intervals[0]:
            noise_interval[start:end] = True
        silence_interval = np.zeros(signal.size, bool)
        for start, end in intervals[1]:
            silence_interval[start:end] = True
        import matplotlib.transforms as mtransforms
        ax = pl.gca()
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(np.arange(signal.size), 0, 1, where=noise_interval, facecolor='red', alpha=0.5, transform=trans)
        ax.fill_between(np.arange(signal.size), 0, 1, where=silence_interval, facecolor='magenta', alpha=0.5, transform=trans)
    return df, H_mag, S_N, capacity, len(peaks)

#processWave(sys.argv[1])
directory = '/Users/peteriannucci/Desktop/sipb_office_results/'
fn_template = directory + '%d%d.wav'

# get shape of various arrays
df, H_mag, S_N, capacity, peaks = processWave(fn_template % (1,1))

bins = H_mag.size

H_mag_all = np.empty((5,6,bins), float)
S_N_all = np.empty((5,6,bins), float)
capacity_all = np.empty((5,6,bins), float)

for x in range(1,7):
    for y in range(1,6):
        fn = fn_template % (y, x)
        df, H_mag, S_N, capacity, peaks = processWave(fn)
        H_mag_all[y-1,x-1] = H_mag
        S_N_all[y-1,x-1] = S_N
        capacity_all[y-1,x-1] = capacity
        assert peaks==5
