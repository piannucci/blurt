#!/usr/bin/env python
import numpy as np
import pylab as pl
import sys
a = open(sys.argv[1], 'r').read()
a = a.strip().split('\n')
a = [aa.split(' @ ') for aa in a if not aa.startswith('#')]
a = [(eval(aa[0]), float(aa[1].split(' ')[0])) for aa in a]
ids = np.array([int(aa[0][:6], 10) for aa in a], int)
id_first = ids[0]
id_last = ids[-1]
idx = ids-id_first
N = id_last-id_first+1
success = np.zeros(N, bool)
snr = np.empty(success.size, float)
success[idx] = True
snr[:] = -np.inf
snr[idx] = np.array([aa[1] for aa in a])
y_hist, x_hist = np.histogram(snr, bins=20000, range=(0, snr.max()))
y_hist = (y_hist.cumsum() + (snr<0).sum()) / float(snr.size)
y_hist_bound_center = (y_hist + 2. / N) / (4. / N + 1.)
y_hist_bound_radius = (y_hist_bound_center**2 - y_hist**2 / (4. / N + 1.))**.5
x_hist = (x_hist[1:] + x_hist[:-1])*.5

runs = np.zeros(10, int)
nruns = 0
run = 0
for i in xrange(success.size):
    if not success[i]:
        run += 1
    else:
        if run < runs.size:
            runs[run] += 1
        run = 0
        nruns += 1

runs = runs.astype(float) / nruns
# Solve for bounds s.t. x = x_upper - 2 std(x_upper) = x_lower + 2 std(x_lower)
runs_bound_center = (runs + 2. / nruns) / (4. / nruns + 1.)
runs_bound_radius = (runs_bound_center**2 - runs**2 / (4. / nruns + 1.))**.5

p = 1./((runs * np.arange(runs.size)).sum() + 1.)

pl.figure()
pl.subplot(211)
pl.title('Distribution of Signal-to-Noise Ratio')
pl.xlabel('SNR (dB)')
pl.ylabel('Cumulative Probability')
pl.plot(x_hist, y_hist)
pl.fill_between(x_hist, y_hist_bound_center+y_hist_bound_radius,
                y_hist_bound_center-y_hist_bound_radius,
                facecolor=(0.,0.,1.,.5),
                linewidth=0.)
pl.ylim(0,1)
pl.xlim(0, snr.max())
pl.subplot(212)
pl.title('Distribution of Error Run Lengths')
pl.xlabel('Consecutive Errors')
pl.ylabel('Probability')
pl.fill_between(np.arange(runs.size),
                np.log10(runs_bound_center+runs_bound_radius),
                np.log10(np.clip(runs_bound_center-runs_bound_radius, 1e-10, np.inf)),
                facecolor=(0.,0.,1.,.5),
                linewidth=0.)
ys = xrange(-3,1)
pl.yticks(ys, ['$10^{%d}$' % y for y in ys])
pl.ylim(round(np.log10(runs[np.where(runs)]).min()))
pl.plot(np.log10(runs), label='Observed Probability')
pl.plot(np.log10((1-p)**np.arange(runs.size)*p), 'k--', label='Geometric fit')
pl.legend()
