#!/usr/bin/env python
from pylab import *
from scipy.optimize import *
from numpy import *

N = 64
upsample_factor = 16
F = matrix(exp(1j*2*pi*arange(fftsize)[:,newaxis]*arange(fftsize)[newaxis,:]/float(fftsize)) / fftsize)
M = hstack((F[:,0], F[:,27:32], F[:,-32:-26]))

def f(x):
    return amax(abs(array(z + M*matrix(x[:12] + 1j*x[12:]).T)))

while True:
    Z = (random.standard_normal(64) + 1j * random.standard_normal(64)) * .5**.5
    Z[0] = 0.
    Z[27:38] = 0.
    Z = r_[Z[:32], zeros(N*(upsample_factor-1)), Z[32:]]
    fftsize = N * upsample_factor
    z = matrix(fft.ifft(Z)).T
    x = minimize(f, zeros(24)).x
    x = matrix(x[:12] + 1j*x[12:]).T
    L0 = amax(abs(array(z)))
    L1 = amax(abs(array(z+M*x)))
    clf()
    subplot(221)
    plot(z.real)
    plot((z+M*x).real)
    hlines([L0,-L0,L1,-L1], 0, fftsize, linestyles='dashed')
    subplot(222)
    plot(z.imag)
    plot((z+M*x).imag)
    hlines([L0,-L0,L1,-L1], 0, fftsize, linestyles='dashed')
    subplot(223)
    axis('scaled')
    plot(z.real, z.imag)
    plot((z+M*x).real, (z+M*x).imag)
    plot(L0*cos(2*pi*arange(201)/200.), L0*sin(2*pi*arange(201)/200.), '--k')
    plot(L1*cos(2*pi*arange(201)/200.), L1*sin(2*pi*arange(201)/200.), '--k')
    xlim(-L0*1.1, L0*1.1)
    ylim(-L0*1.1, L0*1.1)
    subplot(224)
    distortion = 10*log10(abs(fft.fft(array(z+M*x).flatten()) - Z))
    distortion = r_[distortion[:32], distortion[-32:]]
    plot(distortion)
    print 'Reduced L_infty from %f to %f' % (L0, L1)
    draw()
