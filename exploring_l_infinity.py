#!/usr/bin/env python
from pylab import *
from scipy.optimize import *
from numpy import *
import time

N = 64
upsample_factor = 16
fftsize = N * upsample_factor
F = matrix(exp(1j*2*pi*arange(fftsize)[:,newaxis]*arange(fftsize)[newaxis,:]/float(fftsize)) / fftsize)
M = hstack((F[:,0], F[:,27:32], F[:,-32:-26]))

def f(x):
    return absolute(array(z + M*x)).max()

def df(x, p=10):
    y = array(z + M*x)
    ay = absolute(y)
    y = ay**(p-2) * y.conj()
    #y = (ay==ay.max()) * y.conj()
    u = (y * array(M)).sum(0)
    u *= (absolute(u)**2).sum()**-.5
    return matrix(u.conj().reshape(12,1))

def step(x, dx):
    l = 1e-3
    r = 2.
    obj = f(x)
    last_l = 0.
    while True:
      obj2 = f(x - dx * l)
      if obj2 > obj:
        break
      obj = obj2
      last_l = l
      l *= r
    return x - dx * last_l, last_l

def go(count, visualize=False):
    global z
    j = 0
    results = []
    while j < count:
        Z = (random.standard_normal(64) + 1j * random.standard_normal(64)) * .5**.5
        Z[0] = 0.
        Z[27:38] = 0.
        Z = r_[Z[:32], zeros(N*(upsample_factor-1)), Z[32:]]
        fftsize = N * upsample_factor
        z = matrix(fft.ifft(Z)).T
        x = matrix(zeros((12,1),complex128))
        for i in xrange(20):
            x, l = step(x, df(x, 10))
            if l < 1e-4:
                break
        y = z + M*x
        L0 = amax(abs(array(z)))
        L1 = amax(abs(array(y)))
        j += 1
        results.append((L0, L1, mean(abs(array(z))**2), mean(abs(array(y))**2)))
        if visualize:
            clf()
            subplot(221)
            title('Real part')
            plot(z.real, label='$\mathbf{z}$')
            plot(y.real, label='$\mathbf{z}+\mathbf{Mx}$')
            hlines([L0,-L0,L1,-L1], 0, fftsize, linestyles='dashed')
            xlim(0,fftsize-1)
            xlabel('Imaginary part')
            legend()
            subplot(222)
            title('Imaginary part')
            plot(z.imag)
            plot(y.imag)
            hlines([L0,-L0,L1,-L1], 0, fftsize, linestyles='dashed')
            xlim(0,fftsize-1)
            subplot(223)
            axis('scaled')
            plot(z.real, z.imag)
            plot(y.real, y.imag)
            scatter(array(z.real[::upsample_factor]), array(z.imag[::upsample_factor]), facecolor='blue')
            scatter(array(y.real[::upsample_factor]), array(y.imag[::upsample_factor]), facecolor='green')
            plot(L0*cos(2*pi*arange(201)/200.), L0*sin(2*pi*arange(201)/200.), '--k')
            plot(L1*cos(2*pi*arange(201)/200.), L1*sin(2*pi*arange(201)/200.), '--k')
            xlim(-L0*1.1, L0*1.1)
            ylim(-L0*1.1, L0*1.1)
            subplot(224)
            distortion = 10*log10(abs(fft.fft(array(y).flatten()) - Z))
            distortion = r_[distortion[:32], distortion[-32:]]
            plot(distortion)
            draw()
    return array(results)

def processResults(a, p=99):
    bins = arange(0, 5001) * amax(a[:,0]) / 5000.
    gain = percentile(a[:,1], p) / percentile(a[:,0], p)
    power_ratio = a[:,3].mean()/a[:,2].mean()
    change_SNR = 10*log10((1./gain)**2)
    change_power = 10*log10((1./gain)**2) + 10*log10(power_ratio)
    clf()
    h0 = histogram(a[:,0], bins)[0]
    x = (bins[1:] + bins[:-1])*.5
    plot(x, 10*log10(1.-cumsum(h0)/float(sum(h0))))
    h1 = histogram(a[:,1], bins)[0]
    plot(x, 10*log10(1.-cumsum(h1)/float(sum(h1))))
    print 'can increase SNR by %.4f dB' % change_SNR
    print 'while increasing power by %.4f dB' % change_power
    print 'for a power efficiency of %.1f%%' % (100 * change_SNR / change_power)
