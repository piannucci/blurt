import pylab as pl
import numpy as np
import upsample
import iir

Fs = 96000
Fc = 20e3
upsample_factor = 32
nfft = 64

if 0:
    a = 1.5**.5
    x = (np.random.uniform(-a, a, (100, 64)) + np.random.uniform(-a, a, (100, 64)) * 1j)
    outsize = x.size*5//4 * upsample_factor

    n = np.random.standard_normal(outsize) * 10.**(-30 / 20)
    trajectory = iir.stagedLowpass(3e-5)(np.random.standard_normal(outsize)) * 10
    trajectory = trajectory[:outsize]
    # trajectory is in meters
    cs = 340. # m/s
    x[:,0] = 0
    x[:,27:38] = 0
    y = np.tile(np.fft.ifft(x, axis=1), (1,3))[:,48:129]
    y[1:,0] = (y[1:,0] + y[:-1,-1]) * .5
    y = upsample.upsample(y[:,:80].flatten(), upsample_factor)
    y = (y * np.exp(1j*2*np.pi*Fc*np.arange(y.size)/Fs)).real

    pl.figure()
    _=pl.specgram(y+n, NFFT=64*upsample_factor, noverlap=-16*upsample_factor, interpolation='nearest', Fs=Fs)

    t0 = np.arange(y.size)/Fs
    t = t0 - trajectory / cs
    y2 = np.interp(t, t0, y.real)
    y2 = iir.lowpass(1./upsample_factor)(y2 * np.exp(-1j*2*np.pi*Fc*np.arange(y.size)/Fs))[::upsample_factor/4]

    pl.figure()
    _=pl.specgram(y2+n[:y2.size], NFFT=64*4, noverlap=-16*4, interpolation='nearest', Fs=Fs/(upsample_factor/4))

    noisy_y2 = y2 + n[:y2.size]

    z = (np.ediff1d(y2, to_end=0.).conj() * y2 * (1j * np.pi * 2)).real

    w = ((np.arange(64*4) + 32*4) % (64*4) - 32*4)
    w = np.where(abs(w) < 32, w, 0)
    e = abs(np.fft.fft((y2+n[:y2.size]).reshape(-1,80*4)[:,16*4:] * np.blackman(64*4), axis=1))**2

    pl.clf()
    _=pl.specgram(y2+n[:y2.size], NFFT=64*4, noverlap=-16*4, interpolation='nearest', Fs=Fs/(upsample_factor/4))
    pl.plot(np.arange(100)*80*upsample_factor/Fs, 1200 + .1 *(e*w).sum(1))
    pl.plot(np.arange(trajectory.size/(upsample_factor/4))*(upsample_factor/4)/Fs, np.ediff1d(trajectory[::upsample_factor/4], to_end=0.)*10000000)

def dopplerDecode_baseline(y, d0, d1):
    # y: samples of periodic signal at [0,64)
    # [d0, d1+64): range of coordinates of desired samples
    phaseError = 2*np.pi*(d0+np.arange(nfft)*(d1-d0)/nfft)*upsample_factor*Fc/Fs
    return np.fft.fft(y * np.exp(-1j*phaseError))

def dopplerDecode(y, d0, d1):
    # y: samples of periodic signal at [0,64)
    # [d0, d1+64): range of coordinates of desired samples
    ramp = np.arange(nfft) / nfft
    omega = 2*np.pi * (((np.arange(64) + 32) % 64 - 32) / 64)
    y2 = y * np.exp(-1j*2*np.pi*(d0+np.arange(nfft)*(d1-d0)/nfft)*upsample_factor*Fc/Fs)
    return (np.fft.fft(np.diag(y2), axis=1) * np.exp(-1j*omega*(ramp*2-1)*(d1-d0))).sum(0)
    return np.fft.fft(y2*(1-ramp)) * np.exp(-1j*omega*(d0-d1)*.333) + \
           np.fft.fft(y2*   ramp ) * np.exp(-1j*omega*(d1-d0)*.333)

i = np.arange(200)
doppler = 1+(i-i.size/2)*.0002
symbol = (np.random.random_integers(0, 3, (nfft,2)) * np.array([1,1j])).sum(-1) * 2 - (3+3j)
symbol[0] = symbol[27:38] = 0
x = np.tile(np.fft.ifft(np.r_[symbol[:nfft/2], np.zeros(nfft*(upsample_factor-1)), symbol[nfft/2:]]), 3)
t_x = (np.arange(nfft*upsample_factor*3) - nfft*upsample_factor) / Fs
x *= np.exp(1j*2*np.pi*Fc*t_x)
y = np.zeros((i.size, nfft), np.complex128)

for ii in i:
    t_y = np.arange(64) * upsample_factor * doppler[ii] / Fs
    y[ii] = np.interp(t_y, t_x, x.real) + 1j*np.interp(t_y, t_x, x.imag)

t_y = np.arange(nfft)*upsample_factor/Fs
y *= np.exp(-1j*2*np.pi*Fc*t_y)

Y = np.fft.fft(y, axis=1)
Z = np.array([dopplerDecode(y[ii], 0, 64*(doppler[ii]-1)) for ii in i])
Z_baseline = np.array([dopplerDecode_baseline(y[ii], 0, 64*(doppler[ii]-1)) for ii in i])

pl.figure(1)
pl.clf()
pl.imshow(abs(np.fft.fftshift(Y-Y[i.size/2], axes=1)).T, aspect='auto', extent=(doppler[0], doppler[-1] + doppler[1]-doppler[0], -31.5, 32.5), interpolation='nearest', vmin=0., vmax=.2)

pl.figure(2)
pl.clf()
pl.imshow(abs(np.fft.fftshift(Z_baseline-Y[i.size/2], axes=1)).T, aspect='auto', extent=(doppler[0], doppler[-1] + doppler[1]-doppler[0], -31.5, 32.5), interpolation='nearest', vmin=0., vmax=.2)

pl.figure(3)
pl.clf()
pl.imshow(abs(np.fft.fftshift(Z-Y[i.size/2], axes=1)).T, aspect='auto', extent=(doppler[0], doppler[-1] + doppler[1]-doppler[0], -31.5, 32.5), interpolation='nearest', vmin=0., vmax=.2)

trialDoppler = np.arange(-.005, .005, .00001)
phaseError = np.exp(-1j*2*np.pi*(np.arange(nfft)*(trialDoppler * 64)[:,None]/nfft)*upsample_factor*Fc/Fs)
def dedop(y):
    d = trialDoppler[(abs(np.fft.fft(y * phaseError, axis=1)[:,27:38])**2).sum(1).argmin()]
    return dopplerDecode(y, 0, d*64)

error1 = trial_doppler[np.array([est1(yy) for yy in y])] - doppler
error2 = trial_doppler[np.array([(abs(np.array([dopplerDecode(y[ii], 0, 64*(d-1)) for d in trial_doppler])[:,27:38])**2).sum(1).argmin() for ii in i])] - doppler
pl.figure(4)
pl.plot(doppler, error1)
pl.plot(doppler, error2)

#symbol = np.eye(64)
#symbol = np.concatenate((symbol[1:27], symbol[38:]), 0)
#x = np.tile(32*np.fft.ifft(np.concatenate((symbol[:,:32], np.zeros((symbol.shape[0], 64*31)), symbol[:,32:]), 1), axis=1), (1,3))
#y = np.zeros((x.shape[0], 64), np.complex128)
#doppler = 2-1.003
#for i in range(x.shape[0]):
#    y[i].real = np.interp(np.arange(64) * doppler, np.arange(2048 * 3)/32 - 64, x[i].real)
#    y[i].imag = np.interp(np.arange(64) * doppler, np.arange(2048 * 3)/32 - 64, x[i].imag)
#
#K = np.exp(np.log(np.linalg.eigvalsh(np.dot(y, y.conj().T))).ptp())
#print('Condition number: ', K)
#
#pl.clf()
#pl.imshow(np.log10(abs(np.fft.fft(y))), interpolation='nearest')
#pl.colorbar()
