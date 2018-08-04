#!/usr/bin/env python
import audio
import numpy as np
import pylab as pl
import matplotlib.animation as animation
import iir
import scipy.signal
import time

def chirp(f0, f1, length):
    return np.exp(1j*2*np.pi*(np.arange(length)*float(f1-f0)/length + f0).cumsum())

class SonarTransciever(audio.stream.Visualizer):
    def __init__(self):
        self.Fs = Fs = 96e3
        self.pulse_low = 10e3
        self.pulse_high = 30e3
        self.pulse_duration = int(.1 * Fs)
        self.display_duration = 3.
        self.spacing = spacing = int(.02 * Fs)
        simultaneous_pulses = (self.pulse_duration+2*spacing-1) // (2*spacing)
        self.base_chirp = chirp(self.pulse_low/Fs, self.pulse_high/Fs, self.pulse_duration)
        pulse = self.base_chirp.real
        pulse *= np.clip(np.arange(self.pulse_duration, dtype=float)/(Fs*.025), 0, 1)
        pulse *= np.clip(-np.arange(-self.pulse_duration, 0, dtype=float)/(Fs*.025), 0, 1)
        mono_timing = np.tile(np.r_[1, np.zeros(spacing*2-1)], 3*simultaneous_pulses)
        mono = scipy.signal.fftconvolve(mono_timing, pulse, 'full')[(simultaneous_pulses-1)*spacing*2:simultaneous_pulses*spacing*2]
        self.output = np.tile(np.vstack((mono, np.roll(mono, spacing))).T, (20,1))
        self.lpf = iir.IIRFilter.lowpass((self.pulse_high-self.pulse_low)/2/Fs)
        cs = 1116.44 # feet/sec
        self.columns = int(self.display_duration*Fs) // (spacing*2)
        self.buffer = np.zeros((2, spacing, self.columns*2), np.uint16)
        self.window_phase = 0
        self.input_fragment = np.zeros(0, np.complex64)
        self.extent = (0, self.display_duration, -spacing*.1/Fs*cs/2, spacing*.9/Fs*cs/2)
        self.i = 0
        self.phase_locked = None
        super().__init__(channels=2, outBufSize=16384, inBufSize=8192)
        self.fig.delaxes(self.ax)
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.im1 = self.ax1.imshow(self.buffer[0], vmin=0., vmax=10., extent=self.extent, aspect='auto', interpolation='nearest')
        self.ax1.set_xlim(self.extent[:2])
        self.ax1.set_ylim(self.extent[2:])
        self.ax1.grid()
        self.ax2 = self.fig.add_subplot(1,2,2)
        self.im2 = self.ax2.imshow(self.buffer[1], vmin=0., vmax=10., extent=self.extent, aspect='auto', interpolation='nearest')
        self.ax2.set_xlim(self.extent[:2])
        self.ax2.set_ylim(self.extent[2:])
        self.ax2.grid()
        self.ax1.patch.set_visible(False)
        self.ax2.patch.set_visible(False)
        self.fig.patch.set_facecolor('k')
        audio.stream.set_backgroundcolor(self.ax1, 'k')
        audio.stream.set_foregroundcolor(self.ax1, 'w')
        audio.stream.set_backgroundcolor(self.ax2, 'k')
        audio.stream.set_foregroundcolor(self.ax2, 'w')
        self.cm = pl.get_cmap('jet')(np.arange(0,65536)/65535., bytes=True)
    def thread_produce(self):
        return self.output
    def thread_consume(self, input):
        stream = np.r_[self.input_fragment, input]
        advance = 2*self.spacing
        lookahead = self.pulse_duration-1 # needed for convolution
        lookahead += advance # needed for flash suppression
        if self.phase_locked is None:
           lookahead += self.spacing-1 # needed for phase lock
        n = max(0, (stream.size-lookahead) // advance)
        actual_advance = n*advance
        if n > 0:
            try:
                y = stream[:(n+1)*advance+lookahead]
                Omega_carrier = 2*np.pi*(self.pulse_low+self.pulse_high)*.5/self.Fs
                carrier = np.exp(1j*Omega_carrier*np.arange(self.i, self.i+y.size))
                y = self.lpf(carrier.conj() * y) * carrier
                y = scipy.signal.fftconvolve(y,self.base_chirp.conj()[::-1],'valid')
                if self.phase_locked is None:
                    self.phase_locked = (np.abs(y).argmax()-int(self.spacing*.1)) % self.spacing
                    y = y[self.phase_locked:]
                    actual_advance += self.phase_locked
                y = y[:(n+1)*advance].reshape(n+1,2,self.spacing).transpose(1,0,2)
                y = np.log10(np.abs(np.diff(y, axis=1))).transpose(0,2,1)[:,::-1]
                y *= (y>0)
                y *= 65536 / y.max()
                y = np.clip(y, 0, 65536)
                cols = self.columns
                phase = self.window_phase
                if phase+cols+n <= 2*cols:
                    self.buffer[:,:,phase+cols:phase+cols+n] = y
                else:
                    m = phase+n-cols
                    self.buffer[:,:,m-n:] = y[:,:,:n-m]
                    self.buffer[:,:,0:m] = y[:,:,-m:]
                self.window_phase = (phase + n) % cols
                self.buffer[:,:,phase:phase+n] = y
            except (Exception,) as e:
                import traceback
                traceback.print_exc()
        self.input_fragment = stream[actual_advance:]
        self.i += actual_advance
    def updatefig(self, *args):
        t = self.i / self.Fs
        self.extent = (t, t+self.display_duration) + self.extent[2:]
        self.im1.set_extent(self.extent)
        self.im2.set_extent(self.extent)
        self.ax1.set_xlim(t, t+self.display_duration)
        self.ax2.set_xlim(t, t+self.display_duration)
        ticks = np.arange(t, t+self.display_duration+1e-4, .5)
        ticklabels = ['%.1f' % x for x in ticks]
        self.ax1.set_xticks(ticks)
        self.ax2.set_xticks(ticks)
        self.ax1.set_xticklabels(ticklabels)
        self.ax2.set_xticklabels(ticklabels)
        phase = self.window_phase
        b = self.buffer[:,:,phase:phase+self.columns]
        self.im1.set_data(self.cm[b[0]])
        self.im2.set_data(self.cm[b[1]])

Fs = 96e3
xcvr = SonarTransciever()
ap = audio.AudioInterface()
try:
    ap.play(xcvr, Fs)
    ap.record(xcvr, Fs)
    ani = animation.FuncAnimation(xcvr.fig, xcvr.updatefig, interval=10, blit=False)
    pl.show(block=True)
except KeyboardInterrupt:
    ap.stop()
