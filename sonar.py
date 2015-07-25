import audio, numpy as np, pylab as pl, iir, scipy.signal
outputUID = 'AppleHDAEngineOutput:1B,0,1,1:0'
externalMicUID = u'AppleUSBAudioEngine:Creative Technology:SB Live! 24-bit External:14100000:2,1'
internalMicUID = u'AppleHDAEngineInput:1B,0,1,0:1'
output = [i for i,d in enumerate(audio.devices()) if d['deviceUID'] == outputUID][0]
externalMic = [i for i,d in enumerate(audio.devices()) if d['deviceUID'] == externalMicUID][0]
internalMic = [i for i,d in enumerate(audio.devices()) if d['deviceUID'] == internalMicUID][0]
Fs=96000

def chirp(f0, f1, length):
    return np.exp(1j*2*np.pi*(np.arange(length)*float(f1-f0)/length + f0).cumsum())

class SonarTransciever(audio.stream.Visualizer):
    def init(self):
        self.channels = 2
        self.Fs = Fs = 96e3
        self.pulse_low = 20e3
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
        self.output = np.vstack((mono, np.roll(mono, spacing))).T
        self.lpf = iir.lowpass((self.pulse_high-self.pulse_low)/Fs/2, continuous=True, dtype=np.complex128)
        cs = 1116.44 # feet/sec
        self.columns = int(self.display_duration*Fs) // (spacing*2)
        self.buffer = np.zeros((2, spacing, self.columns*2), np.float32)
        self.window_phase = 0
        self.input_fragment = np.zeros(0, np.complex64)
        super(SonarTransciever, self).init()
        self.extent = (0, self.display_duration, -spacing*.1/Fs*cs/2, spacing*.9/Fs*cs/2)
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
        self.i = 0
        self.phase_locked = None
    def thread_produce(self):
        return self.output
    def thread_consume(self, input):
        stream = np.r_[self.input_fragment, input]
        advance = 2*self.spacing
        lookahead = self.pulse_duration-1 # needed for convolution
        lookahead += advance # needed for flash suppression
        if self.phase_locked is None:
           lookahead += self.spacing-1 # needed for phase lock
        n = (stream.size-lookahead) // advance
        if n < 40: n = 0
        actual_advance = n*advance
        if n > 0:
            try:
                y = stream[:(n+1)*advance+lookahead]
                Omega_carrier = 2*np.pi*(self.pulse_low+self.pulse_high)*.5/self.Fs
                carrier = np.exp(1j*Omega_carrier*np.arange(self.i, self.i+y.size))
                y = (carrier.conj() * y).astype(np.complex128)
                y = self.lpf(y) * carrier # ehhh... we're time-warping the filter input
                y = scipy.signal.fftconvolve(y,self.base_chirp.conj()[::-1],'valid')
                if self.phase_locked is None:
                    self.phase_locked = (np.abs(y).argmax()-int(self.spacing*.1)) % self.spacing
                    y = y[self.phase_locked:]
                    actual_advance += self.phase_locked
                y = y[:(n+1)*advance].reshape(n+1,2,self.spacing).transpose(1,0,2)
                y = np.log10(np.abs(np.diff(y, axis=1))).transpose(0,2,1)[:,::-1]
                cols = self.columns
                phase = self.window_phase
                if phase+cols+n <= 2*cols:
                    self.buffer[:,:,phase+cols:phase+cols+n] = y
                else:
                    m = phase+n-cols
                    self.buffer[:,:,m-n:] = y[:,:,:n-m]
                    self.buffer[:,:,0:m] = y[:,:,-m:]
                self.buffer[:,:,phase:phase+n] = y
                phase = (phase + n) % cols
                self.window_phase = phase
                b = self.buffer[:,:,phase:phase+cols]
                t = self.i / self.Fs
                self.extent = (t, t+self.display_duration) + self.extent[2:]
                self.im1.set_extent(self.extent)
                self.im2.set_extent(self.extent)
                self.ax1.set_xlim(t, t+self.display_duration)
                self.ax2.set_xlim(t, t+self.display_duration)
                self.im1.set_data(b[0])
                self.im2.set_data(b[1])
                self.draw()
            except Exception, e:
                import traceback
                traceback.print_exc()
        self.input_fragment = stream[actual_advance:]
        self.i += actual_advance

def sonar(recv_only=False):
    Fs = 96e3
    pulse_low = 20e3
    pulse_high = 30e3
    pulse_duration = int(.1 * Fs)
    spacing = int(.02 * Fs)
    reps = 30
    simultaneous_pulses = (pulse_duration+2*spacing-1) // (2*spacing)
    base_chirp = chirp(pulse_low/Fs, pulse_high/Fs, pulse_duration)
    lpf = iir.lowpass((pulse_high-pulse_low)/Fs/2, continuous=True, dtype=np.complex128)
    cs = 1116.44 # feet/sec
    extent = (0, reps*spacing*2/Fs, -spacing*.1/Fs*cs/2, spacing*.9/Fs*cs/2)
    if recv_only:
        y = audio.record(spacing*2*reps, Fs)
    else:
        pulse = base_chirp.real
        pulse *= np.clip(np.arange(pulse_duration, dtype=float)/(Fs*.025), 0, 1)
        pulse *= np.clip(-np.arange(-pulse_duration, 0, dtype=float)/(Fs*.025), 0, 1)
        mono_timing = np.tile(np.r_[1, np.zeros(spacing*2-1)], 3*simultaneous_pulses)
        mono = scipy.signal.fftconvolve(mono_timing, pulse, 'full')[(simultaneous_pulses-1)*spacing*2:simultaneous_pulses*spacing*2]
        output = np.vstack((mono, np.roll(mono, spacing))).T
        y = audio.play_and_record(np.tile(output, (reps, 1)), Fs)[1]
    y = y.astype(np.complex128)
    carrier = np.exp(1j*2*np.pi*(pulse_low+pulse_high)*.5/Fs*np.arange(y.size))
    y = lpf(carrier.conj()*y) * carrier
    y = scipy.signal.fftconvolve(y,base_chirp.conj()[::-1],'full')
    y = y[(np.abs(y).argmax()-spacing*.1) % spacing:]
    y = y[:y.size//spacing*spacing].reshape(-1,spacing)
    y = y[:y.shape[0]//2*2]
    y = y.reshape(y.shape[0]/2,2,-1).transpose(1,0,2)
    y = np.log10(np.abs(np.diff(y, axis=1))).transpose(0,2,1)[:,::-1]
    pl.clf()
    pl.subplot(121)
    pl.imshow(y[0], extent=extent, aspect='auto', interpolation='nearest')
    pl.grid()
    pl.subplot(122)
    pl.imshow(y[1], extent=extent, aspect='auto', interpolation='nearest')
    pl.grid()

audio.play(SonarTransciever(), Fs)

while True:
    sonar(recv_only=True)
    pl.draw()

def align(x):
    offset = np.abs(np.fft.ifft(np.fft.fft(x[:,1:], axis=0) * np.fft.fft(x[:,0:1], axis=0).conj(), axis=0)).argmax(0)
    y = np.zeros_like(x)
    y[:,0] = x[:,0]
    for i in xrange(1,x.shape[1]):
        y[:-offset[i-1],i] = x[offset[i-1]:,i]
    return y

def spline(weights):
    order = len(weights)-1
    import sympy, sympy.abc
    x,y = sympy.abc.x, sympy.abc.y
    polys = [np.poly1d(np.array(p.as_poly(x).all_coeffs(), dtype=float)) for p in (((x + y)**order).expand().subs(y, 1-x)).as_ordered_terms()][::-1]
    return sum(p*c for p,c in zip(polys, weights))

def sweep(start=0, end=48e3, duration=10, volume=1., rampLength=10e-3, outDevice=None, inDevices=None):
    t = np.arange(Fs*duration, dtype=float)/Fs
    f = (end-start)*t/duration + start
    if callable(volume):
        volume = volume(f)
    ramp = np.clip(t/rampLength, 0, 1) * np.clip((duration-t)/rampLength, 0, 1)
    waveform = volume * np.exp(1j*2*np.pi*f.cumsum()/Fs).real * ramp
    results = audio.play_and_record(waveform, Fs, outDevice, inDevices)
    results = np.vstack(results[1:])[:,:waveform.size]
    return np.vstack((waveform, results)).T

def plotSweep(results):
    pl.clf()
    pl.subplot(311)
    pl.specgram(results[:,0], Fs=Fs, NFFT=1024, noverlap=768)
    pl.xlim(0,results.shape[0]/Fs)
    pl.ylim(0,48e3)
    pl.subplot(312)
    pl.specgram(results[:,1], Fs=Fs, NFFT=1024, noverlap=768)
    pl.xlim(0,results.shape[0]/Fs)
    pl.ylim(0,48e3)
    pl.subplot(313)
    pl.specgram(results[:,2], Fs=Fs, NFFT=1024, noverlap=768)
    pl.xlim(0,results.shape[0]/Fs)
    pl.ylim(0,48e3)
    if 0:
        lpf = iir.lowpass(500./Fs)
        _=pl.plot(f[:-5000:50], 20*np.log10(np.abs(lpf(lpf(align(x)[:-5000,1] * np.exp(1j*2*np.pi*f.cumsum()[:-5000]/Fs).conj())[::50] ))))
        _=pl.plot(f[:-5000:50], np.unwrap(np.angle(lpf(lpf(align(x)[:-5000,1] * np.exp(1j*2*np.pi*f.cumsum()[:-5000]/Fs).conj())[::50] ))))

results = sweep(outDevice=output, inDevices=(internalMic, externalMic), volume=(lambda s:(lambda f:s(f/Fs*2)))(spline([.1,.1,.5,.5,.5])))
plotSweep(results)

if 0:
    samples = audio.record(Fs*2, Fs, externalMic)
    samples = np.r_[samples, audio.record(Fs*2, Fs, internalMic)]
    pl.clf()
    _=pl.specgram(samples, Fs=Fs, NFFT=1024, noverlap=768)
    pl.ylim(0,48e3)
    pl.xlim(0,4)
    pl.colorbar()
