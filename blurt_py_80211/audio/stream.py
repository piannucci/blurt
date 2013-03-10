import numpy as np
import coreaudio as ca
import thread
import Queue
import sys

class StreamArray(np.ndarray):
    def __new__(cls, *args, **kwargs):
        obj = np.asarray([]).view(cls)
        obj.args = args
        obj.kwargs = kwargs
        obj.dtype = obj.kwarg('dtype', np.float32)
        obj.channels = kwargs['channels'] if 'channels' in kwargs else 1
        obj.init()
        return obj
    def kwarg(self, name, default=None):
        if name in self.kwargs:
            return self.kwargs[name]
        return default
    def __array__(self, dtype=None):
        if dtype is not None:
            return StreamArray(*self.args, **self.kwargs)
        return self
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.args = getattr(obj, 'args', ())
        self.kwargs = getattr(obj, 'kwargs', {})
    def __len__(self):
        return sys.maxint
    def __getslice__(self, i, j):
        return self.produce(j-i).astype(self.dtype)
    def append(self, sequence):
        return self.consume(sequence)
    @property
    def shape(self):
        return (len(self), self.channels)

class ThreadedStream(StreamArray):
    def init(self):
        self.in_queue = Queue.Queue(64)
        self.out_queue = Queue.Queue(64)
        if hasattr(self, 'thread_consume'):
            self.in_thread = thread.start_new_thread(self.in_thread_loop, ())
        if hasattr(self, 'thread_produce'):
            self.out_thread = thread.start_new_thread(self.out_thread_loop, ())
        self.out_fragment = None
        self.numSamplesRead = 0
        self.numSamplesWritten = 0
        self.numSamplesProduced = 0
        self.numSamplesConsumed = 0
    def in_thread_loop(self):
        while True:
            work = self.in_queue.get()
            if work is None:
                break
            self.numSamplesConsumed += work.shape[0]
            self.thread_consume(work)
    def out_thread_loop(self):
        while True:
            work = self.thread_produce()
            if work is None:
                break
            self.numSamplesProduced += work.shape[0]
            self.out_queue.put(work)
    def consume(self, sequence):
        self.numSamplesWritten += sequence.shape[0]
        try:
            self.in_queue.put_nowait(sequence)
        except Queue.Full:
            print 'ThreadedStream overrun'
        except Exception, e:
            print 'ThreadedStream exception %s' % repr(e)
    def produce(self, count):
        result = np.empty((count, self.channels), self.dtype)
        i = 0
        if self.out_fragment is not None:
            n = min(self.out_fragment.shape[0], count)
            result[:n] = self.out_fragment[:n]
            i += n
            if n < self.out_fragment.shape[0]:
                self.out_fragment = self.out_fragment[n:]
            else:
                self.out_fragment = None
        while i < count:
            try:
                fragment = self.out_queue.get_nowait()
                if len(fragment.shape) == 1:
                    fragment = fragment[:,np.newaxis]
                if fragment.shape[1] != self.channels and fragment.shape[1] != 1:
                    raise Exception('ThreadedStream produced a stream with the wrong number of channels.')
            except Queue.Empty:
                result[i:] = 0
                print 'ThreadedStream underrun'
                break
            except Exception, e:
                print 'ThreadedStream exception %s' % repr(e)
            else:
                n = min(count-i, fragment.shape[0])
                result[i:i+n] = fragment[:n]
                i += n
                if fragment.shape[0] > n:
                    self.out_fragment = fragment[n:]
        self.numSamplesRead += result.shape[0]
        return result
    def onMainThread(self, fn):
        ca.add_to_main_thread_queue(fn)
    def stop(self):
        self.in_queue.put(None)

class WhiteNoise(ThreadedStream):
    def thread_produce(self):
        return np.random.standard_normal(1024) * .2

class VUMeter(ThreadedStream):
    def init(self):
        super(VUMeter, self).init()
        self.peak = 0.
        self.Fs = self.kwarg('Fs', 96000.)
    def thread_consume(self, sequence):
        volume = 30 + int(round(10.*np.log10((np.abs(sequence).astype(float)**2).mean())))
        peak = 30 + int(round(10.*np.log10((np.abs(sequence).astype(float)**2).max())))
        self.peak = max(peak, self.peak-8*sequence.size/self.Fs)
        bar = '#'*volume
        n = int(np.ceil(self.peak))
        bar = bar + ' ' * (n-len(bar)) + '|'
        sys.stdout.write('\r\x1b[K' + bar)
        sys.stdout.flush()

class Visualizer(ThreadedStream):
    def init(self):
        super(Visualizer, self).init()
        import pylab as pl
        self.pl = pl
        self.fig = pl.figure()
        self.ax = pl.gca()
        widget = self.fig.canvas.get_tk_widget()
        widget.winfo_toplevel().lift()
        ca.sleepDuration = .01
    def draw(self):
        self.onMainThread(self.pl.draw)

class Oscilloscope(Visualizer):
    def init(self):
        super(Oscilloscope, self).init()
        self.line = self.ax.plot(np.zeros(ca.inBufSize))[0]
        self.ax.set_ylim(-.5,.5)
    def thread_consume(self, sequence):
        self.line.set_ydata(sequence)
        self.draw()

class SpectrumAnalyzer(Visualizer):
    def init(self):
        super(SpectrumAnalyzer, self).init()
        self.Fs = self.kwarg('Fs', 96000.)
        self.Fc = self.kwarg('Fc', 0.)
        self.NFFT = self.kwarg('NFFT', 256)
        self.noverlap = self.kwarg('noverlap', 128)
        self.window = self.kwarg('window', self.pl.mlab.window_hanning)(np.ones(self.NFFT))
        self.duration = self.kwarg('duration', 1.)
        self.transpose = self.kwarg('transpose', False)
        self.sides = self.kwarg('sides', 1)
        self.advance = self.NFFT - self.noverlap
        self.columns = int(round(self.duration * self.Fs / self.advance))
        self.buffer = np.zeros((self.NFFT*self.sides/2, self.columns), np.float32)
        self.t = 0.
        #self.im = self.ax.imshow(self.buffer if not self.transpose else self.buffer.T,
        #                         vmin=0., vmax=1.,
        #                         extent=(self.t, self.t+self.duration, -self.Fs/2, self.Fs/2),
        #                         aspect='auto', interpolation='none')
        self.im = self.fig.figimage(self.buffer if not self.transpose else self.buffer.T,
                                    xo=10, yo=10)
        self.fig.delaxes(self.ax)
        self.input_fragment = np.zeros(0, np.complex128)
    def thread_consume(self, input):
        stream = np.r_[self.input_fragment, input]
        n = max(1 + (stream.size-self.NFFT) // self.advance, 0)
        if n < 4: n = 0
        self.input_fragment = stream[n*self.advance:]
        if n > 0:
            input = stream[:self.NFFT + (n-1)*self.advance]
            idx = np.arange(self.NFFT)[:,np.newaxis] + self.advance * np.arange(n)[np.newaxis,:]
            input = input[idx] * self.window[:,np.newaxis]
            input = np.fft.fftshift(np.fft.fft(input, axis=0), 0)
            input = (np.log10(np.abs(input)) / 5.) + 1
            self.buffer[:,:-n] = self.buffer[:,n:]
            self.buffer[:,-n:] = input[:(2-self.sides)*self.NFFT/2]
            self.im.set_data(self.buffer if not self.transpose else self.buffer.T)
            #self.im.set_extent((self.t, self.t+self.duration, -self.Fs/2, self.Fs/2))
            #self.ax.set_xlim(self.t, self.t+self.duration)
            #self.ax.set_ylim(0, self.Fs/2)
            self.t += n * self.advance / self.Fs
            self.fig.draw_artist(self.im)
            self.onMainThread(self.fig.canvas.blit)

