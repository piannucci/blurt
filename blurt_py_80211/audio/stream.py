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
        obj.dtype = kwargs['dtype'] if 'dtype' in kwargs else np.float32
        obj.init()
        return obj
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
        return (len(self),)

class ThreadedStream(StreamArray):
    def init(self):
        self.in_queue = Queue.Queue(8)
        self.out_queue = Queue.Queue(8)
        if hasattr(self, 'thread_consume'):
            self.in_thread = thread.start_new_thread(self.in_thread_loop, ())
        if hasattr(self, 'thread_produce'):
            self.out_thread = thread.start_new_thread(self.out_thread_loop, ())
        self.out_fragment = None
    def in_thread_loop(self):
        while True:
            work = self.in_queue.get()
            if work is None:
                break
            self.thread_consume(work)
    def out_thread_loop(self):
        while True:
            work = self.thread_produce()
            if work is None:
                break
            self.out_queue.put(work)
    def consume(self, sequence):
        try:
            self.in_queue.put_nowait(sequence)
        except Queue.Full:
            print 'Overrun'
    def produce(self, count):
        result = np.empty(count, self.dtype)
        i = 0
        if self.out_fragment is not None:
            n = min(self.out_fragment.size, count)
            result[:n] = self.out_fragment[:n]
            i += n
            if n < self.out_fragment.size:
                self.out_fragment = self.out_fragment[n:]
            else:
                self.out_fragment = None
        while i < count:
            try:
                fragment = self.out_queue.get_nowait()
            except Queue.Empty:
                result[i:] = 0
                print 'Underrun'
                break
            else:
                n = min(count-i, fragment.size)
                result[i:i+n] = fragment[:n]
                i += n
                if fragment.size > n:
                    self.out_fragment = fragment[n:]
        return result

class WhiteNoise(ThreadedStream):
    def thread_produce(self):
        return np.random.standard_normal(1024) * .2

class VUMeter(ThreadedStream):
    def init(self):
        super(VUMeter, self).init()
        self.peak = 0.
        self.Fs = self.kwargs['Fs'] if 'Fs' in self.kwargs else 96000.
    def thread_consume(self, sequence):
        volume = 30 + int(round(10.*np.log10((np.abs(sequence).astype(float)**2).mean())))
        peak = 30 + int(round(10.*np.log10((np.abs(sequence).astype(float)**2).max())))
        self.peak = max(peak, self.peak-8*sequence.size/self.Fs)
        bar = '#'*volume
        n = int(np.ceil(self.peak))
        bar = bar + ' ' * (n-len(bar)) + '|'
        sys.stdout.write('\r\x1b[K' + bar)
        sys.stdout.flush()

class Oscilloscope(ThreadedStream):
    def init(self):
        super(Oscilloscope, self).init()
        import pylab as pl
        self.pl = pl
        self.fig = pl.figure()
        self.ax = pl.gca()
        self.line = self.ax.plot(np.zeros(ca.inBufSize))[0]
        self.ax.set_ylim(-.5,.5)
        widget = self.fig.canvas.get_tk_widget()
        widget.winfo_toplevel().lift()
        ca.sleepDuration = .01
    def thread_consume(self, sequence):
        self.line.set_ydata(sequence)
        ca.add_to_main_thread_queue(self.pl.draw)
