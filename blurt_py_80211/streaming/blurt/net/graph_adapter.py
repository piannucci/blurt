import os
import warnings
import queue
import threading
import select
from ..graph import Output, Input, Block, OverrunWarning
from ..mac.lowpan import Packet

class PollableCondition:
    def __init__(self):
        rd, wr = os.pipe()
        self.pipe = (os.fdopen(rd, 'rb', buffering=0), os.fdopen(wr, 'wb', buffering=0))
    def fileno(self):
        return self.pipe[0].fileno()
    def notify(self):
        self.pipe[1].write(b'\0')
    def wait(self):
        self.pipe[0].read(1)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        return False

class TunnelSource(Block):
    inputs = []
    outputs = [Output(Packet, ())]

    def __init__(self, utun, ll_da):
        super().__init__()
        self.utun = utun
        self.cv = PollableCondition()
        self.ll_da = ll_da
        self.stopping = False
        self.running = False

    def thread_proc(self):
        while True:
            for fd in select.select([self.utun, self.cv], [], [])[0]:
                if fd is self.utun:
                    datagram = self.utun.read()
                    print('%s -> %d B' % (self.utun, len(datagram)))
                    packet = Packet(self.utun.ll_addr, self.ll_da, datagram)
                    try:
                        self.output_queues[0].put_nowait(packet)
                        self.notify()
                    except queue.Full:
                        warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)
                elif fd is self.cv:
                    if self.stopping:
                        return

    def start(self):
        with self.cv:
            if self.running:
                return
            self.thread = threading.Thread(target=self.thread_proc)
            self.running = True
            self.thread.start()

    def stop(self):
        with self.cv:
            if self.stopping or not self.running:
                return
            self.stopping = True
            self.cv.notify()
            self.thread.join()
            self.running = False
            self.stopping = False

    def process(self):
        pass

class TunnelSink(Block):
    inputs = [Input(())]
    outputs = []

    def __init__(self, utun):
        super().__init__()
        self.utun = utun

    def process(self):
        while True:
            try:
                packet = self.input_queues[0].get_nowait()
            except queue.Empty:
                break
            datagram = packet.tail()
            print('%5d B -> %s' % (len(datagram), self.utun))
            self.utun.write(datagram)
