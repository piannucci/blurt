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
        self.pipe = (os.fdopen(rd, 'rb'), os.fdopen(wr, 'wb'))
    def fileno(self):
        return self.pipe[0].fileno()
    def notify(self):
        self.pipe[1].write(b'\0')
        self.pipe[1].flush()
    def wait(self):
        self.pipe[0].read(1)

class TunnelSource(Block):
    inputs = []
    outputs = [Output(Packet, ())]

    def __init__(self, utun, ll_da):
        super().__init__()
        self.utun = utun
        self.cv = PollableCondition()
        self.stopping = False
        self.thread = threading.Thread(target=self.thread_proc)
        self.ll_da = ll_da

    def start(self):
        self.thread.start()

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

    def stop(self):
        self.stopping = True
        self.cv.notify()

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
