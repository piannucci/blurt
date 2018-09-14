import os
import threading
import select
from .graph import Block

def pipe():
    rd, wr = os.pipe()
    return os.fdopen(rd, 'rb', buffering=0), os.fdopen(wr, 'wb', buffering=0)

class SelectorBlock(Block):
    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.pipe = pipe()
        self.stopping = False
        self.running = False
        self.rlist = [self.pipe[0]]
        self.wlist = []
        self.xlist = []
        self.timeout = None

    def thread_proc(self):
        while True:
            with self.lock:
                rlist   = tuple(self.rlist)
                wlist   = tuple(self.wlist)
                xlist   = tuple(self.xlist)
                timeout = self.timeout
            rlist, wlist, xlist = select.select(rlist, wlist, xlist, timeout)
            with self.lock:
                if self.pipe[0] in rlist:
                    self.pipe[0].read(1)
                    self.closeOutput()
                    self.closeInput()
                    return
                for fd in rlist:
                    self.readReady(fd)
                for fd in wlist:
                    self.writeReady(fd)
                for fd in xlist:
                    self.exceptional(fd)

    def start(self):
        with self.lock:
            if self.running or self.stopping:
                return
            self.thread = threading.Thread(target=self.thread_proc)
            self.running = True
            self.thread.start()

    def stop(self):
        with self.lock:
            if self.stopping or not self.running:
                return
            self.stopping = True
            self.pipe[1].write(bytes(1))
        self.thread.join()
        with self.lock:
            assert self.running and self.stopping
            self.running = False
            self.stopping = False

    def readReady(self, fd):
        pass

    def writeReady(self, fd):
        pass

    def exceptional(self, fd):
        pass
