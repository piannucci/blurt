import os
import collections
import threading
import select
import itertools
from .graph import Block
from .timers import TimerQueue

class PipeQueue:
    def __init__(self):
        rd, wr = os.pipe()
        self.pipe = os.fdopen(rd, 'rb', buffering=0), os.fdopen(wr, 'wb', buffering=0)
        self.q = collections.deque()

    def put(self, item):
        self.q.append(item)
        self.pipe[1].write(bytes(1))

    def get(self):
        self.pipe[0].read(1)
        return self.q.popleft()

    def fileno(self):
        return self.pipe[0].fileno()

    def __bool__(self):
        return bool(self.q)

class SelectorBlock(Block):
    conditionEnum = itertools.count()
    QuitCondition = next(conditionEnum)
    UpdatedTimerCondition = next(conditionEnum)

    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.conditionQueue = PipeQueue()
        self.stopping = False
        self.running = False
        self.rlist = [self.conditionQueue]
        self.wlist = []
        self.xlist = []
        self.tq = TimerQueue()

    def thread_proc(self):
        while True:
            with self.lock:
                rlist   = tuple(self.rlist)
                wlist   = tuple(self.wlist)
                xlist   = tuple(self.xlist)
                timeout = self.tq.timeUntilNext()
            rlist, wlist, xlist = select.select(rlist, wlist, xlist, timeout)
            with self.lock:
                for fd in rlist:
                    self.readReady(fd)
                for fd in wlist:
                    self.writeReady(fd)
                for fd in xlist:
                    self.exceptional(fd)
                while self.conditionQueue:
                    if self.condition(self.conditionQueue.get()):
                        return

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
            self.conditionQueue.put(self.QuitCondition)
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

    def condition(self, cond):
        if cond == self.QuitCondition:
            self.closeOutput()
            self.closeInput()
            return True
        elif cond == self.UpdatedTimerCondition:
            pass
        return False

    def insertTimer(self, cb, delay=None, when=None):
        with self.lock:
            timer = self.tq.insert(cb, delay, when)
            self.conditionQueue.put(self.UpdatedTimerCondition)
            return timer

    def removeTimer(self, timer):
        with self.lock:
            self.tq.remove(timer)
            self.conditionQueue.put(self.UpdatedTimerCondition)
