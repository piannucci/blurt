import os
import collections
import threading
import select
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

    def clear(self):
        while bool(self.q):
            self.get()

    def fileno(self):
        return self.pipe[0].fileno()

    def __bool__(self):
        return bool(self.q)

class Selector:
    QuitCondition = object()
    UpdatedTimerCondition = object()

    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.conditionQueue = PipeQueue()
        self.stopping = False
        self.running = False
        self.rlist = {self.conditionQueue : self._conditionHandler}
        self.wlist = {}
        self.xlist = {}
        self.tq = TimerQueue()
        self.startup_handlers = []
        self.shutdown_handlers = []
        self.condition_handlers = {
            self.QuitCondition: self._quitHandler,
        }

    def _quitHandler(self):
        self.closeOutput()
        self.closeInput()
        return True

    def _conditionHandler(self):
        while self.conditionQueue:
            if self.condition_handlers.get(self.conditionQueue.get(), lambda : False)():
                return

    def thread_proc(self):
        with self.lock:
            for cb in self.startup_handlers:
                cb()
        while True:
            with self.lock:
                timeout = 0
                while timeout == 0:
                    self.tq.wake()
                    rlist, wlist, xlist = self.rlist, self.wlist, self.xlist
                    timeout = self.tq.timeUntilNext()
            rlist_ready, wlist_ready, xlist_ready = \
                select.select(tuple(rlist), tuple(wlist), tuple(xlist), timeout)
            with self.lock:
                for fd in rlist_ready:
                    rlist[fd]()
                for fd in wlist_ready:
                    wlist[fd]()
                for fd in xlist_ready:
                    xlist[fd]()
        with self.lock:
            self.conditionQueue.clear()
            for cb in self.shutdown_handlers:
                cb()

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

    def addTimer(self, cb, delay=None, when=None):
        with self.lock:
            timer = self.tq.insert(cb, delay, when)
            self.conditionQueue.put(self.UpdatedTimerCondition)
            return timer

    def removeTimer(self, timer):
        with self.lock:
            self.tq.remove(timer)
            self.conditionQueue.put(self.UpdatedTimerCondition)

    def postCondition(self, cond):
        with self.lock:
            self.conditionQueue.put(cond)
