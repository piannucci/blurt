import os
import collections
import threading
import select
import typing
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

class Event(typing.NamedTuple):
    coalesce : bool = False

class Selector:
    QuitEvent = Event()
    UpdatedTimerEvent = Event(coalesce=True)

    def __init__(self):
        super().__init__()
        self.lock = threading.RLock()
        self.eventQueue = PipeQueue()
        self.stopping = False
        self.running = False
        self.rlist = {self.eventQueue : self._eventHandler}
        self.wlist = {}
        self.xlist = {}
        self.tq = TimerQueue()
        self.startup_handlers = []
        self.shutdown_handlers = []
        self.event_handlers = {
            self.QuitEvent: {self._quitHandler},
        }

    def _quitHandler(self):
        return True

    def _eventHandler(self):
        while self.eventQueue:
            uncoalesced = []
            coalesced = collections.OrderedDict()
            while self.eventQueue:
                c = self.eventQueue.get()
                if c.coalesce:
                    coalesced[c] = None
                else:
                    uncoalesced.append(c)
            for c in list(coalesced) + uncoalesced:
                for cb in self.event_handlers.get(c, ()):
                    if cb():
                        self.threadStopping = True

    def thread_proc(self):
        self.threadStopping = False
        with self.lock:
            for cb in self.startup_handlers:
                cb()
        while not self.threadStopping:
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
            self.eventQueue.clear()
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
            self.eventQueue.put(self.QuitEvent)
        self.thread.join()
        with self.lock:
            assert self.running and self.stopping
            self.running = False
            self.stopping = False

    def addTimer(self, cb, delay=None, when=None):
        with self.lock:
            timer = self.tq.insert(cb, delay, when)
            self.eventQueue.put(self.UpdatedTimerEvent)
            return timer

    def removeTimer(self, timer):
        with self.lock:
            self.tq.remove(timer)
            self.eventQueue.put(self.UpdatedTimerEvent)

    def postCondition(self, cond):
        with self.lock:
            self.eventQueue.put(cond)
