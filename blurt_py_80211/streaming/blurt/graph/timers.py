import heapq
import time

class PQ:
    _removed = object()

    def __init__(self):
        self.pq = []
        self.entry_finder = {}               # mapping of tasks to entries
        self.counter = itertools.count()

    def insert(self, task, priority=0):
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heapq.heappush(self.pq, entry)

    def remove(self, task):
        self.entry_finder.pop(task)[-1] = PQ._removed

    def _head(self, remove=True):
        while self.pq:
            priority, count, task = self.pq[0]
            if task is not PQ._removed:
                if remove:
                    heapq.heappop(self.pq)
                    del self.entry_finder[task]
                return task
            heapq.heappop(self.pq)
        raise IndexError('priority queue is empty')

    def pop(self):
        return self._head(True)

    def peek(self):
        return self._head(False)

class TimerQueue(PQ):
    def insert(self, cb, delay=None, when=None):
        if when is None:
            assert delay is not None
            when = time.monotonic() + delay
        item = (cb, when)
        super().insert(item, when)
        return item

    def timeUntilNext(self):
        try:
            _, when = self.peek()
        except IndexError:
            return None
        return max(0, when - time.monotonic())

    def wake(self):
        while self.pq:
            try:
                cb, when = self.peek()
            except IndexError:
                break
            if when > time.monotonic(): # XXX do we need some tolerance here?
                break
            cb()
            self.pop()
