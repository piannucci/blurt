import queue
import threading
from typing import Type, NamedTuple
from .selector import Selector, Event
from .typing import Subtype, All, UnsatisfiableError

class OverrunWarning(UserWarning):
    pass

class UnderrunWarning(UserWarning):
    pass

class Port(NamedTuple):
    itemtype : Type
    optional : bool = False

class Connection(NamedTuple):
    op : int
    other : Block
    ip : int

class Block:
    inputs = []                 # list of tuples (shape,)
    outputs = []                # list of tuples (dtype, shape)

    def __init__(self):
        self.connections = []   # list of tuples (my output port #, other, other input port #)

    def process(self):
        pass                    # get from self.input_queues, put to self.output_queues

    def propagateClosure(self): # any queue stopped -> stop all queues and return True
        if any(iq.closed for iq in self.input_queues) or \
           any(oq.closed for oq in self.output_queues):
            self.closeOutput()
            self.closeInput()
            return True
        return False

    def closeOutput(self):
        for oq in self.output_queues:
            oq.closed = True

    def closeInput(self):
        for iq in self.input_queues:
            iq.closed = True

    def connect(self, op, other, ip):
        self.connections.append(Connection(op, other, ip))

    def start(self):            # take any special actions for graph start
        self.notify = self.graph.notify
        self.runloop = self.graph.runloop

    def stop(self):             # take any special actions for graph stop
        self.notify = None
        self.runloop = None

    def iterinput(self):
        while not any(iq.empty() for iq in self.input_queues):
            yield tuple(iq.get_nowait() for iq in self.input_queues)

    def output(self, items):
        if not any(iq.full() for iq in self.output_queues):
            for oq, it in zip(self.output_queues, items):
                oq.put_nowait(it)
            return True
        else:
            warnings.warn('%s overrun' % self.__class__.__name__, OverrunWarning)
            return False

class Graph:
    def __init__(self, sourceBlocks, *, runloop=None):
        # check that source blocks have no inputs
        for b in sourceBlocks:
            if b.inputs:
                raise ConnectionError(b, 'source cannot have inputs')

        # get full list of blocks
        finished = set()
        frontier = set(sourceBlocks)
        rank = {b:0 for b in sourceBlocks}
        parents = {b:() for b in sourceBlocks}
        parent_from_ip = {} # (parent, output port) from (block, input port)
        child_from_op = {}  # (child, input port) from (block, output port)
        conditions = []

        while frontier:
            b = frontier.pop()

            # b already has a rank and a parent list; make parent list immutable and unique
            parents[b] = tuple(set(parents[b]))

            # check output completeness
            connected_ops = {op for op, other, ip in b.connections}
            for op, o in enumerate(b.outputs):
                if not o.optional and op not in connected_ops:
                    raise ConnectionError(b, 'missing required output connection %d', op)

            # process downstream connections
            for op, bb, ip in b.connections:
                if bb not in rank:
                    rank[bb] = float('-inf')
                    parents[bb] = []
                rank[bb] = max(rank[bb], rank[b]+1)
                parents[bb].append(b)
                parent_from_ip[bb, ip] = (b, op)
                child_from_op[b, op] = (bb, ip)
                print('Connection', b, op, bb, ip)

                condition = Subtype(b.outputs[op].itemtype, bb.inputs[ip].itemtype)
                conditions.append(Condition.bound_to(condition, b, bb))

                if all(i.optional or (bb,ip) in parent_from_ip for ip,i in enumerate(bb.inputs)):
                    frontier.add(bb)
            finished.add(b)

        for b in set(rank.keys()) - finished:
            ips = sorted(ip for op, other, ip in b.connections)
            raise ConnectionError(b, 'missing required input connection (or cycle)', ips)

        try:
            Condition.sat(All(conditions)).apply()
        except UnsatisfiableError:
            raise ConnectionError('Cannot satisfy type constraints', conditions)

        self.allBlocks = tuple(sorted(finished, key=rank.get))
        self.runningBlocks = set(self.allBlocks)
        self.danglingQueues = []
        for b in self.allBlocks:
            b.graph = self
            b.input_queues = []
            for ip,i in enumerate(b.inputs):
                if (b, ip) in parent_from_ip:
                    bb, op = parent_from_ip[b, ip]
                    iq = bb.output_queues[op]
                elif i.optional:
                    iq = queue.Queue()
                else:
                    assert False, 'Missing required input connection should have been caught earlier'
                b.input_queues.append(iq)
            b.input_queues = tuple(b.input_queues)
            b.output_queues = []
            for op,o in enumerate(b.outputs):
                oq = queue.Queue()
                oq.closed = False
                if (b, op) in child_from_op:
                    pass
                elif o.optional:
                    self.danglingQueues.append(oq)
                else:
                    assert False, 'Missing required output connection should have been caught earlier'
                b.output_queues.append(oq)
            b.output_queues = tuple(b.output_queues)

        # set up the event selector
        if runloop is None:
            runloop = Selector()
        self.runloop = runloop
        self.runloop.startup_handlers.append(self._startupHandler)
        self.runloop.shutdown_handlers.insert(0, self._shutdownHandler)
        self.WakeGraphCondition = Event(coalesce=True)
        self.runloop.condition_handlers[self.WakeGraphCondition] = self._wakeHandler
        self.start = self.runloop.start
        self.stop = self.runloop.stop

    def notify(self):
        self.runloop.postCondition(self.WakeGraphCondition)

    def _startupHandler(self):
        self.notify()
        for b in self.allBlocks:
            b.start()

    def _shutdownHandler(self):
        for b in self.allBlocks:
            b.stop()

    def _wakeHandler(self):
        for b in self.allBlocks:
            if b in self.runningBlocks:
                b.process()
                if b.propagateClosure():
                    self.runningBlocks.remove(b)
        for q in self.danglingQueues:
            while not q.empty():
                q.get_nowait()
        return not bool(self.runningBlocks)
