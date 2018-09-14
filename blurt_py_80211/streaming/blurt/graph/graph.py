import threading
import queue

class OverrunWarning(UserWarning):
    pass

class UnderrunWarning(UserWarning):
    pass

class Output:
    def __new__(cls, dtype, shape):
        return (dtype, shape)

class Input:
    def __new__(cls, shape):
        return (shape,)

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
        self.connections.append((op, other, ip))

    def notify(self):           # called by sources to wake the graph
        self.graph.notify()

    def start(self):            # take any special actions for graph start
        pass

    def stop(self):             # take any special actions for graph stop
        pass

    def input(self):
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
    def __init__(self, sourceBlocks):
        # check that source blocks have no inputs
        for b in sourceBlocks:
            if b.inputs:
                raise ConnectionError(b, 'source cannot have inputs')
        # get full list of blocks
        allBlocks = set()
        completeBlocks = set(sourceBlocks)
        rank = {b:0 for b in sourceBlocks}
        parents = {b:() for b in sourceBlocks}
        parent_from_ip = {} # parent from (other block, input port)
        op_from_ip = {}     # output port from (other block, input port)
        while completeBlocks:
            b = completeBlocks.pop()
            # b already has a rank and a parent list
            parents[b] = tuple(set(parents[b]))                 # parent list immutable and unique
            ops = sorted(op for op, other, ip in b.connections)
            if ops != list(range(len(b.outputs))):              # outputs fully connected
                raise ConnectionError(b, 'outputs', ops)
            for op, bb, ip in b.connections:
                if bb not in rank:
                    rank[bb] = float('-inf')
                    parents[bb] = []
                oshape = b.outputs[op][1]
                if isinstance(oshape, str):
                    oshape = getattr(b, oshape)
                else:
                    oshape = tuple(getattr(b, ol) if isinstance(ol, str) else ol for ol in oshape)
                print('Connection', b, op, bb, ip)
                ishape = bb.inputs[ip][0]
                if isinstance(ishape, str):
                    setattr(bb, ishape, oshape)
                else:
                    if len(oshape) != len(ishape):
                        raise ConnectionError(b, 'ndim', op, bb, ip, oshape, ishape)
                    for ol, il in zip(oshape, ishape):
                        if isinstance(il, str):
                            setattr(bb, il, ol)
                        elif ol != il and ol != 1:
                            raise ConnectionError(b, 'shape', op, bb, ip, oshape, ishape)
                rank[bb] = max(rank[bb], rank[b]+1)
                parents[bb].append(b)
                parent_from_ip[bb, ip] = b
                op_from_ip[bb, ip] = op
                if len(parents[bb]) == len(bb.inputs):
                    ips = sorted(ip for p in parents[bb] for op, other, ip in p.connections if other is bb)
                    if ips != list(range(len(bb.inputs))):      # inputs fully connected
                        raise ConnectionError(b, 'inputs', ips)
                    completeBlocks.add(bb)
            allBlocks.add(b)
        if len(allBlocks) != len(rank):                         # some block has dangling (or extra) inputs
            b = set(rank.keys()).difference(allBlocks).pop()
            ips = sorted(ip for op, other, ip in b.connections)
            raise ConnectionError(b, 'inputs (possible cycle)', ips)
        self.allBlocks = tuple(sorted(allBlocks, key=rank.get))
        for b in self.allBlocks:
            b.output_queues = tuple(queue.Queue() for op in range(len(b.outputs)))
            b.input_queues = tuple(parent_from_ip[b, ip].output_queues[op_from_ip[b, ip]] for ip in range(len(b.inputs)))
            for oq in b.output_queues:
                oq.closed = False
        self.runningBlocks = set(self.allBlocks)
        self.cv = threading.Condition()
        self.notified = True
        for b in sourceBlocks:
            b.graph = self
        self.thread = None

    def start(self):
        with self.cv:
            self.stop()
            self.join()
            self.stopping = False
            self.notified = True
            for b in self.allBlocks:
                b.start()
            self.thread = threading.Thread(target=self.thread_proc, daemon=True)
            self.thread.start()

    def stop(self):
        with self.cv:
            self.stopping = True
            self.cv.notify()

    def join(self):
        with self.cv:
            if self.thread is not None:
                self.thread.join()
            self.thread = None

    def notify(self):
        with self.cv:
            self.notified = True
            self.cv.notify()

    def thread_proc(self):
        while bool(self.runningBlocks):
            for b in tuple(self.runningBlocks):
                b.process()
                if b.propagateClosure():
                    self.runningBlocks.remove(b)
            with self.cv:
                while True:
                    if self.stopping:
                        for b in self.allBlocks:
                            b.stop()
                        return
                    elif self.notified:
                        break
                    self.cv.wait()
                self.notified = False
