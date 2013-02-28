import time, numpy
import thread
import _coreaudio
import time
import Queue

mainThreadQueue = Queue.Queue()
outBufSize = _coreaudio.getOutBufSize()
inBufSize = _coreaudio.getInBufSize()
sleepDuration = .1

class AudioInterface(object):
    def __init__(self, device=None):
        self.playbackOffset = 0
        self.recordingOffset = 0
        self.recordingBuffer = None
        self.playbackBuffer = None
        self.shouldStop = False
        self.device = None
    def playbackCallback(self, buffer):
        if hasattr(self, 'recordingStarted') and not self.recordingStarted:
            buffer[:] = 0
        else:
            count = buffer.shape[0]
            y = self.playbackBuffer[self.playbackOffset:self.playbackOffset+count]
            if len(y.shape) == 1:
                y = y[:,numpy.newaxis]
            buffer[:y.shape[0]] = y
            buffer[y.shape[0]:] = 0
            self.playbackOffset += count
            if self.playbackOffset >= self.playbackBuffer.shape[0] + outBufSize*10:
                return True
        if self.shouldStop:
            return True
        return False
    def recordingCallback(self, data):
        if hasattr(self, 'playbackStarted') and not self.playbackStarted:
            pass
        else:
            self.recordingBuffer.append(data.mean(1).copy())
            count = data.shape[0]
            self.recordingOffset += count
            if self.recordingOffset >= self.recordingLength:
                return True
        if self.shouldStop:
            return True
        return False
    def play(self, buffer, Fs):
        self.playbackBuffer = buffer
        _coreaudio.startPlayback(self, Fs, self.device)
    def record(self, count_or_stream, Fs):
        if hasattr(count_or_stream, 'append'):
            self.recordingLength = len(count_or_stream)
            self.recordingBuffer = count_or_stream
        else:
            self.recordingLength = count_or_stream
            self.recordingBuffer = []
        _coreaudio.startRecording(self, Fs, self.device)
    def isPlaying(self):
        return hasattr(self, 'playbackDeviceID')
    def isRecording(self):
        return hasattr(self, 'recordingDeviceID')
    def idle(self):
        try:
            f = mainThreadQueue.get_nowait()
        except:
            pass
        else:
            f()
    def wait(self):
        try:
            while self.isPlaying() or self.isRecording():
                time.sleep(sleepDuration)
                self.idle()
        except KeyboardInterrupt:
            self.shouldStop = True
            while self.isPlaying() or self.isRecording():
                time.sleep(sleepDuration)
                self.idle()
        if hasattr(self, 'playbackException'):
            raise self.playbackException
        elif hasattr(self, 'recordingException'):
            raise self.recordingException
        if hasattr(self.recordingBuffer, 'stop'):
            self.recordingBuffer.stop()
        if isinstance(self.recordingBuffer, (tuple, list)):
            return numpy.hstack(self.recordingBuffer)
        return None
    def stop(self):
        self.shouldStop = True
        self.wait()

## Have to initialize the threading mechanisms in order for PyGIL_Ensure to work
thread.start_new_thread(lambda: None, ())

def play(buffer, Fs, device=None):
    ap = AudioInterface(device)
    try:
        ap.play(buffer, Fs)
    except KeyboardInterrupt:
        ap.stop()
    return ap.wait()

def record(count_or_stream, Fs, device=None):
    ap = AudioInterface(device)
    try:
        ap.record(count_or_stream, Fs)
    except KeyboardInterrupt:
        ap.stop()
    return ap.wait()

def play_and_record(buffer, Fs, device=None):
    ap = AudioInterface(device)
    try:
        ap.play(buffer, Fs)
        ap.record(buffer.shape[0], Fs)
    except KeyboardInterrupt:
        ap.stop()
    return ap.wait()

def add_to_main_thread_queue(fn):
    mainThreadQueue.put_nowait(fn)
