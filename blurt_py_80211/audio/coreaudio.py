import time, numpy
import thread
import _coreaudio
import time

class AudioInterface(object):
    def __init__(self, device=None):
        self.playbackOffset = 0
        self.recordingOffset = 0
        self.recordingBuffer = []
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
            if self.playbackOffset >= self.playbackBuffer.shape[0] + 512*10:
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
    def record(self, length, Fs):
        self.recordingLength = length
        _coreaudio.startRecording(self, Fs, self.device)
    def isPlaying(self):
        return hasattr(self, 'playbackDeviceID')
    def isRecording(self):
        return hasattr(self, 'recordingDeviceID')
    def wait(self):
        try:
            while self.isPlaying() or self.isRecording():
                time.sleep(.1)
        except KeyboardInterrupt:
            self.shouldStop = True
            while self.isPlaying() or self.isRecording():
                time.sleep(.1)
        if hasattr(self, 'playbackException'):
            raise self.playbackException
        elif hasattr(self, 'recordingException'):
            raise self.recordingException
        if len(self.recordingBuffer):
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

def record(count, Fs, device=None):
    ap = AudioInterface(device)
    try:
        ap.record(count, Fs)
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
