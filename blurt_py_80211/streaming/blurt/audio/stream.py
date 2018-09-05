import numpy as np

class IOStream:
    def read(self, nFrames : int, outputTime : int, now : int) -> np.ndarray:
        return np.empty((0, 1), np.float32)
    def write(self, frames : np.ndarray, inputTime : int, now : int) -> None:
        pass
    def outDone(self):
        return False
    def inDone(self):
        return False

class InArrayStream(IOStream):
    def __init__(self, array):
        if array.ndim == 1:
            array = array[:,None]
        self.inArray = array
        self.nFrames, self.nChannelsPerFrame = array.shape
        self.inFrame = 0
    def write(self, frames, inputTime, now):
        nFrames = min(frames.shape[0], self.inArray.shape[0]-self.inFrame)
        if frames.shape[1] != self.nChannelsPerFrame:
            frames = frames.mean(1)[:,None]
        self.inArray[self.inFrame:self.inFrame+nFrames] = frames[:nFrames]
        self.inFrame += nFrames
    def inDone(self):
        return self.inFrame >= self.nFrames

class OutArrayStream(IOStream):
    def __init__(self, array):
        if array.ndim == 1:
            array = array[:,None]
        self.outArray = array
        self.nFrames, self.nChannelsPerFrame = array.shape
        self.outFrame = 0
    def read(self, nFrames, outputTime, now):
        frames = self.outArray[self.outFrame:self.outFrame+nFrames]
        self.outFrame += nFrames
        return frames
    def outDone(self):
        return self.outFrame >= self.nFrames
