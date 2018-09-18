import numpy as np
from .stream import IOStream
from . import AudioHardware as AH
from . import mach_time

class AGCInStreamAdapter(IOStream):
    def __init__(self, target, stream=None):
        self.target = target # an AudioVolumeControl
        self.curLevel = target[AH.kAudioLevelControlPropertyDecibelValue].value
        self.minLevel = target[AH.kAudioLevelControlPropertyDecibelRange].mMinimum
        self.maxLevel = target[AH.kAudioLevelControlPropertyDecibelRange].mMaximum
        self.historyTime = []
        self.historyLevel = [self.curLevel]
        self.clipSignal = 0
        self.vu = 1e-10
        self.stream = stream

    def write(self, frames, inputTime, now):
        peak = np.abs(frames).max()
        self.vu = peak**2
        oldValue = self.curLevel
        if peak > .5:
            self.curLevel = np.clip(
                    self.target[AH.kAudioLevelControlPropertyDecibelValue].value - 1,
                    self.minLevel, self.maxLevel)
        elif peak < .25:
            self.curLevel = np.clip(
                    self.target[AH.kAudioLevelControlPropertyDecibelValue].value + 1,
                    self.minLevel, self.maxLevel)
        if self.curLevel != oldValue:
            self.target[AH.kAudioLevelControlPropertyDecibelValue] = self.curLevel
            self.historyTime.append(now)
            self.historyLevel.append(self.curLevel)
            expirationTime = now - 0.1
            expirationIndex = np.searchsorted(self.historyTime, expirationTime)
            self.historyTime  = self.historyTime[expirationIndex:]
            self.historyLevel = self.historyLevel[expirationIndex:]
        lag = 0
        factor = 10**(-.05 * self.historyLevel[np.searchsorted(self.historyTime, now-lag)])
        frames *= factor
        return self.stream.write(frames, inputTime, now)

    def inDone(self):
        return self.stream.inDone()

class CSMAOutStreamAdapter(IOStream):
    def __init__(self, agc, threshold_dB, nChannelsPerFrame, stream=None):
        self.agc = agc
        self.threshold_dB = threshold_dB
        self.nChannelsPerFrame = nChannelsPerFrame
        self.stream = stream

    def read(self, nFrames, outputTime, now):
        if self.agc.vu < 10**(.1*self.threshold_dB):
            return self.stream.read(nFrames, outputTime, now)
        return np.zeros((nFrames, self.nChannelsPerFrame), np.float32)

    def outDone(self):
        return self.stream.outDone()

def findMicrophone():
    for dev in AH.AudioSystemObject[AH.kAudioHardwarePropertyDevices,AH.kAudioObjectPropertyScopeInput]:
        if dev[AH.kAudioDevicePropertyTransportType].value != AH.kAudioDeviceTransportTypeBuiltIn.value:
            continue
        for stream in dev[AH.kAudioDevicePropertyStreams,AH.kAudioObjectPropertyScopeInput]:
            if stream[AH.kAudioStreamPropertyDirection].value == 1 and \
               stream[AH.kAudioStreamPropertyTerminalType].value == 513: # INPUT_MICROPHONE from IOAudioTypes.h
                break
        else:
            continue
        return dev
    raise Exception('Microphone not found')

def findInputLevelControl(device):
    for c in device[AH.kAudioObjectPropertyControlList]:
        if c[AH.kAudioControlPropertyScope].value != AH.kAudioObjectPropertyScopeInput.value:
            continue
        if c.classID != AH.kAudioVolumeControlClassID.value:
            continue
        return c

class MicrophoneAGCAdapter(AGCInStreamAdapter):
    def __init__(self, stream=None):
        super().__init__(findInputLevelControl(findMicrophone()), stream=stream)
