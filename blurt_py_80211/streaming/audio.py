import sys
import numpy as np
from ctypes import *
from AudioHardware import *

pythonapi.PyMemoryView_FromMemory.argtypes = (c_void_p, c_ssize_t, c_int)
pythonapi.PyMemoryView_FromMemory.restype = py_object

def findMicrophone():
    for dev in AudioSystemObject[kAudioHardwarePropertyDevices,kAudioObjectPropertyScopeInput]:
        if dev[kAudioDevicePropertyTransportType].value != kAudioDeviceTransportTypeBuiltIn.value:
            continue
        for stream in dev[kAudioDevicePropertyStreams,kAudioObjectPropertyScopeInput]:
            if stream[kAudioStreamPropertyDirection].value == 1 and \
               stream[kAudioStreamPropertyTerminalType].value == 513: # INPUT_MICROPHONE from IOAudioTypes.h
                break
        else:
            continue
        return dev

AudioBufferPointer = POINTER(AudioBuffer)

def arrayFromBuffer(b: AudioBuffer, asbd: AudioStreamBasicDescription):
    if asbd.mFormatFlags & kAudioFormatFlagIsFloat.value:
        fmt = 'f%d'
    elif asbd.mFormatFlags & kAudioFormatFlagIsSignedInteger.value:
        fmt = 'i%d'
    else:
        fmt = 'u%d'
    bytesPerChannel = asbd.mBitsPerChannel // 8
    dtype = np.dtype(fmt % (bytesPerChannel,))
    bytesPerFrame = b.mNumberChannels * bytesPerChannel
    return np.ndarray(
            (b.mDataByteSize // bytesPerFrame, b.mNumberChannels),
            dtype,
            pythonapi.PyMemoryView_FromMemory(b.mData, b.mDataByteSize, 0x100),
            0,
            (asbd.mBytesPerFrame, bytesPerChannel))

class IOStream:
    def __init__(self, dev, scope):
        self.ioProc = AudioDeviceIOProc(self.ioProc)
        self.ioProcID = AudioDeviceIOProcID()
        trap(AudioDeviceCreateIOProcID, dev.objectID, self.ioProc, None, byref(self.ioProcID))
        self.dev = dev
        self.scope = scope
        self.initializeStream()
    def initializeStream(self, scope):
        apf = self.dev[kAudioStreamPropertyAvailablePhysicalFormats,self.scope,0]
        avf = self.dev[kAudioStreamPropertyAvailableVirtualFormats,self.scope,0]
        self.pfDesired = max(apf, key=self.streamFormatKey).mFormat
        self.vfDesired = max(avf, key=self.streamFormatKey).mFormat
        self.pfDesired.mReserved = self.vfDesired.mReserved = 0
        assert self.vfDesired.mFormatID == kAudioFormatLinearPCM.value
        assert self.vfDesired.mFormatFlags == kAudioFormatFlagIsFloat.value | kAudioFormatFlagIsPacked.value
        self.dev[kAudioStreamPropertyPhysicalFormat,self.scope] = self.pfDesired
        self.dev[kAudioStreamPropertyVirtualFormat,self.scope] = self.vfDesired
        self.vfActual = self.dev[kAudioStreamPropertyVirtualFormat,self.scope]
        self.notifiers = [self.dev.notify(k, self.propListener) for k in [
            (kAudioStreamPropertyVirtualFormat,self.scope),
            (kAudioDeviceProcessorOverload,self.scope),
        ]]
    def start(self):
        self.running = True
        self.dev.start(self.ioProcID)
    def stop(self):
        if self.running:
            self.dev.stop(self.ioProcID)
            self.running = False
    def propListener(
            self,
            objectID: AudioObjectID,
            inNumberAddresses: UInt32,
            inAddresses: POINTER(AudioObjectPropertyAddress),
            inClientData: c_void_p):
        obj = AudioObject(objectID)
        if obj == self.dev:
            for i in range(inNumberAddresses):
                if inAddresses[i].mScope == self.scope.value:
                    if inAddresses[i].mSelector == kAudioStreamPropertyVirtualFormat.value:
                        self.vfActual = obj[kAudioStreamPropertyVirtualFormat,self.scope]
                    elif inAddresses[i].mSelector == kAudioDeviceProcessorOverload.value:
                        print('Overrun', file=sys.stderr)
        return 0
    def ioProc(
            self,
            inDevice: AudioObjectID,
            inNow: POINTER(AudioTimeStamp),
            inInputData: POINTER(AudioBufferList),
            inInputTime: POINTER(AudioTimeStamp),
            outOutputData: POINTER(AudioBufferList),
            inOutputTime: POINTER(AudioTimeStamp),
            inClientData: c_void_p):
        pass

class InputStream(IOStream):
    def __init__(self, dev):
        super().__init__(dev, kAudioObjectPropertyScopeInput)
    def ioProc(self, inDevice, inNow, inInputData, inInputTime, outOutputData, inOutputTime, inClientData):
        buffers = cast(inInputData.contents.mBuffers, AudioBufferPointer)
        channelIndex = 1
        for i in range(inInputData.contents.mNumberBuffers):
            self.consume(arrayFromBuffer(buffers[i], self.vfActual).copy(), channelIndex)
            channelIndex += buffers[i].mNumberChannels
        return 0
    def streamFormatKey(self, f):
        if f.mSampleRateRange.mMinimum <= 96000 <= f.mSampleRateRange.mMaximum and
           f.mFormat.mChannelsPerFrame >= 3 and f.mFormat.mBitsPerChannel >= 24 and
           f.mFormat.mFormatID == kAudioFormatLinearPCM.value:
            return (f.mFormat.mChannelsPerFrame, f.mFormat.mBitsPerChannel)
        return -1,

class OneShotInputStream(InputStream):
    def __init__(self, dev, length):
        self.nextBuffer = 0
        self.buffers = [None for i in range(length)]
        self.length = length
        super().__init__(dev)
    def consume(self, frames, channelIndex):
        self.buffers[self.nextBuffer] = frames
        self.nextBuffer = (self.nextBuffer + 1) % len(self.buffers)
        if self.nextBuffer == 0:
            self.stop()
    def capture(self):
        self.start()
        import time
        while self.running:
            time.sleep(.1)
        return np.concatenate(self.buffers, 0).reshape(-1, 4)[:,:3]

dev = findMicrophone()
if dev:
    inputStream = OneShotInputStream(dev, 1024)
    x = inputStream.capture()[:,:3]

import pylab as pl
pl.close()
fig, ax = pl.subplots(3,1,True,True)
for i in range(3):
    pl.sca(ax[i])
    pl.specgram(x[:,i], Fs=inputStream.vfActual.mSampleRate, NFFT=2048, noverlap=2048-256)
