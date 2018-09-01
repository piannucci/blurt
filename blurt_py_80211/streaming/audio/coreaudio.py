import sys
import numpy as np
import threading
import queue
import uuid
from AudioHardware import *
from mach_time import *

defInBufSize = defOutBufSize = 2048

class IOStream:
    def read(self, nFrames : int, outputTime : int, now : int) -> np.ndarray:
        return np.empty((0, 1), np.float32)
    def write(self, frames : np.ndarray, inputTime : int, now : int) -> None:
        pass
    def inDone(self):
        return False
    def outDone(self):
        return False

class AGCInStreamAdapter(IOStream):
    def __init__(self, stream, target):
        self.stream = stream
        self.target = target # an AudioVolumeControl
        self.curLevel = target[kAudioLevelControlPropertyDecibelValue].value
        self.minLevel = target[kAudioLevelControlPropertyDecibelRange].mMinimum
        self.maxLevel = target[kAudioLevelControlPropertyDecibelRange].mMaximum
        self.historyTime = []
        self.historyLevel = [self.curLevel]
        self.clipSignal = 0
    def write(self, frames, inputTime, now):
        peak = np.abs(frames).max()
        oldValue = self.curLevel
        if peak > .5:
            self.curLevel = np.clip(
                    self.target[kAudioLevelControlPropertyDecibelValue].value - 1,
                    self.minLevel, self.maxLevel)
        elif peak < .25:
            self.curLevel = np.clip(
                    self.target[kAudioLevelControlPropertyDecibelValue].value + 1,
                    self.minLevel, self.maxLevel)
        if self.curLevel != oldValue:
            self.target[kAudioLevelControlPropertyDecibelValue] = self.curLevel
            self.historyTime.append(now)
            self.historyLevel.append(self.curLevel)
            expirationTime = now - 100e6 / nanosecondsPerAbsoluteTick()
            expirationIndex = np.searchsorted(self.historyTime, expirationTime)
            self.historyTime  = self.historyTime[expirationIndex:]
            self.historyLevel = self.historyLevel[expirationIndex:]
        lag = 0
        factor = 10**(-.05 * self.historyLevel[np.searchsorted(self.historyTime, now-lag)])
        frames *= factor
        return self.stream.write(frames, inputTime, now)
    def inDone(self):
        return self.stream.inDone()

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

class ThreadedStream(IOStream):
    def __init__(self, nChannelsPerFrame=1, inThread=False, outThread=False, out_queue_depth=64):
        self.dtype = dtype
        self.nChannelsPerFrame = nChannelsPerFrame
        self.inQueue = queue.Queue(64)
        self.outQueue = queue.Queue(out_queue_depth)
        self.outFragment = None
        self.numSamplesRead = 0
        self.numSamplesWritten = 0
        self.numSamplesProduced = 0
        self.numSamplesConsumed = 0
        self.warnOnUnderrun = True
        if inThread:
            self.inThread = threading.Thread(target=self.in_thread_loop, daemon=True)
            self.inThread.start()
        if outThread:
            self.outThread = threading.Thread(target=self.out_thread_loop, daemon=True)
            self.outThread.start()

    def in_thread_loop(self):
        while True:
            work = self.inQueue.get()
            if work is None:
                break
            self.numSamplesConsumed += work.shape[0]
            self.consume(work)

    def out_thread_loop(self):
        while True:
            work = self.produce()
            if work is None:
                break
            self.numSamplesProduced += work.shape[0]
            self.outQueue.put(work)

    def write(self, frames, inputTime, now):
        self.numSamplesWritten += frames.shape[0]
        try:
            self.inQueue.put_nowait(frames)
        except queue.Full:
            print('ThreadedStream overrun')
        except Exception as e:
            print('ThreadedStream exception %s' % repr(e))

    def read(self, nFrames, outputTime, now):
        result = np.empty((nFrames, self.nChannelsPerFrame), self.dtype)
        i = 0
        if self.outFragment is not None:
            n = min(self.outFragment.shape[0], nFrames)
            result[:n] = self.outFragment[:n]
            i += n
            if n < self.outFragment.shape[0]:
                self.outFragment = self.outFragment[n:]
            else:
                self.outFragment = None
        while i < nFrames:
            try:
                fragment = self.immediate_produce()
                if len(fragment.shape) == 1:
                    fragment = fragment[:,np.newaxis]
                if fragment.shape[1] != self.nChannelsPerFrame and fragment.shape[1] != 1:
                    raise Exception('ThreadedStream produced a stream with the wrong number of channels.')
            except queue.Empty:
                result[i:] = 0
                if self.warnOnUnderrun:
                    print('ThreadedStream underrun')
                break
            except Exception as e:
                print('ThreadedStream exception %s' % repr(e))
            else:
                n = min(nFrames-i, fragment.shape[0])
                result[i:i+n] = fragment[:n]
                i += n
                if fragment.shape[0] > n:
                    self.outFragment = fragment[n:]
        self.numSamplesRead += result.shape[0]
        return result

    def immediate_produce(self):
        return self.outQueue.get_nowait()

    def stop(self):
        self.inQueue.put(None)

class IOSession:
    def __init__(self):
        self.cv = threading.Condition()
        with self.cv:
            self.negotiated = {
                kAudioObjectPropertyScopeInput.value: False,
                kAudioObjectPropertyScopeOutput.value: False,
            }
            self.created = False
            self.stopping = False
            self.running = False
            self.masterDeviceUID = None
            self.deviceScopes = []
            self.vfDesired = {} # indexed by device, scope
            self.vfActual = {} # indexed by device, scope
            self.notifiers = {} # indexed by device, scope
            self.bufSize = {} # indexed by scope
            self.ioProc_ptr = AudioDeviceIOProc(self.ioProc)
            self.nsPerAbsoluteTick = nanosecondsPerAbsoluteTick()

    def __del__(self):
        self.destroyAggregate()

    def addDevice(self, device, scope):
        with self.cv:
            key = (device, scope.value)
            if key in self.deviceScopes:
                return
            self.deviceScopes.append(key)
            self.notifiers[key] = [device.notify(k, self.propListener) for k in [
                (kAudioStreamPropertyVirtualFormat, scope),
                (kAudioDeviceProcessorOverload, scope),
            ]]

    def addDefaultInputDevice(self):
        self.addDevice(AudioSystemObject[kAudioHardwarePropertyDefaultInputDevice], kAudioObjectPropertyScopeInput)

    def addDefaultOutputDevice(self):
        self.addDevice(AudioSystemObject[kAudioHardwarePropertyDefaultOutputDevice], kAudioObjectPropertyScopeOutput)

    def negotiateFormat(self, scope,
            minimumSampleRate=96000,
            maximumSampleRate=96000,
            minimumBitsPerChannel=32,
            maximumBitsPerChannel=32,
            inBufSize=defInBufSize,
            outBufSize=defOutBufSize,
        ):
        scope = getattr(scope, 'value', scope)
        with self.cv:
            if self.negotiated[scope]:
                return
            self.bufSize[scope] = outBufSize if scope == kAudioObjectPropertyScopeOutput else inBufSize
            # Filter formats by sample rate range, bit depth range, format ID
            formatRanges = {}
            for d, s in self.deviceScopes:
                if s != scope:
                    continue
                vf = d[kAudioStreamPropertyAvailableVirtualFormats, scope]
                vf_filtered = []
                for f in vf:
                    if f.mSampleRateRange.mMinimum > maximumSampleRate or minimumSampleRate > f.mSampleRateRange.mMaximum:
                        continue
                    if not minimumBitsPerChannel <= f.mFormat.mBitsPerChannel <= maximumBitsPerChannel:
                        continue
                    if f.mFormat.mFormatID != kAudioFormatLinearPCM.value:
                        continue
                    if f.mFormat.mFormatFlags != kAudioFormatFlagIsFloat.value | kAudioFormatFlagIsPacked.value:
                        continue
                    vf_filtered.append(f)
                if not vf_filtered:
                    raise ValueError('No candidate formats for %s' % d)
                formatRanges[d] = vf_filtered
            # Find the lowest candidate sample rate viable for every device
            for sampleRate in np.sort(
                [f.mSampleRateRange.mMinimum for d, vf in formatRanges.items() for f in vf] +
                [f.mSampleRateRange.mMaximum for d, vf in formatRanges.items() for f in vf] +
                [minimumSampleRate, maximumSampleRate]):
                if not minimumSampleRate <= sampleRate <= maximumSampleRate:
                    continue
                formats = {}
                for d, vf in formatRanges.items():
                    formats[d] = []
                    for f in vf:
                        if f.mSampleRateRange.mMinimum <= sampleRate <= f.mSampleRateRange.mMaximum:
                            asbd = AudioStreamBasicDescription()
                            memmove(pointer(asbd), pointer(f.mFormat), sizeof(asbd))
                            asbd.mSampleRate = sampleRate
                            formats[d].append(asbd)
                    if not formats[d]:
                        break
                else:
                    break # every device has a viable format
                continue # some device has no viable format
            else:
                raise ValueError('No format is viable for all devices')
            # Find the greatest number of channels for each device
            for d in formats.keys():
                channels = max(f.mChannelsPerFrame for f in formats[d])
                formats[d] = [f for f in formats[d] if f.mChannelsPerFrame == channels]
            # Find the least bit depth for each device
            for d in formats.keys():
                bitsPerChannel = min(f.mBitsPerChannel for f in formats[d])
                formats[d] = [f for f in formats[d] if f.mBitsPerChannel == bitsPerChannel]
            # Break ties and set format for each device
            for d, vf in formats.items():
                d[kAudioStreamPropertyVirtualFormat, scope] = self.vfDesired[d, scope] = vf[0]
                self.vfActual[d, scope] = d[kAudioStreamPropertyVirtualFormat, scope]
                d[kAudioDevicePropertyBufferFrameSize, scope] = self.bufSize[scope]
            self.negotiated[scope] = True

    def setMasterDevice(self, device):
        self.masterDeviceUID = device[kAudioDevicePropertyDeviceUID]

    def createAggregate(self):
        with self.cv:
            if self.created:
                return
            self.negotiateFormat(kAudioObjectPropertyScopeInput)
            self.negotiateFormat(kAudioObjectPropertyScopeOutput)
            uid = str(uuid.uuid4())
            devices = [d for d, s in self.deviceScopes]
            self.subDevices = sorted(set(devices), key=devices.index)
            if self.masterDeviceUID is None:
                self.masterDeviceUID = self.subDevices[0][kAudioDevicePropertyDeviceUID]
            composition = {
                kAudioAggregateDeviceUIDKey: uid,
                kAudioAggregateDeviceNameKey: 'Python Audio HAL Aggregate Device',
                kAudioAggregateDeviceIsPrivateKey: 1,
                kAudioAggregateDeviceIsStackedKey: 0,
                kAudioAggregateDeviceSubDeviceListKey: [{
                    kAudioSubDeviceUIDKey: d[kAudioDevicePropertyDeviceUID],
                    kAudioSubDeviceInputChannelsKey: self.vfActual[d,kAudioObjectPropertyScopeInput.value].mChannelsPerFrame if (d,kAudioObjectPropertyScopeInput.value) in self.vfActual else 0,
                    kAudioSubDeviceOutputChannelsKey: self.vfActual[d,kAudioObjectPropertyScopeOutput.value].mChannelsPerFrame if (d,kAudioObjectPropertyScopeOutput.value) in self.vfActual else 0,
                    kAudioSubDeviceExtraInputLatencyKey: 0,
                    kAudioSubDeviceExtraOutputLatencyKey: 0,
                    kAudioSubDeviceDriftCompensationKey: 1,
                    kAudioSubDeviceDriftCompensationQualityKey: kAudioSubDeviceDriftCompensationMaxQuality,
                } for d in self.subDevices],
                kAudioAggregateDeviceMasterSubDeviceKey: self.masterDeviceUID,
            }
            composition = CoreFoundation.CFDictionaryRef(composition)
            objectID = AudioObjectID()
            trap(AudioHardwareCreateAggregateDevice, composition.__c_void_p__(), byref(objectID))
            self.device = AudioObject(objectID)
            self.ioProcID = AudioDeviceIOProcID()
            trap(AudioDeviceCreateIOProcID, self.device.objectID, self.ioProc_ptr, None, byref(self.ioProcID))
            for scope in (kAudioObjectPropertyScopeInput, kAudioObjectPropertyScopeOutput):
                self.device[kAudioDevicePropertyBufferFrameSize, scope] = self.bufSize[scope.value]
            self.created = True

    def destroyAggregate(self):
        with self.cv:
            self.stop()
            if self.created:
                trap(AudioDeviceDestroyIOProcID, self.device.objectID, self.ioProcID)
                trap(AudioHardwareDestroyAggregateDevice, self.device.objectID)
                self.created = False

    def start(self, inStream=None, outStream=None, startHostTime=None):
        with self.cv:
            if self.running:
                return
            if startHostTime is None:
                startHostTime = mach_absolute_time()
            self.createAggregate()
            self.ioProcException = None
            self.running = True
            self.ioStartHostTime = startHostTime
            self.inStream = inStream
            self.outStream = outStream
            trap(AudioDeviceStart, self.device.objectID, self.ioProcID)

    def stop(self):
        with self.cv:
            if not self.running:
                return
            self.stopping = True
            self.wait()

    def wait(self):
        with self.cv:
            try:
                while self.running:
                    self.cv.wait()
            except KeyboardInterrupt:
                if not self.stopping:
                    self.stopping = True
                    while self.running:
                        self.cv.wait()
                else:
                    self.stopIO()
                    self.ioProcException = InterruptedError()
            if self.ioProcException:
                raise self.ioProcException
            if hasattr(self.inStream, 'stop'):
                self.inStream.stop()
            return self.inStream

    def stopIO(self):
        with self.cv:
            trap(AudioDeviceStop, self.device.objectID, self.ioProcID)
            self.running = False
            self.cv.notifyAll()

    def ioProc(self,
            inDevice: AudioObjectID,
            inNow: POINTER(AudioTimeStamp),
            inInputData: POINTER(AudioBufferList),
            inInputTime: POINTER(AudioTimeStamp),
            outOutputData: POINTER(AudioBufferList),
            inOutputTime: POINTER(AudioTimeStamp),
            inClientData: c_void_p) -> OSStatus:
        try:
            inNow = inNow.contents
            if not (inNow.mFlags & kAudioTimeStampHostTimeValid.value):
                raise Exception('No host timestamps')
            if inInputData and inInputData.contents.mNumberBuffers:
                inInputTime = inInputTime.contents
                inputBuffers = cast(inInputData.contents.mBuffers, POINTER(AudioBuffer))
                asbd = self.device[kAudioStreamPropertyVirtualFormat, kAudioObjectPropertyScopeInput]
                if not (inInputTime.mFlags & kAudioTimeStampHostTimeValid.value):
                    raise Exception('No host timestamps')
                self.inLatency = inNow.mHostTime - inInputTime.mHostTime;
                samples = np.concatenate([arrayFromBuffer(inputBuffers[i], asbd) for i in range(inInputData.contents.mNumberBuffers)], 1)
                nFrames = samples.shape[0]
                ticksPerFrame = 1e9 / (asbd.mSampleRate * self.nsPerAbsoluteTick)
                firstGoodSample = max(min((self.ioStartHostTime - inInputTime.mHostTime) / ticksPerFrame, nFrames), 0)
                if firstGoodSample:
                    samples = samples[firstGoodSample:]
                self.inStream.write(samples, inInputTime.mHostTime, inNow.mHostTime)
            if outOutputData and outOutputData.contents.mNumberBuffers:
                inOutputTime = inOutputTime.contents
                outputBuffers = cast(outOutputData.contents.mBuffers, POINTER(AudioBuffer))
                asbd = self.device[kAudioStreamPropertyVirtualFormat, kAudioObjectPropertyScopeOutput]
                if not (inOutputTime.mFlags & kAudioTimeStampHostTimeValid.value):
                    raise Exception('No host timestamps')
                self.outLatency = inOutputTime.mHostTime - inNow.mHostTime;
                b = outputBuffers[0]
                nFrames = b.mDataByteSize // asbd.mBytesPerFrame
                ticksPerFrame = 1e9 / (asbd.mSampleRate * self.nsPerAbsoluteTick)
                firstGoodSample = max(min((self.ioStartHostTime - inOutputTime.mHostTime) / ticksPerFrame, nFrames), 0)
                y = self.outStream.read(nFrames - firstGoodSample, inOutputTime.mHostTime, inNow.mHostTime)
                nFramesRead = y.shape[0]
                nextChannel = 0
                for i in range(outOutputData.contents.mNumberBuffers):
                    samples = arrayFromBuffer(outputBuffers[i], asbd)
                    if firstGoodSample:
                        samples[:firstGoodSample] = 0
                        samples = samples[firstGoodSample:]
                    mNumberChannels = outputBuffers[i].mNumberChannels
                    samples[:nFramesRead] = y[:,nextChannel:nextChannel+mNumberChannels]
                    samples[nFramesRead:] = 0
                    nextChannel += mNumberChannels
            inDone = not self.inStream or self.inStream.inDone()
            outDone = not self.outStream or self.outStream.outDone()
            if self.stopping or (inDone and outDone):
                self.stopIO()
        except Exception as e:
            with self.cv:
                self.ioProcException = e
                self.stopIO()
        return 0

    def propListener(self,
            objectID: AudioObjectID,
            inNumberAddresses: UInt32,
            inAddresses: POINTER(AudioObjectPropertyAddress),
            inClientData: c_void_p):
        obj = AudioObject(objectID)
        for i in range(inNumberAddresses):
            scope = inAddresses[i].mScope
            key = (obj, scope)
            if key in self.deviceScopes:
                if inAddresses[i].mSelector == kAudioStreamPropertyVirtualFormat.value:
                    self.vfActual[key] = obj[kAudioStreamPropertyVirtualFormat, scope]
                elif inAddresses[i].mSelector == kAudioDeviceProcessorOverload.value:
                    print('Overrun', file=sys.stderr)
        return 0

    def __enter__(self):
        self.createAggregate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroyAggregate()
        return exc_type is KeyboardInterrupt

def coerceInputStream(inStream):
    if isinstance(count_or_stream, IOStream):
        inStream = count_or_stream
    elif np.ndim(count_or_stream) == 0:
        inStream = InArrayStream(np.empty((int(count_or_stream), 1), np.float32))
    else:
        inStream = InArrayStream(np.asarray(count_or_stream))
    return inStream

def coerceOutputStream(outStream):
    if not isinstance(outStream, IOStream):
        outStream = OutArrayStream(np.asarray(outStream))
    return outStream

def prepareSession(Fs, outDevice, inDevice):
    ios = IOSession()
    try:
        iter(outDevice)
        outDevices = outDevice
    except TypeError:
        outDevices = (outDevice,) if outDevice else ()
    try:
        iter(inDevice)
        inDevices = inDevice
    except TypeError:
        inDevices = (inDevice,) if inDevice else ()
    if outDevices:
        ios.negotiateFormat(kAudioObjectPropertyScopeOutput, minimumSampleRate=Fs, maximumSampleRate=Fs)
        for device in outDevices:
            ios.addDevice(device, kAudioObjectPropertyScopeOutput)
    if inDevices:
        ios.negotiateFormat(kAudioObjectPropertyScopeInput, minimumSampleRate=Fs, maximumSampleRate=Fs)
        for device in inDevices:
            ios.addDevice(device, kAudioObjectPropertyScopeInput)
    return ios

def play(outStream, Fs, outDevice=None):
    outStream = coerceOutputStream(outStream)
    with prepareSession(Fs, outDevice, None) as ios:
        ios.start(outStream=outStream)
        return ios.wait()

def record(count_or_stream, Fs, inDevice=None):
    inStream = coerceInputStream(count_or_stream)
    with prepareSession(Fs, None, inDevice) as ios:
        ios.start(inStream=inStream)
        return ios.wait()

def play_and_record(stream, Fs, outDevice=None, inDevice=None):
    inStream = coerceInputStream(stream)
    outStream = coerceOutputStream(stream)
    with prepareSession(Fs, outDevice, inDevice) as ios:
        ios.start(inStream=inStream, outStream=outStream)
        return ios.wait()

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

def findInputLevelControl(device):
    for c in device[kAudioObjectPropertyControlList]:
        if c[kAudioControlPropertyScope].value != kAudioObjectPropertyScopeInput.value:
            continue
        if c.classID != kAudioVolumeControlClassID.value:
            continue
        return c
