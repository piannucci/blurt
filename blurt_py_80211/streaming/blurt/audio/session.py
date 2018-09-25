import sys
import numpy as np
import threading
import queue
import uuid
from .AudioHardware import *
from .mach_time import *
from .stream import IOStream, InArrayStream, OutArrayStream

defInBufSize = defOutBufSize = 2048

# all timestamps are compatible with time.monotonic(), which internally uses mach_absolute_time
class IOSession:
    def __init__(self):
        self.cv = threading.Condition()
        with self.cv:
            self.negotiated = {
                kAudioObjectPropertyScopeInput.value: False,
                kAudioObjectPropertyScopeOutput.value: False,
            }
            self.created = False
            self.running = False
            self.masterDeviceUID = None
            self.deviceScopes = []
            self.vfDesired = {} # indexed by device, scope
            self.vfActual = {} # indexed by device, scope
            self.notifiers = {} # indexed by device, scope
            self.bufSize = {} # indexed by scope
            self.ioProc_ptr = AudioDeviceIOProc(self.ioProc)
            self.inStream = None
            self.outStream = None

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

    def nChannelsPerFrame(self, scope):
        scope = getattr(scope, 'value', scope)
        with self.cv:
            self.negotiateFormat(scope)
            result = 0
            for d, s in self.deviceScopes:
                if s == scope:
                    result += self.vfActual[d, scope].mChannelsPerFrame
            return result

    def start(self, inStream=None, outStream=None, startTime=None):
        with self.cv:
            if self.running:
                return
            if startTime is None:
                startTime = time.monotonic()
            self.createAggregate()
            self.ioProcException = None
            self.running = True
            self.ioStartHostTime = monotonicToHostTime(startTime)
            if inStream is not None:
                self.inStream = inStream
            if outStream is not None:
                self.outStream = outStream
            trap(AudioDeviceStart, self.device.objectID, self.ioProcID)

    def stop(self):
        with self.cv:
            if not self.running:
                return
            self.stopIO()
            self.wait()

    def wait(self):
        with self.cv:
            while self.running:
                self.cv.wait()
            if self.ioProcException:
                raise self.ioProcException
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
                self.inLatency = (inNow.mHostTime - inInputTime.mHostTime) * d_monotonic_d_hostTime
                samples = np.concatenate([arrayFromBuffer(inputBuffers[i], asbd) for i in range(inInputData.contents.mNumberBuffers)], 1)
                nFrames = samples.shape[0]
                ticksPerFrame = d_hostTime_d_monotonic / asbd.mSampleRate
                firstGoodSample = max(min((self.ioStartHostTime - inInputTime.mHostTime) / ticksPerFrame, nFrames), 0)
                if firstGoodSample:
                    samples = samples[firstGoodSample:]
                self.inStream.write(samples, hostTimeToMonotonic(inInputTime.mHostTime), hostTimeToMonotonic(inNow.mHostTime))
            if outOutputData and outOutputData.contents.mNumberBuffers:
                inOutputTime = inOutputTime.contents
                outputBuffers = cast(outOutputData.contents.mBuffers, POINTER(AudioBuffer))
                asbd = self.device[kAudioStreamPropertyVirtualFormat, kAudioObjectPropertyScopeOutput]
                if not (inOutputTime.mFlags & kAudioTimeStampHostTimeValid.value):
                    raise Exception('No host timestamps')
                self.outLatency = (inOutputTime.mHostTime - inNow.mHostTime) * d_monotonic_d_hostTime
                b = outputBuffers[0]
                nFrames = b.mDataByteSize // asbd.mBytesPerFrame
                ticksPerFrame = d_hostTime_d_monotonic / asbd.mSampleRate
                firstGoodSample = max(min((self.ioStartHostTime - inOutputTime.mHostTime) / ticksPerFrame, nFrames), 0)
                y = self.outStream.read(nFrames - firstGoodSample, hostTimeToMonotonic(inOutputTime.mHostTime), hostTimeToMonotonic(inNow.mHostTime))
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
            if inDone and outDone:
                self.stopIO()
        except Exception as e:
            with self.cv:
                import traceback
                traceback.print_exc()
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
