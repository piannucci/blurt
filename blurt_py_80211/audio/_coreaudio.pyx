import numpy as np
cimport numpy as cnp
from cpython cimport *

cdef extern from "stdlib.h":
    ctypedef int size_t
    ctypedef long intptr_t
    void *malloc(size_t size)
    void free(void* ptr)

cdef extern from "numpy/arrayobject.h":
    PyObject PyArray_Type
    object PyArray_NewFromDescr (PyObject *subtype, cnp.dtype newdtype, int nd,
                                 cnp.npy_intp *dims, cnp.npy_intp *strides, void
                                 *data, int flags, object parent)

def FOUR_CHAR_CODE(val):
    from struct import unpack
    return unpack('!I', val)[0]

def INV_FOUR_CHAR_CODE(val):
    from struct import pack
    return pack('!I', val)

kAudioHardwarePropertyProcessIsMaster = FOUR_CHAR_CODE('mast')
kAudioHardwarePropertyIsInitingOrExiting = FOUR_CHAR_CODE('inot')
kAudioHardwarePropertyUserIDChanged = FOUR_CHAR_CODE('euid')
kAudioHardwarePropertyDevices = FOUR_CHAR_CODE('dev#')
kAudioHardwarePropertyDefaultInputDevice = FOUR_CHAR_CODE('dIn ')
kAudioHardwarePropertyDefaultOutputDevice = FOUR_CHAR_CODE('dOut')
kAudioHardwarePropertyDefaultSystemOutputDevice = FOUR_CHAR_CODE('sOut')
kAudioHardwarePropertyDeviceForUID = FOUR_CHAR_CODE('duid')
kAudioHardwarePropertyProcessIsAudible = FOUR_CHAR_CODE('pmut')
kAudioHardwarePropertySleepingIsAllowed = FOUR_CHAR_CODE('slep')
kAudioHardwarePropertyUnloadingIsAllowed = FOUR_CHAR_CODE('unld')
kAudioHardwarePropertyHogModeIsAllowed = FOUR_CHAR_CODE('hogr')
kAudioHardwarePropertyRunLoop = FOUR_CHAR_CODE('rnlp')
kAudioHardwarePropertyPlugInForBundleID = FOUR_CHAR_CODE('pibi')
kAudioHardwarePropertyUserSessionIsActiveOrHeadless = FOUR_CHAR_CODE('user')
kAudioHardwarePropertyMixStereoToMono = FOUR_CHAR_CODE('stmo')

kAudioDevicePropertyDeviceName = FOUR_CHAR_CODE('name')
kAudioDevicePropertyDeviceIsRunning = FOUR_CHAR_CODE('goin')
kAudioDevicePropertyNominalSampleRate = FOUR_CHAR_CODE('nsrt')
kAudioDevicePropertyAvailableNominalSampleRates = FOUR_CHAR_CODE('nsr#')
kAudioDevicePropertyActualSampleRate = FOUR_CHAR_CODE('asrt')
kAudioDevicePropertyBufferFrameSize = FOUR_CHAR_CODE('fsiz')
kAudioDevicePropertyStreamFormat = FOUR_CHAR_CODE('sfmt')

kAudioObjectPropertyScopeGlobal = FOUR_CHAR_CODE('glob')
kAudioObjectPropertyElementMaster = 0
kAudioObjectClassID = FOUR_CHAR_CODE('aobj')
kAudioObjectClassIDWildcard = FOUR_CHAR_CODE('****')
kAudioObjectUnknown = 0

kAudioObjectSystemObject = 1

kAudioDevicePropertyScopeInput = FOUR_CHAR_CODE('inpt')
kAudioDevicePropertyScopeOutput = FOUR_CHAR_CODE('outp')
kAudioDevicePropertyScopePlayThrough = FOUR_CHAR_CODE('ptru')
kAudioDeviceClassID = FOUR_CHAR_CODE('adev')

kAudioFormatLinearPCM = FOUR_CHAR_CODE('lpcm')

kAudioFormatFlagIsFloat = (1 << 0)
kAudioFormatFlagIsSignedInteger = (1 << 2)
kAudioFormatFlagIsPacked = (1 << 3)

cdef extern from "CoreAudio/AudioHardware.h":
    ctypedef unsigned int UInt32
    ctypedef unsigned int OSStatus
    ctypedef unsigned char Boolean

    ctypedef UInt32 AudioObjectID
    ctypedef UInt32 AudioHardwarePropertyID
    ctypedef UInt32 AudioDeviceID
    ctypedef UInt32 AudioDevicePropertyID
    ctypedef UInt32 AudioStreamID

    ctypedef UInt32 AudioObjectPropertySelector
    ctypedef UInt32 AudioObjectPropertyScope
    ctypedef UInt32 AudioObjectPropertyElement

    ctypedef double Float64
    ctypedef unsigned long long UInt64
    ctypedef short int SInt16

    ctypedef struct AudioValueRange:
        Float64 mMinimum
        Float64 mMaximum

    ctypedef struct SMPTETime:
        UInt64  mCounter #;         //  total number of messages received
        UInt32  mType #;                //  the SMPTE type (see constants)
        UInt32  mFlags #;               //  flags indicating state (see constants
        SInt16  mHours #;               //  number of hours in the full message
        SInt16  mMinutes #;         //  number of minutes in the full message
        SInt16  mSeconds #;         //  number of seconds in the full message
        SInt16  mFrames #;          //  number of frames in the full message

    ctypedef struct AudioTimeStamp:
        Float64         mSampleTime #;  //  the absolute sample time
        UInt64          mHostTime #;        //  the host's root timebase's time
        Float64         mRateScalar #;  //  the system rate scalar
        UInt64          mWordClockTime #;   //  the word clock time
        SMPTETime       mSMPTETime #;       //  the SMPTE time
        UInt32          mFlags #;           //  the flags indicate which fields are valid
        UInt32          mReserved #;        //  reserved, pads the structure out to force 8 byte alignment

    ctypedef struct AudioStreamBasicDescription:
        Float64 mSampleRate #;      //  the native sample rate of the audio stream
        UInt32  mFormatID #;            //  the specific encoding type of audio stream
        UInt32  mFormatFlags #;     //  flags specific to each format
        UInt32  mBytesPerPacket #;  //  the number of bytes in a packet
        UInt32  mFramesPerPacket #; //  the number of frames in each packet
        UInt32  mBytesPerFrame #;       //  the number of bytes in a frame
        UInt32  mChannelsPerFrame #;    //  the number of channels in each frame
        UInt32  mBitsPerChannel #;  //  the number of bits in each channel
        UInt32  mReserved #;            //  reserved, pads the structure out to force 8 byte alignment

    ctypedef struct AudioObjectPropertyAddress:
        AudioObjectPropertySelector mSelector
        AudioObjectPropertyScope mScope
        AudioObjectPropertyElement mElement

    ctypedef OSStatus (*AudioObjectPropertyListenerProc)(AudioObjectID inObjectID, UInt32 inNumberAddresses, AudioObjectPropertyAddress *inAddresses, void *inClientData)
    void AudioObjectShow(AudioObjectID inObjectID)
    Boolean AudioObjectHasProperty(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress)
    OSStatus AudioObjectIsPropertySettable(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress, Boolean *outIsSettable)

    OSStatus AudioObjectGetPropertyDataSize(
        AudioObjectID inObjectID,
        AudioObjectPropertyAddress *inAddress,
        UInt32 inQualifierDataSize,
        void *inQualifierData,
        UInt32 *outDataSize)

    OSStatus AudioObjectGetPropertyData(
        AudioObjectID inObjectID,
        AudioObjectPropertyAddress *inAddress,
        UInt32 inQualifierDataSize,
        void *inQualifierData,
        UInt32 *ioDataSize,
        void *outData)

    OSStatus AudioObjectSetPropertyData(
        AudioObjectID inObjectID,
        AudioObjectPropertyAddress *inAddress,
        UInt32 inQualifierDataSize,
        void *inQualifierData,
        UInt32 inDataSize,
        void *inData)

    OSStatus AudioObjectAddPropertyListener(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress, AudioObjectPropertyListenerProc inListener, void *inClientData)
    OSStatus AudioObjectRemovePropertyListener(AudioObjectID inObjectID, AudioObjectPropertyAddress *inAddress, AudioObjectPropertyListenerProc inListener, void *inClientData)

    ctypedef struct AudioBuffer:
        UInt32 mNumberChannels
        UInt32 mDataByteSize
        void* mData

    ctypedef struct AudioBufferList:
        UInt32 mNumberBuffers
        AudioBuffer mBuffers[1]

    ctypedef OSStatus (*AudioDeviceIOProc)(AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime, void *inClientData)
    ctypedef AudioDeviceIOProc AudioDeviceIOProcID
    OSStatus AudioDeviceStart(AudioDeviceID inDevice, AudioDeviceIOProc inProc)
    OSStatus AudioDeviceStop(AudioDeviceID inDevice, AudioDeviceIOProc inProc)
    OSStatus AudioDeviceRemoveIOProc(AudioDeviceID inDevice, AudioDeviceIOProc inProc)
    OSStatus AudioDeviceCreateIOProcID(AudioDeviceID inDevice, AudioDeviceIOProc inProc, void *inClientData, AudioDeviceIOProcID *outIOProcID)
    OSStatus AudioDeviceDestroyIOProcID(AudioDeviceID inDevice, AudioDeviceIOProcID inIOProcID)


cdef object arrayFromBuffer(AudioBuffer buffer, asbd):
    cdef UInt32 flags = asbd['mFormatFlags']

    if flags & kAudioFormatFlagIsFloat:
        format = 'f%d'
    elif flags & kAudioFormatFlagIsSignedInteger:
        format = 'i%d'
    else:
        format = 'u%d'

    cdef UInt32 bytesPerChannel = asbd['mBitsPerChannel'] // 8
    cdef cnp.dtype dt = np.dtype(format % bytesPerChannel)
    Py_INCREF(<object> dt)

    cdef UInt32 channelsPerFrame = asbd['mChannelsPerFrame']
    cdef UInt32 bytesPerFrame = asbd['mBytesPerFrame']

    cdef int ndims = 1 if channelsPerFrame == 1 else 2
    cdef cnp.npy_intp dims[2], strides[2]
    dims[0] = buffer.mDataByteSize // bytesPerFrame
    strides[0] = bytesPerFrame
    dims[1] = channelsPerFrame
    strides[1] = bytesPerChannel

    cdef UInt32 narrflags = cnp.NPY_WRITEABLE | cnp.NPY_C_CONTIGUOUS | cnp.NPY_F_CONTIGUOUS
    cdef cnp.ndarray narr = PyArray_NewFromDescr(&PyArray_Type, dt, ndims, dims, strides,
                                                 buffer.mData,
                                                 narrflags, <object>NULL)
    return narr


cdef OSStatus playbackCallback(
    AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime,
    void *inClientData) with gil:

    cdef object cb = <object> inClientData

    try:
        cb.playbackStarted = True
        for i from 0 <= i < outOutputData.mNumberBuffers:
            if cb.playbackCallback(arrayFromBuffer(outOutputData.mBuffers[i], cb.playbackASBD)):
                stopPlayback(cb)
                break
    except Exception, e:
        stopPlayback(cb)
        cb.playbackException = e

    return 0

cdef OSStatus recordingCallback(
    AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime,
    void *inClientData) with gil:

    cdef object cb = <object> inClientData

    try:
        cb.recordingStarted = True
        for i from 0 <= i < inInputData.mNumberBuffers:
            if cb.recordingCallback(arrayFromBuffer(inInputData.mBuffers[0], cb.recordingASBD)):
                stopRecording(cb)
                break
    except Exception, e:
        stopRecording(cb)
        cb.recordingException = e

    return 0

cdef AudioObjectGetGlobalProperty(AudioObjectID obj, AudioObjectPropertySelector selector, UInt32 propertySize, void *prop):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = kAudioObjectPropertyScopeGlobal
    address.mElement = kAudioObjectPropertyElementMaster
    cdef OSStatus status = AudioObjectGetPropertyData(obj, &address, 0, NULL, &propertySize, prop)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to get property %s" % pack("!I", selector)

cdef AudioObjectGetGlobalPropertySize(AudioObjectID obj, AudioObjectPropertySelector selector):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = kAudioObjectPropertyScopeGlobal
    address.mElement = kAudioObjectPropertyElementMaster
    cdef UInt32 propertySize
    cdef OSStatus status = AudioObjectGetPropertyDataSize(obj, &address, 0, NULL, &propertySize)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to get property size %s" % pack("!I", selector)
    return propertySize

cdef AudioObjectSetOutputProperty(AudioObjectID obj, AudioObjectPropertySelector selector, UInt32 propertySize, void *prop):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = kAudioDevicePropertyScopeOutput
    address.mElement = 0
    cdef OSStatus status = AudioObjectSetPropertyData(obj, &address, 0, NULL, propertySize, prop)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to set property %s" % pack("!I", selector)

cdef AudioObjectSetInputProperty(AudioObjectID obj, AudioObjectPropertySelector selector, UInt32 propertySize, void *prop):
    cdef AudioObjectPropertyAddress address
    address.mSelector = selector
    address.mScope = kAudioDevicePropertyScopeInput
    address.mElement = 0
    cdef OSStatus status = AudioObjectSetPropertyData(obj, &address, 0, NULL, propertySize, prop)
    if status:
        from struct import pack
        raise RuntimeError, "Unable to set property %s" % pack("!I", selector)

def getDevices():
    cdef int size = AudioObjectGetGlobalPropertySize(kAudioObjectSystemObject, kAudioHardwarePropertyDevices)
    cdef UInt32 *devices = <UInt32 *>malloc(size)
    AudioObjectGetGlobalProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDevices, sizeof(devices), devices)
    result = [None] * (size/4)
    for i from 0 <= i < size / 4:
        result[i] = devices[i]
    free(devices)
    return result

def startPlayback(cb, sampleRate, device):
    cdef AudioDeviceID outputDeviceID = 0
    if device is None:
        # Get the default sound output device
        AudioObjectGetGlobalProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultOutputDevice, sizeof(outputDeviceID), &outputDeviceID)
        if not outputDeviceID:
            raise RuntimeError, "Default audio device was unknown."
    else:
        devices = getDevices()
        if 0 <= device < len(devices):
            outputDeviceID = devices[device]
        else:
            raise RuntimeError, "No such audio device."

    cdef UInt32 bufSize = 512
    AudioObjectSetOutputProperty(outputDeviceID, kAudioDevicePropertyBufferFrameSize, sizeof(bufSize), &bufSize)

    cdef AudioStreamBasicDescription sbd
    sbd.mSampleRate = sampleRate
    sbd.mFormatID = kAudioFormatLinearPCM
    sbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
    sbd.mBytesPerPacket = 8
    sbd.mFramesPerPacket = 1
    sbd.mBytesPerFrame = 8
    sbd.mChannelsPerFrame = 2
    sbd.mBitsPerChannel = 32
    sbd.mReserved = 0
    AudioObjectSetOutputProperty(outputDeviceID, kAudioDevicePropertyStreamFormat, sizeof(sbd), &sbd)

    cdef AudioDeviceIOProcID ioProcID
    if AudioDeviceCreateIOProcID(outputDeviceID, <AudioDeviceIOProc>&playbackCallback, <void *>cb, &ioProcID):
        raise RuntimeError, "Failed to add the IO Proc."
    if AudioDeviceStart(outputDeviceID, ioProcID):
        raise RuntimeError, "Couldn't start the device."

    Py_INCREF(cb)
    cb.playbackDeviceID = outputDeviceID
    cb.playbackFs = sampleRate
    cb.playbackIOProcID = <long>ioProcID
    cb.playbackASBD = sbd
    cb.playbackStarted = False


def stopPlayback(cb):
    cdef AudioDeviceIOProcID ioProcID = <AudioDeviceIOProcID><long>cb.playbackIOProcID
    AudioDeviceStop(cb.playbackDeviceID, ioProcID)
    AudioDeviceDestroyIOProcID(cb.playbackDeviceID, ioProcID)
    del cb.playbackDeviceID
    del cb.playbackFs
    del cb.playbackIOProcID
    del cb.playbackASBD
    del cb.playbackStarted
    Py_DECREF(cb)


def startRecording(cb, sampleRate, device):
    # Get the default sound input device
    cdef AudioDeviceID inputDeviceID = 0
    AudioObjectGetGlobalProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultInputDevice, sizeof(inputDeviceID), &inputDeviceID)
    if not inputDeviceID:
        raise RuntimeError, "Default audio device was unknown."

    cdef UInt32 bufSize = 512
    AudioObjectSetInputProperty(inputDeviceID, kAudioDevicePropertyBufferFrameSize, sizeof(bufSize), &bufSize)

    cdef AudioStreamBasicDescription sbd
    sbd.mSampleRate = sampleRate
    sbd.mFormatID = kAudioFormatLinearPCM
    sbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
    sbd.mBytesPerPacket = 8
    sbd.mFramesPerPacket = 1
    sbd.mBytesPerFrame = 8
    sbd.mChannelsPerFrame = 2
    sbd.mBitsPerChannel = 32
    sbd.mReserved = 0
    AudioObjectSetInputProperty(inputDeviceID, kAudioDevicePropertyStreamFormat, sizeof(sbd), &sbd)

    cdef AudioDeviceIOProcID ioProcID
    if AudioDeviceCreateIOProcID(inputDeviceID, <AudioDeviceIOProc>&recordingCallback, <void *>cb, &ioProcID):
        raise RuntimeError, "Failed to add the IO Proc."
    if AudioDeviceStart(inputDeviceID, ioProcID):
        raise RuntimeError, "Couldn't start the device."

    Py_INCREF(cb)
    cb.recordingDeviceID = inputDeviceID
    cb.recordingFs = sampleRate
    cb.recordingIOProcID = <long>ioProcID
    cb.recordingASBD = sbd
    cb.recordingStarted = False

def stopRecording(cb):
    cdef AudioDeviceIOProcID ioProcID = <AudioDeviceIOProcID><long>cb.recordingIOProcID
    AudioDeviceStop(cb.recordingDeviceID, ioProcID)
    AudioDeviceDestroyIOProcID(cb.recordingDeviceID, ioProcID)
    del cb.recordingDeviceID
    del cb.recordingFs
    del cb.recordingIOProcID
    del cb.recordingASBD
    del cb.recordingStarted
    Py_DECREF(cb)


cnp.import_array()
