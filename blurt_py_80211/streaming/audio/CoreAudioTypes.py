from MacTypes import *
import numpy as np

kAudio_UnimplementedError     = OSStatus(-4)
kAudio_FileNotFoundError      = OSStatus(-43)
kAudio_FilePermissionError    = OSStatus(-54)
kAudio_TooManyFilesOpenError  = OSStatus(-42)
kAudio_BadFilePathError       = OSStatus(fourcc('!pth'))
kAudio_ParamError             = OSStatus(-50)
kAudio_MemFullError           = OSStatus(-108)

class AudioValueRange(Structure):
    _fields_ = [
        ('mMinimum', Float64),
        ('mMaximum', Float64),
    ]

class AudioValueTranslation(Structure):
    _fields_ = [
        ('mInputData', c_void_p),
        ('mInputDataSize', UInt32),
        ('mOutputData', c_void_p),
        ('mOutputDataSize', UInt32),
    ]

class AudioBuffer(Structure):
    _fields_ = [
        ('mNumberChannels', UInt32),
        ('mDataByteSize', UInt32),
        ('mData', c_void_p),
    ]

class AudioBufferList(Structure):
    _fields_ = [
        ('mNumberBuffers', UInt32),
        ('mBuffers', AudioBuffer*1),
    ]

AudioFormatID = UInt32
AudioFormatFlags = UInt32

class AudioStreamBasicDescription(Structure):
    _fields_ = [
        ('mSampleRate', Float64),
        ('mFormatID', AudioFormatID),
        ('mFormatFlags', AudioFormatFlags),
        ('mBytesPerPacket', UInt32),
        ('mFramesPerPacket', UInt32),
        ('mBytesPerFrame', UInt32),
        ('mChannelsPerFrame', UInt32),
        ('mBitsPerChannel', UInt32),
        ('mReserved', UInt32),
    ]

kAudioStreamAnyRate = Float64(0.0)

kAudioFormatLinearPCM               = AudioFormatID(fourcc('lpcm'))
kAudioFormatAC3                     = AudioFormatID(fourcc('ac-3'))
kAudioFormat60958AC3                = AudioFormatID(fourcc('cac3'))
kAudioFormatAppleIMA4               = AudioFormatID(fourcc('ima4'))
kAudioFormatMPEG4AAC                = AudioFormatID(fourcc('aac '))
kAudioFormatMPEG4CELP               = AudioFormatID(fourcc('celp'))
kAudioFormatMPEG4HVXC               = AudioFormatID(fourcc('hvxc'))
kAudioFormatMPEG4TwinVQ             = AudioFormatID(fourcc('twvq'))
kAudioFormatMACE3                   = AudioFormatID(fourcc('MAC3'))
kAudioFormatMACE6                   = AudioFormatID(fourcc('MAC6'))
kAudioFormatULaw                    = AudioFormatID(fourcc('ulaw'))
kAudioFormatALaw                    = AudioFormatID(fourcc('alaw'))
kAudioFormatQDesign                 = AudioFormatID(fourcc('QDMC'))
kAudioFormatQDesign2                = AudioFormatID(fourcc('QDM2'))
kAudioFormatQUALCOMM                = AudioFormatID(fourcc('Qclp'))
kAudioFormatMPEGLayer1              = AudioFormatID(fourcc('.mp1'))
kAudioFormatMPEGLayer2              = AudioFormatID(fourcc('.mp2'))
kAudioFormatMPEGLayer3              = AudioFormatID(fourcc('.mp3'))
kAudioFormatTimeCode                = AudioFormatID(fourcc('time'))
kAudioFormatMIDIStream              = AudioFormatID(fourcc('midi'))
kAudioFormatParameterValueStream    = AudioFormatID(fourcc('apvs'))
kAudioFormatAppleLossless           = AudioFormatID(fourcc('alac'))
kAudioFormatMPEG4AAC_HE             = AudioFormatID(fourcc('aach'))
kAudioFormatMPEG4AAC_LD             = AudioFormatID(fourcc('aacl'))
kAudioFormatMPEG4AAC_ELD            = AudioFormatID(fourcc('aace'))
kAudioFormatMPEG4AAC_ELD_SBR        = AudioFormatID(fourcc('aacf'))
kAudioFormatMPEG4AAC_ELD_V2         = AudioFormatID(fourcc('aacg'))
kAudioFormatMPEG4AAC_HE_V2          = AudioFormatID(fourcc('aacp'))
kAudioFormatMPEG4AAC_Spatial        = AudioFormatID(fourcc('aacs'))
kAudioFormatAMR                     = AudioFormatID(fourcc('samr'))
kAudioFormatAMR_WB                  = AudioFormatID(fourcc('sawb'))
kAudioFormatAudible                 = AudioFormatID(fourcc('AUDB'))
kAudioFormatiLBC                    = AudioFormatID(fourcc('ilbc'))
kAudioFormatDVIIntelIMA             = AudioFormatID(0x6d730011)
kAudioFormatMicrosoftGSM            = AudioFormatID(0x6d730031)
kAudioFormatAES3                    = AudioFormatID(fourcc('aes3'))
kAudioFormatEnhancedAC3             = AudioFormatID(fourcc('ec-3'))
kAudioFormatFLAC                    = AudioFormatID(fourcc('flac'))
kAudioFormatOpus                    = AudioFormatID(fourcc('opus'))
kAudioFormatFlagIsFloat             = AudioFormatFlags(1 << 0)
kAudioFormatFlagIsBigEndian         = AudioFormatFlags(1 << 1)
kAudioFormatFlagIsSignedInteger     = AudioFormatFlags(1 << 2)
kAudioFormatFlagIsPacked            = AudioFormatFlags(1 << 3)
kAudioFormatFlagIsAlignedHigh       = AudioFormatFlags(1 << 4)
kAudioFormatFlagIsNonInterleaved    = AudioFormatFlags(1 << 5)
kAudioFormatFlagIsNonMixable        = AudioFormatFlags(1 << 6)
kAudioFormatFlagsAreAllClear        = AudioFormatFlags(0x80000000)
kLinearPCMFormatFlagIsFloat                 = AudioFormatFlags(kAudioFormatFlagIsFloat.value)
kLinearPCMFormatFlagIsBigEndian             = AudioFormatFlags(kAudioFormatFlagIsBigEndian.value)
kLinearPCMFormatFlagIsSignedInteger         = AudioFormatFlags(kAudioFormatFlagIsSignedInteger.value)
kLinearPCMFormatFlagIsPacked                = AudioFormatFlags(kAudioFormatFlagIsPacked.value)
kLinearPCMFormatFlagIsAlignedHigh           = AudioFormatFlags(kAudioFormatFlagIsAlignedHigh.value)
kLinearPCMFormatFlagIsNonInterleaved        = AudioFormatFlags(kAudioFormatFlagIsNonInterleaved.value)
kLinearPCMFormatFlagIsNonMixable            = AudioFormatFlags(kAudioFormatFlagIsNonMixable.value)
kLinearPCMFormatFlagsSampleFractionShift    = AudioFormatFlags(7)
kLinearPCMFormatFlagsSampleFractionMask     = AudioFormatFlags(0x3F << kLinearPCMFormatFlagsSampleFractionShift.value)
kLinearPCMFormatFlagsAreAllClear            = AudioFormatFlags(kAudioFormatFlagsAreAllClear.value)
kAppleLosslessFormatFlag_16BitSourceData    = AudioFormatFlags(1)
kAppleLosslessFormatFlag_20BitSourceData    = AudioFormatFlags(2)
kAppleLosslessFormatFlag_24BitSourceData    = AudioFormatFlags(3)
kAppleLosslessFormatFlag_32BitSourceData    = AudioFormatFlags(4)
kAudioFormatFlagsNativeEndian               = AudioFormatFlags(0)
kAudioFormatFlagsNativeFloatPacked          = AudioFormatFlags(kAudioFormatFlagIsFloat.value | kAudioFormatFlagsNativeEndian.value | kAudioFormatFlagIsPacked.value)

class AudioStreamPacketDescription(Structure):
    _fields_ = [
        ('mStartOffset', SInt64),
        ('mVariableFramesInPacket', UInt32),
        ('mDataByteSize', UInt32),
    ]

SMPTETimeType           = UInt32
kSMPTETimeType24        = SMPTETimeType(0)
kSMPTETimeType25        = SMPTETimeType(1)
kSMPTETimeType30Drop    = SMPTETimeType(2)
kSMPTETimeType30        = SMPTETimeType(3)
kSMPTETimeType2997      = SMPTETimeType(4)
kSMPTETimeType2997Drop  = SMPTETimeType(5)
kSMPTETimeType60        = SMPTETimeType(6)
kSMPTETimeType5994      = SMPTETimeType(7)
kSMPTETimeType60Drop    = SMPTETimeType(8)
kSMPTETimeType5994Drop  = SMPTETimeType(9)
kSMPTETimeType50        = SMPTETimeType(10)
kSMPTETimeType2398      = SMPTETimeType(11)
SMPTETimeFlags          = UInt32
kSMPTETimeUnknown       = SMPTETimeFlags(0)
kSMPTETimeValid         = SMPTETimeFlags(1 << 0)
kSMPTETimeRunning       = SMPTETimeFlags(1 << 1)

class SMPTETime(Structure):
    _fields_ = [
        ('mSubframes',           SInt16),
        ('mSubframeDivisor',     SInt16),
        ('mCounter',             UInt32),
        ('mType',                SMPTETimeType),
        ('mFlags',               SMPTETimeFlags),
        ('mHours',               SInt16),
        ('mMinutes',             SInt16),
        ('mSeconds',             SInt16),
        ('mFrames',              SInt16),
    ]

AudioTimeStampFlags = UInt32
kAudioTimeStampNothingValid         = AudioTimeStampFlags(0)
kAudioTimeStampSampleTimeValid      = AudioTimeStampFlags(1 << 0)
kAudioTimeStampHostTimeValid        = AudioTimeStampFlags(1 << 1)
kAudioTimeStampRateScalarValid      = AudioTimeStampFlags(1 << 2)
kAudioTimeStampWordClockTimeValid   = AudioTimeStampFlags(1 << 3)
kAudioTimeStampSMPTETimeValid       = AudioTimeStampFlags(1 << 4)
kAudioTimeStampSampleHostTimeValid  = AudioTimeStampFlags(kAudioTimeStampSampleTimeValid.value | kAudioTimeStampHostTimeValid.value)

class AudioTimeStamp(Structure):
    _fields_ = [
        ('mSampleTime', Float64),
        ('mHostTime', UInt64, ),
        ('mRateScalar', Float64),
        ('mWordClockTime', UInt64),
        ('mSMPTETime', SMPTETime),
        ('mFlags', AudioTimeStampFlags),
        ('mReserved', UInt32),
    ]

FourCharCode = UInt32
OSType = FourCharCode

class AudioClassDescription(Structure):
    _fields_ = [
        ('mType', OSType),
        ('mSubType', OSType),
        ('mManufacturer', OSType),
    ]

AudioChannelLabel                               = UInt32
AudioChannelLayoutTag                           = UInt32
kAudioChannelLabel_Unknown                      = AudioChannelLabel(0xffffffff)
kAudioChannelLabel_Unused                       = AudioChannelLabel(0)
kAudioChannelLabel_UseCoordinates               = AudioChannelLabel(100)
kAudioChannelLabel_Left                         = AudioChannelLabel(1)
kAudioChannelLabel_Right                        = AudioChannelLabel(2)
kAudioChannelLabel_Center                       = AudioChannelLabel(3)
kAudioChannelLabel_LFEScreen                    = AudioChannelLabel(4)
kAudioChannelLabel_LeftSurround                 = AudioChannelLabel(5)
kAudioChannelLabel_RightSurround                = AudioChannelLabel(6)
kAudioChannelLabel_LeftCenter                   = AudioChannelLabel(7)
kAudioChannelLabel_RightCenter                  = AudioChannelLabel(8)
kAudioChannelLabel_CenterSurround               = AudioChannelLabel(9)
kAudioChannelLabel_LeftSurroundDirect           = AudioChannelLabel(10)
kAudioChannelLabel_RightSurroundDirect          = AudioChannelLabel(11)
kAudioChannelLabel_TopCenterSurround            = AudioChannelLabel(12)
kAudioChannelLabel_VerticalHeightLeft           = AudioChannelLabel(13)
kAudioChannelLabel_VerticalHeightCenter         = AudioChannelLabel(14)
kAudioChannelLabel_VerticalHeightRight          = AudioChannelLabel(15)
kAudioChannelLabel_TopBackLeft                  = AudioChannelLabel(16)
kAudioChannelLabel_TopBackCenter                = AudioChannelLabel(17)
kAudioChannelLabel_TopBackRight                 = AudioChannelLabel(18)
kAudioChannelLabel_RearSurroundLeft             = AudioChannelLabel(33)
kAudioChannelLabel_RearSurroundRight            = AudioChannelLabel(34)
kAudioChannelLabel_LeftWide                     = AudioChannelLabel(35)
kAudioChannelLabel_RightWide                    = AudioChannelLabel(36)
kAudioChannelLabel_LFE2                         = AudioChannelLabel(37)
kAudioChannelLabel_LeftTotal                    = AudioChannelLabel(38)
kAudioChannelLabel_RightTotal                   = AudioChannelLabel(39)
kAudioChannelLabel_HearingImpaired              = AudioChannelLabel(40)
kAudioChannelLabel_Narration                    = AudioChannelLabel(41)
kAudioChannelLabel_Mono                         = AudioChannelLabel(42)
kAudioChannelLabel_DialogCentricMix             = AudioChannelLabel(43)
kAudioChannelLabel_CenterSurroundDirect         = AudioChannelLabel(44)
kAudioChannelLabel_Haptic                       = AudioChannelLabel(45)
kAudioChannelLabel_Ambisonic_W                  = AudioChannelLabel(200)
kAudioChannelLabel_Ambisonic_X                  = AudioChannelLabel(201)
kAudioChannelLabel_Ambisonic_Y                  = AudioChannelLabel(202)
kAudioChannelLabel_Ambisonic_Z                  = AudioChannelLabel(203)
kAudioChannelLabel_MS_Mid                       = AudioChannelLabel(204)
kAudioChannelLabel_MS_Side                      = AudioChannelLabel(205)
kAudioChannelLabel_XY_X                         = AudioChannelLabel(206)
kAudioChannelLabel_XY_Y                         = AudioChannelLabel(207)
kAudioChannelLabel_HeadphonesLeft               = AudioChannelLabel(301)
kAudioChannelLabel_HeadphonesRight              = AudioChannelLabel(302)
kAudioChannelLabel_ClickTrack                   = AudioChannelLabel(304)
kAudioChannelLabel_ForeignLanguage              = AudioChannelLabel(305)
kAudioChannelLabel_Discrete                     = AudioChannelLabel(400)
kAudioChannelLabel_Discrete_0                   = AudioChannelLabel((1<<16) | 0)
kAudioChannelLabel_Discrete_1                   = AudioChannelLabel((1<<16) | 1)
kAudioChannelLabel_Discrete_2                   = AudioChannelLabel((1<<16) | 2)
kAudioChannelLabel_Discrete_3                   = AudioChannelLabel((1<<16) | 3)
kAudioChannelLabel_Discrete_4                   = AudioChannelLabel((1<<16) | 4)
kAudioChannelLabel_Discrete_5                   = AudioChannelLabel((1<<16) | 5)
kAudioChannelLabel_Discrete_6                   = AudioChannelLabel((1<<16) | 6)
kAudioChannelLabel_Discrete_7                   = AudioChannelLabel((1<<16) | 7)
kAudioChannelLabel_Discrete_8                   = AudioChannelLabel((1<<16) | 8)
kAudioChannelLabel_Discrete_9                   = AudioChannelLabel((1<<16) | 9)
kAudioChannelLabel_Discrete_10                  = AudioChannelLabel((1<<16) | 10)
kAudioChannelLabel_Discrete_11                  = AudioChannelLabel((1<<16) | 11)
kAudioChannelLabel_Discrete_12                  = AudioChannelLabel((1<<16) | 12)
kAudioChannelLabel_Discrete_13                  = AudioChannelLabel((1<<16) | 13)
kAudioChannelLabel_Discrete_14                  = AudioChannelLabel((1<<16) | 14)
kAudioChannelLabel_Discrete_15                  = AudioChannelLabel((1<<16) | 15)
kAudioChannelLabel_Discrete_65535               = AudioChannelLabel((1<<16) | 65535)
kAudioChannelLabel_HOA_ACN                      = AudioChannelLabel(500)
kAudioChannelLabel_HOA_ACN_0                    = AudioChannelLabel((2 << 16) | 0)
kAudioChannelLabel_HOA_ACN_1                    = AudioChannelLabel((2 << 16) | 1)
kAudioChannelLabel_HOA_ACN_2                    = AudioChannelLabel((2 << 16) | 2)
kAudioChannelLabel_HOA_ACN_3                    = AudioChannelLabel((2 << 16) | 3)
kAudioChannelLabel_HOA_ACN_4                    = AudioChannelLabel((2 << 16) | 4)
kAudioChannelLabel_HOA_ACN_5                    = AudioChannelLabel((2 << 16) | 5)
kAudioChannelLabel_HOA_ACN_6                    = AudioChannelLabel((2 << 16) | 6)
kAudioChannelLabel_HOA_ACN_7                    = AudioChannelLabel((2 << 16) | 7)
kAudioChannelLabel_HOA_ACN_8                    = AudioChannelLabel((2 << 16) | 8)
kAudioChannelLabel_HOA_ACN_9                    = AudioChannelLabel((2 << 16) | 9)
kAudioChannelLabel_HOA_ACN_10                   = AudioChannelLabel((2 << 16) | 10)
kAudioChannelLabel_HOA_ACN_11                   = AudioChannelLabel((2 << 16) | 11)
kAudioChannelLabel_HOA_ACN_12                   = AudioChannelLabel((2 << 16) | 12)
kAudioChannelLabel_HOA_ACN_13                   = AudioChannelLabel((2 << 16) | 13)
kAudioChannelLabel_HOA_ACN_14                   = AudioChannelLabel((2 << 16) | 14)
kAudioChannelLabel_HOA_ACN_15                   = AudioChannelLabel((2 << 16) | 15)
kAudioChannelLabel_HOA_ACN_65024                = AudioChannelLabel((2 << 16) | 65024)
AudioChannelBitmap                              = UInt32
kAudioChannelBit_Left                           = AudioChannelBitmap(1<<0)
kAudioChannelBit_Right                          = AudioChannelBitmap(1<<1)
kAudioChannelBit_Center                         = AudioChannelBitmap(1<<2)
kAudioChannelBit_LFEScreen                      = AudioChannelBitmap(1<<3)
kAudioChannelBit_LeftSurround                   = AudioChannelBitmap(1<<4)
kAudioChannelBit_RightSurround                  = AudioChannelBitmap(1<<5)
kAudioChannelBit_LeftCenter                     = AudioChannelBitmap(1<<6)
kAudioChannelBit_RightCenter                    = AudioChannelBitmap(1<<7)
kAudioChannelBit_CenterSurround                 = AudioChannelBitmap(1<<8)
kAudioChannelBit_LeftSurroundDirect             = AudioChannelBitmap(1<<9)
kAudioChannelBit_RightSurroundDirect            = AudioChannelBitmap(1<<10)
kAudioChannelBit_TopCenterSurround              = AudioChannelBitmap(1<<11)
kAudioChannelBit_VerticalHeightLeft             = AudioChannelBitmap(1<<12)
kAudioChannelBit_VerticalHeightCenter           = AudioChannelBitmap(1<<13)
kAudioChannelBit_VerticalHeightRight            = AudioChannelBitmap(1<<14)
kAudioChannelBit_TopBackLeft                    = AudioChannelBitmap(1<<15)
kAudioChannelBit_TopBackCenter                  = AudioChannelBitmap(1<<16)
kAudioChannelBit_TopBackRight                   = AudioChannelBitmap(1<<17)
AudioChannelFlags                               = UInt32
kAudioChannelFlags_AllOff                       = AudioChannelFlags(0)
kAudioChannelFlags_RectangularCoordinates       = AudioChannelFlags(1<<0)
kAudioChannelFlags_SphericalCoordinates         = AudioChannelFlags(1<<1)
kAudioChannelFlags_Meters                       = AudioChannelFlags(1<<2)
AudioChannelCoordinateIndex                     = UInt32
kAudioChannelCoordinates_LeftRight              = AudioChannelCoordinateIndex(0)
kAudioChannelCoordinates_BackFront              = AudioChannelCoordinateIndex(1)
kAudioChannelCoordinates_DownUp                 = AudioChannelCoordinateIndex(2)
kAudioChannelCoordinates_Azimuth                = AudioChannelCoordinateIndex(0)
kAudioChannelCoordinates_Elevation              = AudioChannelCoordinateIndex(1)
kAudioChannelCoordinates_Distance               = AudioChannelCoordinateIndex(2)
kAudioChannelLayoutTag_UseChannelDescriptions   = AudioChannelLayoutTag((0<<16) | 0)
kAudioChannelLayoutTag_UseChannelBitmap         = AudioChannelLayoutTag((1<<16) | 0)
kAudioChannelLayoutTag_Mono                     = AudioChannelLayoutTag((100<<16) | 1)
kAudioChannelLayoutTag_Stereo                   = AudioChannelLayoutTag((101<<16) | 2)
kAudioChannelLayoutTag_StereoHeadphones         = AudioChannelLayoutTag((102<<16) | 2)
kAudioChannelLayoutTag_MatrixStereo             = AudioChannelLayoutTag((103<<16) | 2)
kAudioChannelLayoutTag_MidSide                  = AudioChannelLayoutTag((104<<16) | 2)
kAudioChannelLayoutTag_XY                       = AudioChannelLayoutTag((105<<16) | 2)
kAudioChannelLayoutTag_Binaural                 = AudioChannelLayoutTag((106<<16) | 2)
kAudioChannelLayoutTag_Ambisonic_B_Format       = AudioChannelLayoutTag((107<<16) | 4)

class AudioChannelDescription(Structure):
    _fields_ = [
        ('mChannelLabel', AudioChannelLabel),
        ('mChannelFlags', AudioChannelFlags),
        ('mCoordinates', Float32*3),
    ]

class AudioChannelLayout(Structure):
    _fields_ = [
        ('mChannelLayoutTag', AudioChannelLayoutTag),
        ('mChannelBitmap', AudioChannelBitmap),
        ('mNumberChannelDescriptions', UInt32),
        ('mChannelDescriptions', AudioChannelDescription*1),
    ]

def AudioChannelLayoutTag_GetNumberOfChannels(layoutTag):
    return UInt32(layoutTag & 0x0000ffff)

pythonapi.PyMemoryView_FromMemory.argtypes = (c_void_p, c_ssize_t, c_int)
pythonapi.PyMemoryView_FromMemory.restype = py_object

_dtypeCache = {}
def dtypeForStream(asbd: AudioStreamBasicDescription):
    key = asbd.mFormatID, asbd.mFormatFlags, asbd.mBitsPerChannel
    if key not in _dtypeCache:
        if asbd.mFormatFlags & kAudioFormatFlagIsFloat.value:
            fmt = 'f%d'
        elif asbd.mFormatFlags & kAudioFormatFlagIsSignedInteger.value:
            fmt = 'i%d'
        else:
            fmt = 'u%d'
        _dtypeCache[key] = np.dtype(fmt % (asbd.mBitsPerChannel // 8,))
    return _dtypeCache[key]

def arrayFromBuffer(b: AudioBuffer, asbd: AudioStreamBasicDescription):
    dtype = dtypeForStream(asbd)
    shape = (b.mDataByteSize // asbd.mBytesPerFrame, b.mNumberChannels)
    strides = (asbd.mBytesPerFrame, dtype.itemsize)
    data = pythonapi.PyMemoryView_FromMemory(b.mData, b.mDataByteSize, 0x200)
    a = np.ndarray(shape, dtype, data, 0, strides)
    a.flags.writeable = True
    return a

AudioBufferPointer = POINTER(AudioBuffer)
