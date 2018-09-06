import objc, CoreFoundation
from ctypes import Array as ctypes_Array
from .MacTypes import *
from .AudioHardwareBase import *

CFDictionaryRef = c_void_p

AudioObjectPropertyListenerProc = CFUNCTYPE(OSStatus, AudioObjectID, UInt32, POINTER(AudioObjectPropertyAddress), c_void_p)
AudioDeviceIOProc = CFUNCTYPE(OSStatus, AudioObjectID, POINTER(AudioTimeStamp), POINTER(AudioBufferList), POINTER(AudioTimeStamp), POINTER(AudioBufferList), POINTER(AudioTimeStamp), c_void_p)
AudioDeviceIOProcID = AudioDeviceIOProc

class AudioHardwareIOProcStreamUsage(Structure):
    _fields_ = [
        ('mIOProc', c_void_p),
        ('mNumberStreams', UInt32),
        ('mStreamIsOn', UInt32*1),
    ]

kAudioObjectPropertyCreator                                     = AudioObjectPropertySelector(fourcc('oplg'))
kAudioObjectPropertyListenerAdded                               = AudioObjectPropertySelector(fourcc('lisa'))
kAudioObjectPropertyListenerRemoved                             = AudioObjectPropertySelector(fourcc('lisr'))
kAudioObjectSystemObject                                        = UInt32(1)
kAudioSystemObjectClassID                                       = AudioClassID(fourcc('asys'))
AudioHardwarePowerHint                                          = UInt32
kAudioHardwarePowerHintNone                                     = AudioHardwarePowerHint(0)
kAudioHardwarePowerHintFavorSavingPower                         = AudioHardwarePowerHint(1)
kAudioHardwarePropertyDevices                                   = AudioObjectPropertySelector(fourcc('dev#'))
kAudioHardwarePropertyDefaultInputDevice                        = AudioObjectPropertySelector(fourcc('dIn '))
kAudioHardwarePropertyDefaultOutputDevice                       = AudioObjectPropertySelector(fourcc('dOut'))
kAudioHardwarePropertyDefaultSystemOutputDevice                 = AudioObjectPropertySelector(fourcc('sOut'))
kAudioHardwarePropertyTranslateUIDToDevice                      = AudioObjectPropertySelector(fourcc('uidd'))
kAudioHardwarePropertyMixStereoToMono                           = AudioObjectPropertySelector(fourcc('stmo'))
kAudioHardwarePropertyPlugInList                                = AudioObjectPropertySelector(fourcc('plg#'))
kAudioHardwarePropertyTranslateBundleIDToPlugIn                 = AudioObjectPropertySelector(fourcc('bidp'))
kAudioHardwarePropertyTransportManagerList                      = AudioObjectPropertySelector(fourcc('tmg#'))
kAudioHardwarePropertyTranslateBundleIDToTransportManager       = AudioObjectPropertySelector(fourcc('tmbi'))
kAudioHardwarePropertyBoxList                                   = AudioObjectPropertySelector(fourcc('box#'))
kAudioHardwarePropertyTranslateUIDToBox                         = AudioObjectPropertySelector(fourcc('uidb'))
kAudioHardwarePropertyClockDeviceList                           = AudioObjectPropertySelector(fourcc('clk#'))
kAudioHardwarePropertyTranslateUIDToClockDevice                 = AudioObjectPropertySelector(fourcc('uidc'))
kAudioHardwarePropertyProcessIsMaster                           = AudioObjectPropertySelector(fourcc('mast'))
kAudioHardwarePropertyIsInitingOrExiting                        = AudioObjectPropertySelector(fourcc('inot'))
kAudioHardwarePropertyUserIDChanged                             = AudioObjectPropertySelector(fourcc('euid'))
kAudioHardwarePropertyProcessIsAudible                          = AudioObjectPropertySelector(fourcc('pmut'))
kAudioHardwarePropertySleepingIsAllowed                         = AudioObjectPropertySelector(fourcc('slep'))
kAudioHardwarePropertyUnloadingIsAllowed                        = AudioObjectPropertySelector(fourcc('unld'))
kAudioHardwarePropertyHogModeIsAllowed                          = AudioObjectPropertySelector(fourcc('hogr'))
kAudioHardwarePropertyUserSessionIsActiveOrHeadless             = AudioObjectPropertySelector(fourcc('user'))
kAudioHardwarePropertyServiceRestarted                          = AudioObjectPropertySelector(fourcc('srst'))
kAudioHardwarePropertyPowerHint                                 = AudioObjectPropertySelector(fourcc('powh'))
kAudioPlugInCreateAggregateDevice                               = AudioObjectPropertySelector(fourcc('cagg'))
kAudioPlugInDestroyAggregateDevice                              = AudioObjectPropertySelector(fourcc('dagg'))
kAudioTransportManagerCreateEndPointDevice                      = AudioObjectPropertySelector(fourcc('cdev'))
kAudioTransportManagerDestroyEndPointDevice                     = AudioObjectPropertySelector(fourcc('ddev'))
kAudioDeviceStartTimeIsInputFlag                                = UInt32(1 << 0)
kAudioDeviceStartTimeDontConsultDeviceFlag                      = UInt32(1 << 1)
kAudioDeviceStartTimeDontConsultHALFlag                         = UInt32(1 << 2)
kAudioDevicePropertyPlugIn                                      = AudioObjectPropertySelector(fourcc('plug'))
kAudioDevicePropertyDeviceHasChanged                            = AudioObjectPropertySelector(fourcc('diff'))
kAudioDevicePropertyDeviceIsRunningSomewhere                    = AudioObjectPropertySelector(fourcc('gone'))
kAudioDeviceProcessorOverload                                   = AudioObjectPropertySelector(fourcc('over'))
kAudioDevicePropertyIOStoppedAbnormally                         = AudioObjectPropertySelector(fourcc('stpd'))
kAudioDevicePropertyHogMode                                     = AudioObjectPropertySelector(fourcc('oink'))
kAudioDevicePropertyBufferFrameSize                             = AudioObjectPropertySelector(fourcc('fsiz'))
kAudioDevicePropertyBufferFrameSizeRange                        = AudioObjectPropertySelector(fourcc('fsz#'))
kAudioDevicePropertyUsesVariableBufferFrameSizes                = AudioObjectPropertySelector(fourcc('vfsz'))
kAudioDevicePropertyIOCycleUsage                                = AudioObjectPropertySelector(fourcc('ncyc'))
kAudioDevicePropertyStreamConfiguration                         = AudioObjectPropertySelector(fourcc('slay'))
kAudioDevicePropertyIOProcStreamUsage                           = AudioObjectPropertySelector(fourcc('suse'))
kAudioDevicePropertyActualSampleRate                            = AudioObjectPropertySelector(fourcc('asrt'))
kAudioDevicePropertyClockDevice                                 = AudioObjectPropertySelector(fourcc('apcd'))
kAudioDevicePropertyJackIsConnected                             = AudioObjectPropertySelector(fourcc('jack'))
kAudioDevicePropertyVolumeScalar                                = AudioObjectPropertySelector(fourcc('volm'))
kAudioDevicePropertyVolumeDecibels                              = AudioObjectPropertySelector(fourcc('vold'))
kAudioDevicePropertyVolumeRangeDecibels                         = AudioObjectPropertySelector(fourcc('vdb#'))
kAudioDevicePropertyVolumeScalarToDecibels                      = AudioObjectPropertySelector(fourcc('v2db'))
kAudioDevicePropertyVolumeDecibelsToScalar                      = AudioObjectPropertySelector(fourcc('db2v'))
kAudioDevicePropertyStereoPan                                   = AudioObjectPropertySelector(fourcc('span'))
kAudioDevicePropertyStereoPanChannels                           = AudioObjectPropertySelector(fourcc('spn#'))
kAudioDevicePropertyMute                                        = AudioObjectPropertySelector(fourcc('mute'))
kAudioDevicePropertySolo                                        = AudioObjectPropertySelector(fourcc('solo'))
kAudioDevicePropertyPhantomPower                                = AudioObjectPropertySelector(fourcc('phan'))
kAudioDevicePropertyPhaseInvert                                 = AudioObjectPropertySelector(fourcc('phsi'))
kAudioDevicePropertyClipLight                                   = AudioObjectPropertySelector(fourcc('clip'))
kAudioDevicePropertyTalkback                                    = AudioObjectPropertySelector(fourcc('talb'))
kAudioDevicePropertyListenback                                  = AudioObjectPropertySelector(fourcc('lsnb'))
kAudioDevicePropertyDataSource                                  = AudioObjectPropertySelector(fourcc('ssrc'))
kAudioDevicePropertyDataSources                                 = AudioObjectPropertySelector(fourcc('ssc#'))
kAudioDevicePropertyDataSourceNameForIDCFString                 = AudioObjectPropertySelector(fourcc('lscn'))
kAudioDevicePropertyDataSourceKindForID                         = AudioObjectPropertySelector(fourcc('ssck'))
kAudioDevicePropertyClockSource                                 = AudioObjectPropertySelector(fourcc('csrc'))
kAudioDevicePropertyClockSources                                = AudioObjectPropertySelector(fourcc('csc#'))
kAudioDevicePropertyClockSourceNameForIDCFString                = AudioObjectPropertySelector(fourcc('lcsn'))
kAudioDevicePropertyClockSourceKindForID                        = AudioObjectPropertySelector(fourcc('csck'))
kAudioDevicePropertyPlayThru                                    = AudioObjectPropertySelector(fourcc('thru'))
kAudioDevicePropertyPlayThruSolo                                = AudioObjectPropertySelector(fourcc('thrs'))
kAudioDevicePropertyPlayThruVolumeScalar                        = AudioObjectPropertySelector(fourcc('mvsc'))
kAudioDevicePropertyPlayThruVolumeDecibels                      = AudioObjectPropertySelector(fourcc('mvdb'))
kAudioDevicePropertyPlayThruVolumeRangeDecibels                 = AudioObjectPropertySelector(fourcc('mvd#'))
kAudioDevicePropertyPlayThruVolumeScalarToDecibels              = AudioObjectPropertySelector(fourcc('mv2d'))
kAudioDevicePropertyPlayThruVolumeDecibelsToScalar              = AudioObjectPropertySelector(fourcc('mv2s'))
kAudioDevicePropertyPlayThruStereoPan                           = AudioObjectPropertySelector(fourcc('mspn'))
kAudioDevicePropertyPlayThruStereoPanChannels                   = AudioObjectPropertySelector(fourcc('msp#'))
kAudioDevicePropertyPlayThruDestination                         = AudioObjectPropertySelector(fourcc('mdds'))
kAudioDevicePropertyPlayThruDestinations                        = AudioObjectPropertySelector(fourcc('mdd#'))
kAudioDevicePropertyPlayThruDestinationNameForIDCFString        = AudioObjectPropertySelector(fourcc('mddc'))
kAudioDevicePropertyChannelNominalLineLevel                     = AudioObjectPropertySelector(fourcc('nlvl'))
kAudioDevicePropertyChannelNominalLineLevels                    = AudioObjectPropertySelector(fourcc('nlv#'))
kAudioDevicePropertyChannelNominalLineLevelNameForIDCFString    = AudioObjectPropertySelector(fourcc('lcnl'))
kAudioDevicePropertyHighPassFilterSetting                       = AudioObjectPropertySelector(fourcc('hipf'))
kAudioDevicePropertyHighPassFilterSettings                      = AudioObjectPropertySelector(fourcc('hip#'))
kAudioDevicePropertyHighPassFilterSettingNameForIDCFString      = AudioObjectPropertySelector(fourcc('hipl'))
kAudioDevicePropertySubVolumeScalar                             = AudioObjectPropertySelector(fourcc('svlm'))
kAudioDevicePropertySubVolumeDecibels                           = AudioObjectPropertySelector(fourcc('svld'))
kAudioDevicePropertySubVolumeRangeDecibels                      = AudioObjectPropertySelector(fourcc('svd#'))
kAudioDevicePropertySubVolumeScalarToDecibels                   = AudioObjectPropertySelector(fourcc('sv2d'))
kAudioDevicePropertySubVolumeDecibelsToScalar                   = AudioObjectPropertySelector(fourcc('sd2v'))
kAudioDevicePropertySubMute                                     = AudioObjectPropertySelector(fourcc('smut'))
kAudioAggregateDeviceClassID                        = AudioClassID(fourcc('aagg'))
kAudioAggregateDeviceUIDKey                         = "uid"
kAudioAggregateDeviceNameKey                        = "name"
kAudioAggregateDeviceSubDeviceListKey               = "subdevices"
kAudioAggregateDeviceMasterSubDeviceKey             = "master"
kAudioAggregateDeviceClockDeviceKey                 = "clock"
kAudioAggregateDeviceIsPrivateKey                   = "private"
kAudioAggregateDeviceIsStackedKey                   = "stacked"
kAudioAggregateDevicePropertyFullSubDeviceList      = AudioObjectPropertySelector(fourcc('grup'))
kAudioAggregateDevicePropertyActiveSubDeviceList    = AudioObjectPropertySelector(fourcc('agrp'))
kAudioAggregateDevicePropertyComposition            = AudioObjectPropertySelector(fourcc('acom'))
kAudioAggregateDevicePropertyMasterSubDevice        = AudioObjectPropertySelector(fourcc('amst'))
kAudioAggregateDevicePropertyClockDevice            = AudioObjectPropertySelector(fourcc('apcd'))
kAudioSubDeviceClassID                              = AudioClassID(fourcc('asub'))
kAudioSubDeviceDriftCompensationMinQuality          = UInt32(0)
kAudioSubDeviceDriftCompensationLowQuality          = UInt32(0x20)
kAudioSubDeviceDriftCompensationMediumQuality       = UInt32(0x40)
kAudioSubDeviceDriftCompensationHighQuality         = UInt32(0x60)
kAudioSubDeviceDriftCompensationMaxQuality          = UInt32(0x7f)
kAudioSubDeviceUIDKey                               = "uid"
kAudioSubDeviceNameKey                              = "name"
kAudioSubDeviceInputChannelsKey                     = "channels-in"
kAudioSubDeviceOutputChannelsKey                    = "channels-out"
kAudioSubDeviceExtraInputLatencyKey                 = "latency-in"
kAudioSubDeviceExtraOutputLatencyKey                = "latency-out"
kAudioSubDeviceDriftCompensationKey                 = "drift"
kAudioSubDeviceDriftCompensationQualityKey          = "drift quality"
kAudioSubDevicePropertyExtraLatency                 = AudioObjectPropertySelector(fourcc('xltc'))
kAudioSubDevicePropertyDriftCompensation            = AudioObjectPropertySelector(fourcc('drft'))
kAudioSubDevicePropertyDriftCompensationQuality     = AudioObjectPropertySelector(fourcc('drfq'))

class Array:
    def __init__(self, of):
        self.of = of

class Translation:
    def __init__(self, fromType, toType):
        self.fromType = fromType
        self.toType = toType

class Object:
    pass

class CFObject:
    def __new__(self, *args):
        return c_void_p(*args)
class CFString(CFObject): pass
class CFURL(CFObject): pass
class CFDictionary(CFObject): pass
class CFArray(CFObject): pass

pid_t = c_int32
AudioServerPlugIn_PropertyScope = AudioObjectPropertyScope
AudioServerPlugIn_PropertyElement = AudioObjectPropertyElement

class PropInfo:
    def __init__(self, valueType, qualType=None):
        self.valueType = valueType
        self.qualType = qualType

audioPropertyInfo = {
    kAudioObjectPropertyBaseClass.value: PropInfo(AudioClassID),
    kAudioObjectPropertyClass.value: PropInfo(AudioClassID),
    kAudioObjectPropertyOwner.value: PropInfo(Object),
    kAudioObjectPropertyName.value: PropInfo(CFString),
    kAudioObjectPropertyModelName.value: PropInfo(CFString),
    kAudioObjectPropertyManufacturer.value: PropInfo(CFString),
    kAudioObjectPropertyElementName.value: PropInfo(CFString),
    kAudioObjectPropertyElementCategoryName.value: PropInfo(CFString),
    kAudioObjectPropertyElementNumberName.value: PropInfo(CFString),
    kAudioObjectPropertyOwnedObjects.value: PropInfo(Array(Object), Array(AudioClassID)),
    kAudioObjectPropertyIdentify.value: PropInfo(UInt32),
    kAudioObjectPropertySerialNumber.value: PropInfo(CFString),
    kAudioObjectPropertyFirmwareVersion.value: PropInfo(CFString),
    kAudioPlugInPropertyBundleID.value: PropInfo(CFString),
    kAudioPlugInPropertyDeviceList.value: PropInfo(Array(Object)),
    kAudioPlugInPropertyTranslateUIDToDevice.value: PropInfo(Object),
    kAudioPlugInPropertyBoxList.value: PropInfo(Array(Object)),
    kAudioPlugInPropertyTranslateUIDToBox.value: PropInfo(Array(Object)),
    kAudioPlugInPropertyClockDeviceList.value: PropInfo(Array(Object)),
    kAudioPlugInPropertyTranslateUIDToClockDevice.value: PropInfo(Object),
    kAudioTransportManagerPropertyEndPointList.value: PropInfo(Array(Object)),
    kAudioTransportManagerPropertyTranslateUIDToEndPoint.value: PropInfo(Object),
    kAudioTransportManagerPropertyTransportType.value: PropInfo(UInt32),
    kAudioBoxPropertyBoxUID.value: PropInfo(CFString),
    kAudioBoxPropertyTransportType.value: PropInfo(UInt32),
    kAudioBoxPropertyHasAudio.value: PropInfo(UInt32),
    kAudioBoxPropertyHasVideo.value: PropInfo(UInt32),
    kAudioBoxPropertyHasMIDI.value: PropInfo(UInt32),
    kAudioBoxPropertyIsProtected.value: PropInfo(UInt32),
    kAudioBoxPropertyAcquired.value: PropInfo(UInt32),
    kAudioBoxPropertyAcquisitionFailed.value: PropInfo(OSStatus),
    kAudioBoxPropertyDeviceList.value: PropInfo(Array(Object)),
    kAudioBoxPropertyClockDeviceList.value: PropInfo(Array(Object)),
    kAudioDevicePropertyConfigurationApplication.value: PropInfo(CFString),
    kAudioDevicePropertyDeviceUID.value: PropInfo(CFString),
    kAudioDevicePropertyModelUID.value: PropInfo(CFString),
    kAudioDevicePropertyTransportType.value: PropInfo(UInt32),
    kAudioDevicePropertyRelatedDevices.value: PropInfo(Array(Object)),
    kAudioDevicePropertyClockDomain.value: PropInfo(UInt32),
    kAudioDevicePropertyDeviceIsAlive.value: PropInfo(UInt32),
    kAudioDevicePropertyDeviceIsRunning.value: PropInfo(UInt32),
    kAudioDevicePropertyDeviceCanBeDefaultDevice.value: PropInfo(UInt32),
    kAudioDevicePropertyDeviceCanBeDefaultSystemDevice.value: PropInfo(UInt32),
    kAudioDevicePropertyLatency.value: PropInfo(UInt32),
    kAudioDevicePropertyStreams.value: PropInfo(Array(Object)),
    kAudioObjectPropertyControlList.value: PropInfo(Array(Object)),
    kAudioDevicePropertySafetyOffset.value: PropInfo(UInt32),
    kAudioDevicePropertyNominalSampleRate.value: PropInfo(Float64),
    kAudioDevicePropertyAvailableNominalSampleRates.value: PropInfo(Array(AudioValueRange)),
    kAudioDevicePropertyIcon.value: PropInfo(CFURL),
    kAudioDevicePropertyIsHidden.value: PropInfo(UInt32),
    kAudioDevicePropertyPreferredChannelsForStereo.value: PropInfo(UInt32*2),
    kAudioDevicePropertyPreferredChannelLayout.value: PropInfo(AudioChannelLayout),
    kAudioClockDeviceClassID.value: PropInfo(AudioClassID),
    kAudioClockDevicePropertyDeviceUID.value: PropInfo(CFString),
    kAudioClockDevicePropertyTransportType.value: PropInfo(UInt32),
    kAudioClockDevicePropertyClockDomain.value: PropInfo(UInt32),
    kAudioClockDevicePropertyDeviceIsAlive.value: PropInfo(UInt32),
    kAudioClockDevicePropertyDeviceIsRunning.value: PropInfo(UInt32),
    kAudioClockDevicePropertyLatency.value: PropInfo(UInt32),
    kAudioClockDevicePropertyControlList.value: PropInfo(Array(Object)),
    kAudioClockDevicePropertyNominalSampleRate.value: PropInfo(Float64),
    kAudioClockDevicePropertyAvailableNominalSampleRates.value: PropInfo(Array(AudioValueRange)),
    kAudioEndPointDevicePropertyComposition.value: PropInfo(CFDictionary),
    kAudioEndPointDevicePropertyEndPointList.value: PropInfo(Array(Object)),
    kAudioEndPointDevicePropertyIsPrivate.value: PropInfo(pid_t),
    kAudioStreamPropertyIsActive.value: PropInfo(UInt32),
    kAudioStreamPropertyDirection.value: PropInfo(UInt32),
    kAudioStreamPropertyTerminalType.value: PropInfo(UInt32),
    kAudioStreamPropertyStartingChannel.value: PropInfo(UInt32),
    kAudioStreamPropertyVirtualFormat.value: PropInfo(AudioStreamBasicDescription),
    kAudioStreamPropertyAvailableVirtualFormats.value: PropInfo(Array(AudioStreamRangedDescription)),
    kAudioStreamPropertyPhysicalFormat.value: PropInfo(AudioStreamBasicDescription),
    kAudioStreamPropertyAvailablePhysicalFormats.value: PropInfo(Array(AudioStreamRangedDescription)),
    kAudioControlPropertyScope.value: PropInfo(AudioServerPlugIn_PropertyScope),
    kAudioControlPropertyElement.value: PropInfo(AudioServerPlugIn_PropertyElement),
    kAudioSliderControlPropertyValue.value: PropInfo(UInt32),
    kAudioSliderControlPropertyRange.value: PropInfo(UInt32*2),
    kAudioLevelControlPropertyScalarValue.value: PropInfo(Float32),
    kAudioLevelControlPropertyDecibelValue.value: PropInfo(Float32),
    kAudioLevelControlPropertyDecibelRange.value: PropInfo(AudioValueRange),
    kAudioLevelControlPropertyConvertScalarToDecibels.value: PropInfo(Float32),
    kAudioLevelControlPropertyConvertDecibelsToScalar.value: PropInfo(Float32),
    kAudioBooleanControlPropertyValue.value: PropInfo(UInt32),
    kAudioSelectorControlPropertyCurrentItem.value: PropInfo(Array(UInt32)),
    kAudioSelectorControlPropertyAvailableItems.value: PropInfo(Array(UInt32)),
    kAudioSelectorControlPropertyItemName.value: PropInfo(CFString, UInt32),
    kAudioSelectorControlPropertyItemKind.value: PropInfo(UInt32, UInt32),
    kAudioStereoPanControlPropertyValue.value: PropInfo(Float32),
    kAudioStereoPanControlPropertyPanningChannels.value: PropInfo(UInt32*2),
    kAudioObjectPropertyCreator.value: PropInfo(CFString),
    kAudioObjectPropertyListenerAdded.value: PropInfo(AudioObjectPropertyAddress),
    kAudioObjectPropertyListenerRemoved.value: PropInfo(AudioObjectPropertyAddress),
    kAudioHardwarePropertyDevices.value: PropInfo(Array(Object)),
    kAudioHardwarePropertyDefaultInputDevice.value: PropInfo(Object),
    kAudioHardwarePropertyDefaultOutputDevice.value: PropInfo(Object),
    kAudioHardwarePropertyDefaultSystemOutputDevice.value: PropInfo(Object),
    kAudioHardwarePropertyTranslateUIDToDevice.value: PropInfo(Object),
    kAudioHardwarePropertyMixStereoToMono.value: PropInfo(UInt32),
    kAudioHardwarePropertyPlugInList.value: PropInfo(Array(Object)),
    kAudioHardwarePropertyTranslateBundleIDToPlugIn.value: PropInfo(Object),
    kAudioHardwarePropertyTransportManagerList.value: PropInfo(Array(Object)),
    kAudioHardwarePropertyTranslateBundleIDToTransportManager.value: PropInfo(Object),
    kAudioHardwarePropertyBoxList.value: PropInfo(Array(Object)),
    kAudioHardwarePropertyTranslateUIDToBox.value: PropInfo(Object),
    kAudioHardwarePropertyClockDeviceList.value: PropInfo(Array(Object)),
    kAudioHardwarePropertyTranslateUIDToClockDevice.value: PropInfo(Object),
    kAudioHardwarePropertyProcessIsMaster.value: PropInfo(UInt32),
    kAudioHardwarePropertyIsInitingOrExiting.value: PropInfo(UInt32),
    kAudioHardwarePropertyUserIDChanged.value: PropInfo(UInt32),
    kAudioHardwarePropertyProcessIsAudible.value: PropInfo(UInt32),
    kAudioHardwarePropertySleepingIsAllowed.value: PropInfo(UInt32),
    kAudioHardwarePropertyUnloadingIsAllowed.value: PropInfo(UInt32),
    kAudioHardwarePropertyHogModeIsAllowed.value: PropInfo(UInt32),
    kAudioHardwarePropertyUserSessionIsActiveOrHeadless.value: PropInfo(UInt32),
    kAudioHardwarePropertyServiceRestarted.value: PropInfo(UInt32),
    kAudioHardwarePropertyPowerHint.value: PropInfo(UInt32),
    kAudioPlugInCreateAggregateDevice.value: PropInfo(Object),
    kAudioPlugInDestroyAggregateDevice.value: PropInfo(Object),
    kAudioTransportManagerCreateEndPointDevice.value: PropInfo(Object),
    kAudioTransportManagerDestroyEndPointDevice.value: PropInfo(Object),
    kAudioDevicePropertyPlugIn.value: PropInfo(OSStatus),
    kAudioDevicePropertyDeviceHasChanged.value: PropInfo(UInt32),
    kAudioDevicePropertyDeviceIsRunningSomewhere.value: PropInfo(UInt32),
    kAudioDeviceProcessorOverload.value: PropInfo(UInt32),
    kAudioDevicePropertyIOStoppedAbnormally.value: PropInfo(UInt32),
    kAudioDevicePropertyHogMode.value: PropInfo(pid_t),
    kAudioDevicePropertyBufferFrameSize.value: PropInfo(UInt32),
    kAudioDevicePropertyBufferFrameSizeRange.value: PropInfo(AudioValueRange),
    kAudioDevicePropertyUsesVariableBufferFrameSizes.value: PropInfo(UInt32),
    kAudioDevicePropertyIOCycleUsage.value: PropInfo(Float32),
    kAudioDevicePropertyStreamConfiguration.value: PropInfo(AudioBufferList),
    kAudioDevicePropertyIOProcStreamUsage.value: PropInfo(AudioHardwareIOProcStreamUsage),
    kAudioDevicePropertyActualSampleRate.value: PropInfo(Float64),
    kAudioDevicePropertyClockDevice.value: PropInfo(CFString),
    kAudioDevicePropertyJackIsConnected.value: PropInfo(UInt32),
    kAudioDevicePropertyVolumeScalar.value: PropInfo(Float32),
    kAudioDevicePropertyVolumeDecibels.value: PropInfo(Float32),
    kAudioDevicePropertyVolumeRangeDecibels.value: PropInfo(AudioValueRange),
    kAudioDevicePropertyVolumeScalarToDecibels.value: PropInfo(Float32),
    kAudioDevicePropertyVolumeDecibelsToScalar.value: PropInfo(Float32),
    kAudioDevicePropertyStereoPan.value: PropInfo(Float32),
    kAudioDevicePropertyStereoPanChannels.value: PropInfo(2*UInt32),
    kAudioDevicePropertyMute.value: PropInfo(UInt32),
    kAudioDevicePropertySolo.value: PropInfo(UInt32),
    kAudioDevicePropertyPhantomPower.value: PropInfo(UInt32),
    kAudioDevicePropertyPhaseInvert.value: PropInfo(UInt32),
    kAudioDevicePropertyClipLight.value: PropInfo(UInt32),
    kAudioDevicePropertyTalkback.value: PropInfo(UInt32),
    kAudioDevicePropertyListenback.value: PropInfo(UInt32),
    kAudioDevicePropertyDataSource.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyDataSources.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyDataSourceNameForIDCFString.value: PropInfo(Translation(UInt32, CFString)),
    kAudioDevicePropertyDataSourceKindForID.value: PropInfo(Translation(UInt32, UInt32)),
    kAudioDevicePropertyClockSource.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyClockSources.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyClockSourceNameForIDCFString.value: PropInfo(Translation(UInt32, CFString)),
    kAudioDevicePropertyClockSourceKindForID.value: PropInfo(Translation(UInt32, UInt32)),
    kAudioDevicePropertyPlayThru.value: PropInfo(UInt32),
    kAudioDevicePropertyPlayThruSolo.value: PropInfo(UInt32),
    kAudioDevicePropertyPlayThruVolumeScalar.value: PropInfo(Float32),
    kAudioDevicePropertyPlayThruVolumeDecibels.value: PropInfo(Float32),
    kAudioDevicePropertyPlayThruVolumeRangeDecibels.value: PropInfo(AudioValueRange),
    kAudioDevicePropertyPlayThruVolumeScalarToDecibels.value: PropInfo(Float32),
    kAudioDevicePropertyPlayThruVolumeDecibelsToScalar.value: PropInfo(Float32),
    kAudioDevicePropertyPlayThruStereoPan.value: PropInfo(Float32),
    kAudioDevicePropertyPlayThruStereoPanChannels.value: PropInfo(UInt32*2),
    kAudioDevicePropertyPlayThruDestination.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyPlayThruDestinations.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyPlayThruDestinationNameForIDCFString.value: PropInfo(Translation(UInt32, CFString)),
    kAudioDevicePropertyChannelNominalLineLevel.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyChannelNominalLineLevels.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyChannelNominalLineLevelNameForIDCFString.value: PropInfo(Translation(UInt32, CFString)),
    kAudioDevicePropertyHighPassFilterSetting.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyHighPassFilterSettings.value: PropInfo(Array(UInt32)),
    kAudioDevicePropertyHighPassFilterSettingNameForIDCFString.value: PropInfo(Translation(UInt32, CFString)),
    kAudioDevicePropertySubVolumeScalar.value: PropInfo(Float32),
    kAudioDevicePropertySubVolumeDecibels.value: PropInfo(Float32),
    kAudioDevicePropertySubVolumeRangeDecibels.value: PropInfo(AudioValueRange),
    kAudioDevicePropertySubVolumeScalarToDecibels.value: PropInfo(Float32),
    kAudioDevicePropertySubVolumeDecibelsToScalar.value: PropInfo(Float32),
    kAudioDevicePropertySubMute.value: PropInfo(UInt32),
    kAudioAggregateDevicePropertyFullSubDeviceList.value: PropInfo(CFArray),
    kAudioAggregateDevicePropertyActiveSubDeviceList.value: PropInfo(Array(Object)),
    kAudioAggregateDevicePropertyComposition.value: PropInfo(CFDictionary),
    kAudioAggregateDevicePropertyMasterSubDevice.value: PropInfo(CFString),
    kAudioAggregateDevicePropertyClockDevice.value: PropInfo(CFString),
    kAudioSubDevicePropertyExtraLatency.value: PropInfo(Float64),
    kAudioSubDevicePropertyDriftCompensation.value: PropInfo(UInt32),
    kAudioSubDevicePropertyDriftCompensationQuality.value: PropInfo(UInt32),
}

audioObjectClassNames = {
    'AudioSystemObject',
    'AudioAggregateDevice',
    'AudioSubDevice',
    'AudioObject',
    'AudioPlugIn',
    'AudioTransportManager',
    'AudioBox',
    'AudioDevice',
    'AudioClockDevice',
    'AudioEndPointDevice',
    'AudioEndPoint',
    'AudioStream',
    'AudioControl',
    'AudioSliderControl',
    'AudioLevelControl',
    'AudioVolumeControl',
    'AudioLFEVolumeControl',
    'AudioBooleanControl',
    'AudioMuteControl',
    'AudioSoloControl',
    'AudioJackControl',
    'AudioLFEMuteControl',
    'AudioPhantomPowerControl',
    'AudioPhaseInvertControl',
    'AudioClipLightControl',
    'AudioTalkbackControl',
    'AudioListenbackControl',
    'AudioSelectorControl',
    'AudioDataSourceControl',
    'AudioDataDestinationControl',
    'AudioClockSourceControl',
    'AudioLineLevelControl',
    'AudioHighPassFilterControl',
    'AudioStereoPanControl',
}

audioObjectClassIDs = {name: globals()['k%sClassID' % name] for name in audioObjectClassNames}
audioObjectClassIDs_rev = {v.value:k for k,v in audioObjectClassIDs.items()}

CoreAudio = CDLL('/System/Library/Frameworks/CoreAudio.framework/CoreAudio')
def _(**kwargs):
    functions = {}
    for k, v in kwargs.items():
        f = functions[k] = getattr(CoreAudio, k)
        f.restype, *f.argtypes = v
    globals().update(functions)
    return functions

audioFunctions = _(
    AudioObjectShow =                        (None, AudioObjectID),
    AudioObjectHasProperty =                 (Boolean,  AudioObjectID, POINTER(AudioObjectPropertyAddress)),
    AudioObjectIsPropertySettable =          (OSStatus, AudioObjectID, POINTER(AudioObjectPropertyAddress), POINTER(Boolean)),
    AudioObjectGetPropertyDataSize =         (OSStatus, AudioObjectID, POINTER(AudioObjectPropertyAddress), UInt32, c_void_p, POINTER(UInt32)),
    AudioObjectGetPropertyData =             (OSStatus, AudioObjectID, POINTER(AudioObjectPropertyAddress), UInt32, c_void_p, POINTER(UInt32), c_void_p),
    AudioObjectSetPropertyData =             (OSStatus, AudioObjectID, POINTER(AudioObjectPropertyAddress), UInt32, c_void_p, UInt32, c_void_p),
    AudioObjectAddPropertyListener =         (OSStatus, AudioObjectID, POINTER(AudioObjectPropertyAddress), AudioObjectPropertyListenerProc, c_void_p),
    AudioObjectRemovePropertyListener =      (OSStatus, AudioObjectID, POINTER(AudioObjectPropertyAddress), AudioObjectPropertyListenerProc, c_void_p),
    AudioHardwareUnload =                    (OSStatus, ),
    AudioHardwareCreateAggregateDevice =     (OSStatus, CFDictionaryRef, POINTER(AudioObjectID)),
    AudioHardwareDestroyAggregateDevice =    (OSStatus, AudioObjectID),
    AudioDeviceCreateIOProcID =              (OSStatus, AudioObjectID, AudioDeviceIOProc, c_void_p, POINTER(AudioDeviceIOProcID)),
    AudioDeviceDestroyIOProcID =             (OSStatus, AudioObjectID, AudioDeviceIOProcID),
    AudioDeviceStart =                       (OSStatus, AudioObjectID, AudioDeviceIOProcID),
    AudioDeviceStartAtTime =                 (OSStatus, AudioObjectID, AudioDeviceIOProcID, POINTER(AudioTimeStamp), UInt32),
    AudioDeviceStop =                        (OSStatus, AudioObjectID, AudioDeviceIOProcID),
    AudioDeviceGetCurrentTime =              (OSStatus, AudioObjectID, POINTER(AudioTimeStamp)),
    AudioDeviceTranslateTime =               (OSStatus, AudioObjectID, POINTER(AudioTimeStamp), POINTER(AudioTimeStamp)),
    AudioDeviceGetNearestStartTime =         (OSStatus, AudioObjectID, POINTER(AudioTimeStamp), UInt32),
)

audioFunctions_rev = {cast(v, c_void_p).value:k for k,v in audioFunctions.items()}

audioErrorNames = {
    'kAudio_UnimplementedError',
    'kAudio_FileNotFoundError',
    'kAudio_FilePermissionError',
    'kAudio_TooManyFilesOpenError',
    'kAudio_BadFilePathError',
    'kAudio_ParamError',
    'kAudio_MemFullError',
    'kAudioHardwareNoError',
    'kAudioHardwareNotRunningError',
    'kAudioHardwareUnspecifiedError',
    'kAudioHardwareUnknownPropertyError',
    'kAudioHardwareBadPropertySizeError',
    'kAudioHardwareIllegalOperationError',
    'kAudioHardwareBadObjectError',
    'kAudioHardwareBadDeviceError',
    'kAudioHardwareBadStreamError',
    'kAudioHardwareUnsupportedOperationError',
    'kAudioDeviceUnsupportedFormatError',
    'kAudioDevicePermissionsError',
}

audioErrors = {name: globals()[name] for name in audioErrorNames}
audioErrors_rev = {v.value:k for k,v in audioErrors.items()}

def trap(fn, *args):
    result = fn(*args)
    if result:
        fn_name = audioFunctions_rev[cast(fn, c_void_p).value]
        error_name = audioErrors_rev[result]
        raise Exception('%s returned %s' % (fn_name, error_name,))

def decode(data, dataType):
    if dataType is Object:
        return AudioObject(AudioObjectID.from_buffer_copy(data))
    elif isinstance(dataType, Array):
        if dataType.of is Object:
            oids = (AudioObjectID * (len(data) // sizeof(AudioObjectID))).from_buffer_copy(data)
            return [AudioObject(oid) for oid in oids]
        return (dataType.of * (len(data) // sizeof(dataType.of))).from_buffer_copy(data)
    elif issubclass(dataType, CFObject):
        return objc.objc.objc_object(c_void_p=c_void_p.from_buffer_copy(data))
    return dataType.from_buffer_copy(data)

def encode(data, dataType): # returns a ctype object and a lifetime object
    if dataType is None:
        return None, None
    elif dataType is Object:
        if isinstance(data, AudioObjectID):
            return data, None
        elif isinstance(data, (int, AudioObjectID)):
            return AudioObjectID(data), None
        elif isinstance(data, AudioObject):
            return AudioObjectID(data.objectID), None
    elif isinstance(dataType, Array):
        if dataType.of is Object:
            if isinstance(data, ctypes_Array):
                if data._type_ is AudioObjectID:
                    return data, None
            elif hasattr(data, '__iter__'):
                oids, lifetimes = zip(*[encode(o, Object) for o in data])
                return (AudioObjectID * len(data))(*oids), lifetimes
            elif data is None:
                return (AudioObjectID * 0)(), None
        elif isinstance(data, ctypes_Array):
            if data._type_ is dataType.of:
                return data, None
        elif hasattr(data, '__iter__'):
            if len(data) == 0:
                return (dataType.of * 0)(), None
            else:
                objs, lifetimes = zip(*[encode(o, dataType.of) for o in data])
                return (dataType.of * len(data))(*objs), lifetimes
        elif data is None:
            return (dataType.of * 0)(), None
    elif issubclass(dataType, CFObject):
        if isinstance(data, dataType):
            return data, data
        elif dataType is CFString and isinstance(data, str):
            o = CoreFoundation.CFStringRef(data)
            return o.__c_void_p__(), o
        elif dataType is CFArray and hasattr(data, '__iter__'):
            o = CoreFoundation.CFArrayRef(data)
            return o.__c_void_p__(), o
        elif dataType is CFDictionary and hasattr(data, '__iter__'):
            o = CoreFoundation.CFDictionaryRef(data)
            return o.__c_void_p__(), o
    elif isinstance(data, dataType):
        return data, None
    else:
        return dataType(getattr(data, 'value', data)), None
    raise TypeError('Expected %s, got %s' % (dataType, type(data)))

class AudioPropertyNotify:
    def __init__(self, objectID, addr, cb):
        self.objectID = objectID
        self.addr = addr
        self.cb = cb
        self.ccb = AudioObjectPropertyListenerProc(cb)
        trap(AudioObjectAddPropertyListener, self.objectID, self.addr, self.ccb, None)
    def __del__(self):
        trap(AudioObjectRemovePropertyListener, self.objectID, self.addr, self.ccb, None)

def classNameForClassID(classID):
    if classID in audioObjectClassIDs_rev:
        return audioObjectClassIDs_rev[classID].lstrip('k').split('ClassID', 1)[0]
    else:
        return "fourcc('%s')" % bytes(UInt32(classID))[::-1].decode()

class AudioObject:
    def __new__(cls, objectID):
        if isinstance(objectID, UInt32):
            objectID = objectID.value
        if objectID == kAudioObjectUnknown.value:
            return None
        return super().__new__(cls)
    def __init__(self, objectID):
        self.objectID = objectID
    def resolveKey(self, key, forGet, forSet):
        scope, el, qual = kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMaster, None
        if not isinstance(key, tuple):  sel = key
        elif len(key) == 1:             sel, = key
        elif len(key) == 2:             sel, scope = key
        elif len(key) == 3:             sel, scope, el = key
        else:                           sel, scope, el, qual = key
        addr = AudioObjectPropertyAddress(sel, scope, el)
        if forGet or forSet:
            if not AudioObjectHasProperty(self.objectID, addr):
                raise KeyError
        if forSet:
            settable = Boolean()
            trap(AudioObjectIsPropertySettable, self.objectID, addr, settable)
            if not settable:
                raise TypeError('Cannot write to %s' % (sel,))
        propInfo = audioPropertyInfo.get(addr.mSelector)
        q, ql, qs = None, None, None
        if propInfo:
            q, ql = encode(qual, propInfo.qualType)
            qs = sizeof(q) if qual is not None else 0
            if q is not None:
                q = byref(q)
        size = None
        if forGet or forSet:
            size = UInt32()
            trap(AudioObjectGetPropertyDataSize, self.objectID, addr, qs, q, size)
        return addr, size, (q, qs, ql), propInfo
    def __getitem__(self, key):
        addr, size, (q, qs, ql), propInfo = self.resolveKey(key, True, False)
        v = (c_uint8 * size.value)()
        trap(AudioObjectGetPropertyData, self.objectID, addr, qs, q, size, v)
        return decode(v, propInfo.valueType)
    def __setitem__(self, key, value):
        addr, size, (q, qs, ql), propInfo = self.resolveKey(key, True, True)
        v, vl = encode(value, propInfo.valueType)
        if size.value != sizeof(v):
            raise ValueError
        trap(AudioObjectSetPropertyData, self.objectID, addr, qs, q, size, byref(v))
    def __contains__(self, key):
        addr, *_ = self.resolveKey(key, False, False)
        return AudioObjectHasProperty(self.objectID, addr)
    def translate(self, key, value):
        addr, size, (q, qs, ql), propInfo = self.resolveKey(key, True, False)
        assert size.value == sizeof(AudioValueTranslation) and isinstance(propInfo.valueType, Translation)
        inputData, inputDataLifetime = encode(value, propInfo.valueType.fromType)
        outputData = propInfo.valueType.toType()
        v = AudioValueTranslation(addressof(inputData), sizeof(inputData), addressof(outputData), sizeof(outputData))
        trap(AudioObjectGetPropertyData, self.objectID, addr, qs, q, size, byref(v))
        return decode(outputData, propInfo.valueType.toType)
    def notify(self, key, cb):
        addr, *_ = self.resolveKey(key, True, False)
        return AudioPropertyNotify(self.objectID, addr, cb)
    @property
    def classID(self):
        return self[kAudioObjectPropertyClass,0,0].value
    @property
    def className(self):
        return classNameForClassID(self.classID)
    @property
    def baseClassID(self):
        return self[kAudioObjectPropertyBaseClass,0,0].value
    @property
    def baseClassName(self):
        return classNameForClassID(self.baseClassID)
    def owner(self):
        return self[kAudioObjectPropertyOwner,0,0]
    def ownedObjects(self, *types):
        return self[kAudioObjectPropertyOwnedObjects,0,0,types]
    def name(self):
        try:
            return self[kAudioObjectPropertyName,0,0]
        except KeyError:
            return None
    def __repr__(self):
        name = self.name()
        oid = getattr(self.objectID, 'value', self.objectID)
        if name is not None:
            return '<%s: %d %s>' % (self.className, oid, repr(name))
        else:
            return '<%s: %d>' % (self.className, oid)
    def __eq__(self, other):
        if not isinstance(other, AudioObject):
            return False
        return self.objectID == other.objectID
    def __hash__(self):
        return hash(self.objectID.value)

AudioSystemObject = AudioObject(kAudioObjectSystemObject)
