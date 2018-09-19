//
//  SoundRecorder.m
//  BlurtDemo
//
//  Created by Peter Iannucci on 11/14/15.
//  Copyright © 2015 MIT. All rights reserved.
//

#import "SoundRecorder.h"
#import <stdlib.h>
#import <CoreFoundation/CoreFoundation.h>
#import <mach/mach_time.h>


void AudioObjectGetProperty(AudioObjectID obj, AudioObjectPropertySelector selector, AudioObjectPropertyScope scope, UInt32 propertySize, void *prop)
{
    AudioObjectPropertyAddress address;
    address.mSelector = selector;
    address.mScope = scope;
    address.mElement = kAudioObjectPropertyElementMaster;
    AudioObjectGetPropertyData(obj, &address, 0, NULL, &propertySize, prop);
}

void AudioObjectSetProperty(AudioObjectID obj, AudioObjectPropertySelector selector, AudioObjectPropertyScope scope, UInt32 propertySize, void *prop)
{
    AudioObjectPropertyAddress address;
    address.mSelector = selector;
    address.mScope = scope;
    address.mElement = 0;
    AudioObjectSetPropertyData(obj, &address, 0, NULL, propertySize, prop);
}

@interface SoundRecorder ()

- (BOOL)recordedFrames:(void*)frames withCount:(size_t)frameCount basicDescription:(AudioStreamBasicDescription)asbd;

@end


OSStatus recordingCallback(AudioDeviceID inDevice, AudioTimeStamp *inNow, AudioBufferList *inInputData, AudioTimeStamp *inInputTime, AudioBufferList *outOutputData, AudioTimeStamp *inOutputTime, void *inClientData)
{
    SoundRecorder *cb = (__bridge SoundRecorder *)inClientData;
    AudioStreamBasicDescription sbd = cb.recordingASBD;
    
    cb.recordingLatency = inNow->mHostTime - inInputTime->mHostTime;
    
    UInt64 startTime = cb.recordingStartHostTime;
    UInt64 inputTimeStart = inInputTime->mHostTime;
    UInt64 framesProvided = 0;
    double ticksPerFrame = 1e9 / (sbd.mSampleRate * cb.nanosecondsPerAbsoluteTick);
    UInt32 bytesPerFrame = sbd.mBytesPerFrame;
    
    cb.recordingStarted = YES;
    for (int i=0; i < inInputData->mNumberBuffers; i++)
    {
        UInt64 inputTime = inputTimeStart + framesProvided * ticksPerFrame;
        size_t frameCount = inInputData->mBuffers[i].mDataByteSize / bytesPerFrame;
        framesProvided += frameCount;
        void *buffer = inInputData->mBuffers[i].mData;
        
        if (inputTime < startTime)
        {
            size_t skipFrames = MIN((int)((startTime - inputTime) / ticksPerFrame), frameCount);
            buffer += bytesPerFrame * skipFrames;
            frameCount -= skipFrames;
        }
        if ([cb recordedFrames:buffer withCount:frameCount basicDescription:sbd])
        {
            [cb stopRecording];
            break;
        }
    }
    
    return 0;
}

@implementation SoundRecorder

- (instancetype) init
{
    if (self = [super init])
    {
        self.inBufSize = 2048;
        
        mach_timebase_info_data_t info;
        mach_timebase_info(&info);
        self.nanosecondsPerAbsoluteTick = (double)info.numer / info.denom;
    }
    return self;
}

- (BOOL)recordedFrames:(void*)frames withCount:(size_t)frameCount basicDescription:(AudioStreamBasicDescription)asbd
{
    return [self.delegate soundRecorder:self recordedFrames:frames withCount:frameCount basicDescription:asbd];
}

- (void) stopRecording
{
    AudioDeviceStop(self.recordingDeviceID, self.recordingIOProcID);
    AudioDeviceDestroyIOProcID(self.recordingDeviceID, self.recordingIOProcID);
}

- (void) startRecordingAtRate:(double)sampleRate
{
    [self startRecordingAtRate:sampleRate time:mach_absolute_time()];
}

- (void) startRecordingAtRate:(double)sampleRate time:(UInt64)startTime
{
    AudioDeviceID inputDeviceID = 0;
    // Get the default sound input device
    AudioObjectGetProperty(kAudioObjectSystemObject, kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal, sizeof(inputDeviceID), &inputDeviceID);
    
    UInt32 inBufSize = self.inBufSize;
    
    AudioObjectSetProperty(inputDeviceID, kAudioDevicePropertyBufferFrameSize, kAudioDevicePropertyScopeInput, sizeof(inBufSize), &inBufSize);
    
    AudioStreamBasicDescription sbd;
    sbd.mSampleRate = sampleRate;
    sbd.mFormatID = kAudioFormatLinearPCM;
    sbd.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
    sbd.mBytesPerPacket = 8;
    sbd.mFramesPerPacket = 1;
    sbd.mBytesPerFrame = 8;
    sbd.mChannelsPerFrame = 2;
    sbd.mBitsPerChannel = 32;
    sbd.mReserved = 0;
    AudioObjectSetProperty(inputDeviceID, kAudioDevicePropertyStreamFormat, kAudioDevicePropertyScopeInput, sizeof(sbd), &sbd);
    
    AudioDeviceIOProcID ioProcID;
    AudioDeviceCreateIOProcID(inputDeviceID, (AudioDeviceIOProc)&recordingCallback, (__bridge void *)self, &ioProcID);
    AudioDeviceStart(inputDeviceID, ioProcID);
    
    self.recordingDeviceID = inputDeviceID;
    self.recordingFs = sampleRate;
    self.recordingIOProcID = ioProcID;
    self.recordingASBD = sbd;
    self.recordingStarted = NO;
    self.recordingStartHostTime = (UInt64)startTime;
}

@end