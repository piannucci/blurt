//
//  SoundRecorder.h
//  BlurtDemo
//
//  Created by Peter Iannucci on 11/14/15.
//  Copyright © 2015 MIT. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <AudioToolbox/AudioToolbox.h>

@class SoundRecorder;

@protocol SoundRecorderDelegate

- (BOOL)soundRecorder:(SoundRecorder *)soundRecorder recordedFrames:(void*)frames withCount:(size_t)frameCount basicDescription:(AudioStreamBasicDescription)asbd;

@end

@interface SoundRecorder : NSObject

- (void)stopRecording;
- (void)startRecordingAtRate:(double)sampleRate;
- (void)startRecordingAtRate:(double)sampleRate time:(UInt64)startTime;

@property AudioStreamBasicDescription recordingASBD;
@property UInt64 recordingLatency;
@property UInt64 recordingStartHostTime;
@property double nanosecondsPerAbsoluteTick;
@property BOOL recordingStarted;
@property AudioDeviceIOProcID recordingIOProcID;
@property AudioObjectID recordingDeviceID;
@property int inBufSize;
@property double recordingFs;

@property (weak) id<SoundRecorderDelegate> delegate;

@end
