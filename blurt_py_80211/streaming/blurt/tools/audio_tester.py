#!/usr/bin/env python3.7
import time
import typing
import numpy as np
from .graph import Graph
from .graph.tee import LambdaBlock
from .audio import IOSession, MicrophoneAGCAdapter
from .audio import InStream_SourceBlock, OutStream_SinkBlock, IOSession_Block
from .audio import AudioHardware as AH

Fs = 96000
ios = IOSession()
ios.addDefaultInputDevice()
ios.addDefaultOutputDevice()
ios.negotiateFormat(AH.kAudioObjectPropertyScopeOutput, minimumSampleRate=Fs, maximumSampleRate=Fs, outBufSize=256)
ios.negotiateFormat(AH.kAudioObjectPropertyScopeInput, minimumSampleRate=Fs, maximumSampleRate=Fs, inBufSize=256)
ios_b = IOSession_Block(ios)
outputChannels = ios.nChannelsPerFrame(AH.kAudioObjectPropertyScopeOutput)
agc = MicrophoneAGCAdapter()
is_b = InStream_SourceBlock(ios)
loopback_b = LambdaBlock(
    typing.Tuple[np.ndarray, int, int], ('nChannelsPerFrame',),
    None, ('nChannelsPerFrame',),
    lambda item: item[0]
)
os_b = OutStream_SinkBlock()
ios.inStream = agc
agc.stream = is_b
is_b.connect(0, loopback_b, 0)
loopback_b.connect(0, os_b, 0)
ios.outStream = os_b
g = Graph([is_b, ios_b])

g.start()
clearLine = '\r\x1b[2K'
try:
    while True:
        time.sleep(.05)
        vu = int(max(0, 80 + 10*np.log10(agc.vu)))
        bar = [' '] * 100
        bar[:vu] = ['.'] * vu
        bar[80] = '|'
        print(clearLine + ''.join(bar) + ' %3d' % vu, end='')
except KeyboardInterrupt:
    pass
print(clearLine, end='')
g.stop()
