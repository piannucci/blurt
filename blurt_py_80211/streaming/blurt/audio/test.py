import sys
import numpy as np
import pylab as pl
from .coreaudio import *

inArray = np.empty((4*96000, 4), np.float32)
outArray = np.sin(np.arange(4*96000) * 2 * np.pi / 96000 * 440)

io = IOSession()
io.addDefaultInputDevice()
io.addDefaultOutputDevice()

outStream = OutArrayStream(outArray)
inStream = InArrayStream(inArray)
if 1: # AGC experiment
    inStream = AGCInStreamAdapter(inStream, findInputLevelControl(findMicrophone()))

io.start(inStream=inStream, outStream=outStream)
io.wait()

x = inArray
pl.close()
fig, ax = pl.subplots(3,1,True,True)
for i in range(3):
    pl.sca(ax[i])
    pl.specgram(x[:,i], Fs=96000, NFFT=2048, noverlap=2048-256)
print(np.cov(x.T) / (np.diag(np.cov(x.T))[:,None] * np.diag(np.cov(x.T))[None,:])**.5)
