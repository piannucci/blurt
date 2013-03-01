#!/usr/bin/env python
import numpy as np
import scrambler, util

# short training sequence
sts_freq = (13./6.)**.5 * np.array([
    0, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0,
    1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 1+1j, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1+1j, 0, 0, 0, -1-1j, 0, 0, 0,
    1+1j, 0, 0, 0, -1-1j, 0, 0, 0, -1-1j, 0, 0, 0, 1+1j, 0, 0, 0])
sts = np.fft.ifft(sts_freq)
sts_final = np.tile(sts, 4)[32:][:161]

# long training sequence
# taken from slide 30 of
# http://140.117.160.140/course/pdfdownload/9222/Introduction_to_OFDM.pdf
lts_freq = np.array([
    0, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,
    1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,
    1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1])
lts = np.fft.ifft(lts_freq)
lts_final = np.tile(lts, 4)[32:][:161]

def stitch(*x):
    output = np.zeros(sum(len(xx)-1 for xx in x)+1,np.complex)
    i = 0
    for xx in x:
        yy = xx[:]
        yy[0] *= .5
        yy[-1] *= .5
        output[i:i+len(yy)] += yy
        i += len(yy)-1
    return output

class OFDM:
    def __init__(self):
        self.dataSubcarriers = np.r_[np.arange(6,11), np.arange(12,25), np.arange(26,32), np.arange(33, 39), np.arange(40,53), np.arange(54,59)]
        self.pilotSubcarriers = np.array([32-21, 32-7, 32+7, 32+21])
        self.pilotTemplate = np.array([1,1,1,-1])
        self.Nsc = 48
        self.nfft = 64
        self.ncp = 64
        self.lts_freq = lts_freq
    def pilotPolarity(self):
        return (1 - 2 * x for x in scrambler.scrambler(0x7F))
    def encodeSymbols(self, dataSubcarriers, pilotPolarity):
        Ns = dataSubcarriers.shape[0]
        symbols = np.zeros((Ns,self.nfft), np.complex)
        symbols[:,self.dataSubcarriers] = dataSubcarriers
        symbols[:,self.pilotSubcarriers] = self.pilotTemplate * np.array(list(util.truncate(pilotPolarity, Ns)))[:,np.newaxis]
        return np.tile(np.fft.ifft(np.fft.ifftshift(symbols, axes=(-1,))), (1,3))[:,self.nfft-self.ncp:2*self.nfft+1]
    def encode(self, signal_subcarriers, data_subcarriers):
        pilotPolarity = self.pilotPolarity()
        signal_output = self.encodeSymbols(signal_subcarriers[np.newaxis,:], pilotPolarity)
        data_output = self.encodeSymbols(data_subcarriers, pilotPolarity)
        return stitch(sts_final, lts_final, signal_output[0], *data_output)
