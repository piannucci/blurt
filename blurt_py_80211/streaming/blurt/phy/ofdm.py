#!/usr/bin/env python
import numpy as np
from . import scrambler

class OFDM:
    @classmethod
    def encodeSymbols(self, symbols, oversample, reps=1, axis=-1):
        nfft = self.nfft
        ncp = self.ncp
        ndim = np.ndim(symbols)
        if axis < 0:
            axis += ndim
        # stuff zeros in the middle of the specified axis to pad it to length nfft * oversample
        front_index = (slice(None),) * axis + (slice(None,nfft//2),) + (slice(None),) * (ndim-axis-1)
        back_index = (slice(None),) * axis + (slice(nfft//2,None),) + (slice(None),) * (ndim-axis-1)
        zero_shape = symbols.shape[:axis] + (nfft*(oversample-1),) + symbols.shape[axis+1:]
        symbols = np.concatenate((symbols[front_index], np.zeros(zero_shape), symbols[back_index]), axis=axis)
        # perform ifft
        symbols = np.fft.ifft(symbols, axis=axis)*oversample
        # perform cyclic extension
        num_tiles = (ncp*reps-1) // nfft + reps + 2
        zero_pos = ((ncp*reps-1) // nfft + 1) * nfft
        start = zero_pos - ncp * reps
        stop = zero_pos + nfft * reps + 1
        tile_shape = (1,) * axis + (num_tiles,) + (1,) * (ndim-axis-1)
        extract_index = (slice(None),) * axis + (slice(start*oversample, stop*oversample),) + (slice(None),) * (ndim-axis-1)
        return np.tile(symbols, tile_shape)[extract_index]

    @classmethod
    def blendSymbols(self, subsequences, oversample):
        output = np.zeros(sum(map(len, subsequences)) - (len(subsequences) - 1) * oversample, complex)
        i = 0
        ramp = np.linspace(0,1,2+oversample)[1:-1]
        for x in subsequences:
            weight = np.ones(x.size)
            weight[-1:-oversample-1:-1] = weight[:oversample] = ramp
            output[i:i+len(x)] += weight * x
            i += len(x) - oversample
        return output


class L(OFDM): # Legacy (802.11a) mode
    nfft = 64
    ncp = 16
    ts_reps = 2
    sts_freq = np.zeros(64, np.complex128)
    sts_freq.put([4, 8, 12, 16, 20, 24, -24, -20, -16, -12, -8, -4],
                 (13./6.)**.5 * (1+1j) * np.array([-1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1]))
    lts_freq = np.array([
        0, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,
        1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,
        1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1])
    dataSubcarriers = np.r_[-26:-21,-20:-7,-6:0,1:7,8:21,22:27]
    pilotSubcarriers = np.array([-21,-7,7,21])
    pilotTemplate = np.array([1,1,1,-1])
    Nsc = dataSubcarriers.size
    Nsc_used = dataSubcarriers.size + pilotSubcarriers.size

    @classmethod
    def encode(self, parts, oversample):
        signal, data = parts
        subcarriers = np.concatenate((signal, data), axis=0)
        pilotPolarity = np.resize(scrambler.pilot_sequence, subcarriers.shape[0])
        symbols = np.zeros((subcarriers.shape[0], self.nfft), complex)
        symbols[:,self.dataSubcarriers] = subcarriers
        symbols[:,self.pilotSubcarriers] = self.pilotTemplate * (1. - 2.*pilotPolarity)[:,None]
        sts_time = self.encodeSymbols(self.sts_freq, oversample, self.ts_reps, -1)
        lts_time = self.encodeSymbols(self.lts_freq, oversample, self.ts_reps, -1)
        symbols  = self.encodeSymbols(symbols, oversample, 1, -1)
        return self.blendSymbols([sts_time, lts_time] + list(symbols), oversample)

class HT20(L): # High Throughput 20 MHz bandwidth
    nfft = 64
    ts_reps = 2
    lts_freq = L.lts_freq.copy()
    lts_freq.put([27, 28, -28, -27], [-1, -1, 1, 1])
    dataSubcarriers = np.r_[-28:-21,-20:-7,-6:0,1:7,8:21,22:29]
    Nsc = dataSubcarriers.size
    Nsc_used = dataSubcarriers.size + L.pilotSubcarriers.size
    pilotTemplates = { # indexed by N_sts, then by sts
        1:np.array([[ 1, 1, 1,-1]]),
        2:np.array([[ 1, 1,-1,-1],
                    [ 1,-1,-1, 1]]),
        3:np.array([[ 1, 1,-1,-1],
                    [ 1,-1, 1,-1],
                    [-1, 1, 1,-1]]),
        4:np.array([[ 1, 1, 1,-1],
                    [ 1, 1,-1, 1],
                    [ 1,-1, 1, 1],
                    [-1, 1, 1, 1]]),
    }

class HT40(OFDM): # High Throughput 40 MHz bandwidth
    nfft = 128
    ts_reps = 2
    sts_freq = np.r_[np.fft.fftshift(L.sts_freq), np.fft.fftshift(L.sts_freq)]
    lts_freq = np.r_[np.fft.fftshift(L.lts_freq), np.fft.fftshift(L.lts_freq)]
    lts_freq.put([-32, -5, -4, -3, -2, 2, 3, 4, 5, 32], [1, -1, -1, -1, 1, -1, 1, 1, -1, 1])
    dataSubcarriers = np.r_[-58:-53,-52:-25,-24:-11,-10:-1,2:11,12:25,26:53,54:59]
    pilotSubcarriers = np.array([-53,-25,-11,11,25,53])
    Nsc = dataSubcarriers.size
    pilotTemplates = { # indexed by N_sts, then by sts
        1:np.array([[ 1, 1, 1,-1,-1, 1]]),
        2:np.array([[ 1, 1,-1,-1,-1,-1],
                    [ 1, 1, 1,-1, 1, 1]]),
        3:np.array([[ 1, 1,-1,-1,-1,-1],
                    [ 1, 1, 1,-1, 1, 1],
                    [ 1,-1, 1,-1,-1, 1]]),
        4:np.array([[ 1, 1,-1,-1,-1,-1],
                    [ 1, 1, 1,-1, 1, 1],
                    [ 1,-1, 1,-1,-1, 1],
                    [-1, 1, 1, 1,-1, 1]]),
    }

class HT20_400ns(HT20):
    ncp = 8

class HT20_800ns(HT20):
    ncp = 16

class HT40_400ns(HT40):
    ncp = 16

class HT40_800ns(HT40):
    ncp = 32
