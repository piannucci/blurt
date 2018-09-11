import numpy as np
from .ofdm import L, HT20_400ns, HT20_800ns, HT40_400ns, HT40_800ns

class QAM:
    def __init__(self, Nbpsc):
        self.Nbpsc = Nbpsc
        if Nbpsc == 1:
            self.symbols = np.array([-1,1])
        else:
            n = Nbpsc//2
            grayRevCode = sum(((np.arange(1<<n) >> i) & 1) << (n-1-i) for i in range(n))
            grayRevCode ^= grayRevCode >> 1
            grayRevCode ^= grayRevCode >> 2
            symbols = (2*grayRevCode+1-(1<<n)) * (1.5 / ((1<<Nbpsc) - 1))**.5
            self.symbols = np.tile(symbols, 1<<n) + 1j*np.repeat(symbols, 1<<n)
    def demap(self, y, dispersion):
        n = self.Nbpsc
        squared_distance = np.abs(self.symbols - y.flatten()[:,None])**2
        ll = -np.log(np.pi * dispersion) - squared_distance / dispersion
        ll -= np.logaddexp.reduce(ll, 1)[:,None]
        j = np.arange(1<<n)
        llr = np.zeros((y.size, n), int)
        for i in range(n):
            llr[:,i] = 10 * (np.logaddexp.reduce(ll[:,0 != (j & (1<<i))], 1) - \
                             np.logaddexp.reduce(ll[:,0 == (j & (1<<i))], 1))
        return np.clip(llr, -1e4, 1e4)

class Rate:
    def __init__(self, Nbpscs, ratio, Nss=1, ofdm_format=L):
        self.ofdm_format = ofdm_format
        self.Nbpscs = np.asarray(Nbpscs)
        if np.ndim(self.Nbpscs) == 0:
            self.Nbpscs = np.resize(self.Nbpscs, Nss)
        self.Nbpsc = self.Nbpscs.sum()
        self.constellation = [QAM(n) for n in self.Nbpscs]
        if ofdm_format is not None:
            self.Ncbpss = ofdm_format.Nsc * self.Nbpscs
            self.Ncbps = self.Ncbpss.sum()
            self.Ndbps = self.Ncbps * ratio[0] // ratio[1]
        self.puncturingMatrix = np.bool_({
            (1,2):[1,1],
            (2,3):[1,1,1,0],
            (3,4):[1,1,1,0,0,1],
            (5,6):[1,1,1,0,0,1,1,0,0,1],
            (7,8):[1,1,1,0,1,0,1,0,0,1,1,0,0,1],
        }[ratio])
        self.ratio = ratio
    def depuncture(self, y):
        output_size = (y.size + self.ratio[1]-1) // self.ratio[1] * self.ratio[0] * 2
        output = np.zeros(output_size, y.dtype)
        output[np.resize(self.puncturingMatrix, output.size)] = y
        return output

_l_rate_params = {
    0xb: (1, (1,2)), # BPSK (1/2)
    0xf: (1, (3,4)), # BPSK (3/4)
    0xa: (2, (1,2)), # QPSK (1/2)
    0xe: (2, (3,4)), # QPSK (3/4)
    0x9: (4, (1,2)), # 16-QAM (1/2)
    0xd: (4, (3,4)), # 16-QAM (3/4)
    0x8: (6, (2,3)), # 64-QAM (2/3)
    0xc: (6, (3,4)), # 64-QAM (3/4)
}

_ht_rate_params = (
    (1, (1,2)), # BPSK (1/2)
    (2, (1,2)), # QPSK (1/2)
    (2, (3,4)), # QPSK (3/4)
    (4, (1,2)), # 16-QAM (1/2)
    (4, (3,4)), # 16-QAM (3/4)
    (6, (2,3)), # 64-QAM (2/3)
    (6, (3,4)), # 64-QAM (3/4)
    (6, (5,6)), # 64-QAM (5/6)
)

_vht_rate_params = _ht_rate_params + (
    (8, (3,4)), # 256-QAM (3/4)
    (8, (5,6)), # 256-QAM (5/6)
)

_HT = {
    400: {
        20: HT20_400ns,
        40: HT40_400ns,
#       80: VHT80_400ns,
#       160: VHT160_400ns,
    },
    800: {
        20: HT20_800ns,
        40: HT40_800ns,
#       80: VHT80_800ns,
#       160: VHT160_800ns,
    },
}

def L_rate(encoding):
    if not encoding in _l_rate_params:
        return None
    Nbpscs, ratio = _l_rate_params[encoding]
    return Rate(Nbpscs, ratio, 1, L)

def HT_rate(bw, gi, mcs):
    if not 0 <= mcs < 31:
        return None
    Nss = mcs // 8 + 1
    Nbpscs, ratio = _ht_rate_params[mcs % 8]
    return Rate(Nbpscs, ratio, Nss, _HT[gi][bw])

def VHT_rate(bw, gi, Nss, mcs):
    if not 0 <= mcs < 10:
        return None
    if not 0 < Nss <= 8:
        return None
    Nbpscs, ratio = _vht_rate_params[mcs]
    return Rate(Nbpscs, ratio, Nss, _HT[gi][bw])
