import numpy as np
from . import kernels

############################ CC ############################

def mul(a, b):
    return np.bitwise_xor.reduce(a[:,None,None] * (b[:,None] & (1<<np.arange(7))), -1)

output_map = 1 & (mul(np.array([109, 79]), np.arange(128)) >> 6)
output_map_soft = output_map * 2 - 1

def encode(y):
    return kernels.encode(y, output_map)

def decode(llr):
    N = llr.size//2
    return kernels.decode(N, (llr[:N*2].reshape(-1,2,1)*output_map_soft).sum(1))

############################ Rates ############################

class Rate:
    def __init__(self, Nbpsc, ratio):
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

l_rates = {
    0xb: Rate(1, (1,2)),
    0xf: Rate(1, (3,4)),
    0xa: Rate(2, (1,2)),
    0xe: Rate(2, (3,4)),
    0x9: Rate(4, (1,2)),
    0xd: Rate(4, (3,4)),
    0x8: Rate(6, (2,3)),
    0xc: Rate(6, (3,4)),
}

ht_rates = {
    0: Rate(1, (1,2)),
    1: Rate(2, (1,2)),
    2: Rate(2, (3,4)),
    3: Rate(4, (1,2)),
    4: Rate(4, (3,4)),
    5: Rate(6, (2,3)),
    6: Rate(6, (3,4)),
    7: Rate(6, (5,6)),
}
