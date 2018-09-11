import numpy as np
from . import kernels

def mul(a, b):
    return np.bitwise_xor.reduce(a[:,None,None] * (b[:,None] & (1<<np.arange(7))), -1)

output_map = 1 & (mul(np.array([109, 79]), np.arange(128)) >> 6)
output_map_soft = output_map * 2 - 1

def encode(y):
    return kernels.encode(y, output_map)

def decode(llr):
    N = llr.size//2
    return kernels.decode(N, (llr[:N*2].reshape(-1,2,1)*output_map_soft).sum(1))
