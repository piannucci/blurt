#!/usr/bin/env python
import numpy as np

def interleave(input, Ncbps, Nbpsc, HT=False, bandwidth_40=False, reverse=False, i_SS=1):
    s = max(Nbpsc//2, 1)
    Nrow = Ncbps//16 if not HT else [4 * Nbpsc, 6 * Nbpsc][bandwidth_40]
    Ncol = 16 if not HT else [13, 18][bandwidth_40]
    Nrot = 0 if not HT else [11, 29][bandwidth_40]
    if reverse:
        k = np.arange(Ncbps)
        i = Nrow * (k % Ncol) + (k//Ncol)
        j = s * (i//s) + (i + Ncbps - (Ncol * i // Ncbps)) % s
        r = (j - (((i_SS-1) * 2) % 3 + 3 * ((i_SS-1) // 3))*Nrot*Nbpsc) % Ncbps
        p = r
    else:
        r = np.arange(Ncbps)
        j = (r + (((i_SS-1) * 2) % 3 + 3 * ((i_SS-1) // 3))*Nrot*Nbpsc) % Ncbps
        i = s * (j//s) + (j + (Ncol*j//Ncbps)) % s
        k = Ncol * i - (Ncbps - 1) * (i // Nrow)
        p = k
    return input[:input.size//Ncbps*Ncbps].reshape(-1,Ncbps)[:,p].flatten()
