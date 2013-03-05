#!/usr/bin/env python
import numpy as np
import util
from scipy import weave
from scipy.weave import converters

def remainder1(a, b):
    # clobbers input!
    for i in xrange(a.size-b.size+1):
        if a[i]:
            a[i:i+b.size] ^= b
    return a[1-b.size:]

# # remainder1 rewritten in terms of bit arithmetic
# def remainder2(a, b):
#     N = a.size
#     M = b.size-1
#     m = (1<<M)-1
#     L = 8
#     r = sum(a[M-1::-1] << np.arange(M))
#     a = util.shiftin(a[:M-1:-1], L)[::-1]
#     b = sum(b[M:0:-1] << np.arange(M))
#     for aa in a:
#         for l in xrange(L):
#             r = ((r << 1) & m) ^ ((aa >> (L-1-l)) & 1) ^ (b if (r >> (M-1)) & 1 else 0)
#     return (r >> np.arange(M)[::-1]) & 1

# # remainder2 rewritten in terms of a look-up table
# def remainder4(a):
#     if a.size % L:
#         a = np.r_[a[::-1], np.zeros(-a.size%L, int)]
#     else:
#         a = a[::-1]
#     r = 0
#     a = util.shiftin(a, L)[::-1]
#     for i in xrange(0, a.size):
#         r = ((r << L) & m) ^ a[i] ^ _lut[r >> s]
#     return (r >> np.arange(M)[::-1]) & 1

def remainder5(a, no_shift_out=False):
    if a.size % L:
        a = np.r_[a[::-1], np.zeros(-a.size%L, int)]
    else:
        a = a[::-1]
    a = util.shiftin(a, L)[::-1]
    A = a.size
    rf = np.zeros(1, np.uint32)
    code = """
    uint32_t r = 0;
    for (int i=0; i<A; i++)
        r = ((r << L) & m) ^ a(i) ^ _lut(r >> s);
    rf(0) = r;
    """
    weave.inline(code, ['A', 'rf', 'L', 'm', 'a', '_lut', 's'], type_converters=converters.blitz)
    if no_shift_out:
        return rf[0]
    else:
        return (rf[0] >> np.arange(M)[::-1]) & 1

def lut_bootstrap(b, new_L):
    global L, _lut, M, s, M_L, m
    M = b.size-1
    if L is None:
        fn = lambda a: util.shiftin(remainder5(a, b)[::-1], M)[0]
    else:
        fn = lambda a: remainder5(a, True)
    lut = np.empty(1<<new_L, int)
    for i in xrange(lut.size):
        lut[i] = fn(np.r_[util.shiftout(np.r_[i], new_L)[::-1], np.zeros(M, int)])
    _lut = lut
    L = new_L
    s = M-L
    M_L = M/L
    m = (1<<M)-1

def lut_dump(fn):
    import cPickle
    with open(fn, 'wb') as f:
        cPickle.dump((_lut,M,L,s,M_L,m), f)

def lut_load(fn):
    import cPickle
    global _lut, M, L, s, M_L, m
    try:
        with open(fn, 'rb') as f:
            _lut, M, L, s, M_L, m = cPickle.load(f)
        return True
    except:
        return False

def FCS(calculationFields):
    k = calculationFields.size
    return 1 & ~remainder(np.r_[np.ones(32, int), np.zeros(k, int)] ^ np.r_[calculationFields, np.zeros(32, int)], G)

def checkFCS(frame):
    k = frame.size
    x = remainder(np.r_[np.ones(32, int), np.zeros(k, int)] ^ np.r_[frame, np.zeros(32, int)], G)
    return all(x == correct_remainder)

remainder = lambda a, b: remainder5(a)

# The FCS field is transmitted commencing with the coefficient of the highest-order term.
# polynomial from x^32 down to x^0: 0b100000100110000010001110110110111
CRC32_802_11_FCS_G = 0b100000100110000010001110110110111
CRC32_802_11_FCS_remainder = 0b11000111000001001101110101111011
G = util.shiftout(np.r_[CRC32_802_11_FCS_G], 33)[::-1]
correct_remainder = util.shiftout(np.r_[CRC32_802_11_FCS_remainder], 32)[::-1]

L = None
fn = 'crc_lut_16'
if not lut_load(fn):
    lut_bootstrap(G, 8)
    lut_bootstrap(G, 16)
    lut_dump(fn)
