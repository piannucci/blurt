from .MacTypes import *
import time
import numpy as np

class mach_timebase_info_data_t(Structure):
    _fields_ = [
        ('numer', UInt32),
        ('denom', UInt32),
    ]

mach_absolute_time = CDLL(None).mach_absolute_time
mach_absolute_time.restype = UInt64
mach_absolute_time.argtypes = ()

mach_timebase_info = CDLL(None).mach_timebase_info
mach_timebase_info.restype = c_int
mach_timebase_info.argtypes = (POINTER(mach_timebase_info_data_t),)

_info = mach_timebase_info_data_t()
mach_timebase_info(pointer(_info))

assert time.get_clock_info('monotonic').implementation == 'mach_absolute_time()'

# reconstruct value of t0 in
# https://github.com/python/cpython/blob/e42b705188271da108de42b55d9344642170aa2b/Python/pytime.c#L855
def pymonotonic_t0():
    t1 = time.monotonic()
    t2 = mach_absolute_time()
    t3 = time.monotonic()
    return t2 - int((t1+t3)/2 * 1e9 * _info.denom / _info.numer)

pymonotonic_t0 = int(np.array([pymonotonic_t0() for i in range(1000)], np.uint64).sum() / 1000 + .5)

def hostTimeToMonotonic(hostTime):
    return (hostTime - pymonotonic_t0) * _info.numer / _info.denom * 1e-9

def monotonicToHostTime(monotonic):
    return int(monotonic * 1e9 * _info.denom / _info.numer) + pymonotonic_t0

# partial derivatives
d_hostTime_d_monotonic = 1e9 * _info.denom / _info.numer
d_monotonic_d_hostTime = _info.numer / _info.denom * 1e-9
