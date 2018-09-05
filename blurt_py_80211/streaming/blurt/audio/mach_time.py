from .MacTypes import *

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

def nanosecondsPerAbsoluteTick():
    info = mach_timebase_info_data_t()
    mach_timebase_info(pointer(info))
    return info.numer / info.denom
