from ctypes import *

Float32 = c_float
Float64 = c_double
UInt64 = c_uint64
SInt64 = c_int64
UInt32 = c_uint32
SInt32 = c_int32
UInt16 = c_uint16
SInt16 = c_int16
Boolean = c_uint8

OSStatus = SInt32

ntohl = CDLL(None).ntohl
def fourcc(s):
    return ntohl(c_uint32.from_buffer_copy(s.encode()))
