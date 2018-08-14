#!/usr/bin/env python3.4
from ctypes import *

class sockaddr_dl(Structure):
    _fields_ = [
        ('len', c_byte),
        ('family', c_byte),
        ('index', c_ushort),
        ('type', c_byte),
        ('nlen', c_byte),
        ('alen', c_byte),
        ('slen', c_byte),
        ('data', c_char*12),
    ]

class ifaddrs(Structure):
    pass
ifaddrs._fields_ = [
    ('ifa_next', POINTER(ifaddrs)),
    ('ifa_name', c_char_p),
    ('ifa_flags', c_uint),
    ('ifa_addr', POINTER(sockaddr_dl)),
    ('ifa_netmask', POINTER(sockaddr_dl)),
    ('ifa_dstaddr', POINTER(sockaddr_dl)),
    ('ifa_data', c_void_p),
]

getifaddrs = CDLL('libSystem.dylib').getifaddrs
getifaddrs.restype = None
getifaddrs.argtypes = [POINTER(POINTER(ifaddrs))]
freeifaddrs = CDLL('libSystem.dylib').freeifaddrs
freeifaddrs.restype = None
freeifaddrs.argtypes = [POINTER(ifaddrs)]

AF_LINK = 18
IFT_ETHER = 6

def getlinkaddrs():
    addrs = POINTER(ifaddrs)()
    getifaddrs(pointer(addrs))
    ifa = addrs
    result = {}
    while ifa:
        ifname = ifa.contents.ifa_name.decode()
        if bool(ifa.contents.ifa_addr) and ifa.contents.ifa_addr.contents.family == AF_LINK:
            sdl = ifa.contents.ifa_addr
            if sdl.contents.alen and sdl.contents.type == IFT_ETHER:
                ether = (c_char * sdl.contents.alen).from_address(addressof(sdl.contents) + sockaddr_dl.data.offset + sdl.contents.nlen)[:]
                if ifname not in result:
                    result[ifname] = set()
                result[ifname].add(ether)
        ifa = ifa.contents.ifa_next
    freeifaddrs(addrs)
    return result

print(':'.join('%02x' % x for x in next(iter(getlinkaddrs()['en0']))))
