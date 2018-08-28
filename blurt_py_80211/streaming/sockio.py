import ctypes
import socket

UTUN_CONTROL_NAME = b"com.apple.net.utun_control"
MAX_KCTL_NAME = 96
CTLIOCGINFO = 0xc0000000 | 100 << 16 | ord('N') << 8 | 3
UTUN_OPT_IFNAME = 2

SIOCLL_START = 0xc0806982
SIOCSIFMTU = 0x80206934

IFT_ETHER = 6

class ctl_info(ctypes.Structure):
    _fields_ = [("ctl_id", ctypes.c_uint32),
                ("ctl_name", ctypes.c_char * MAX_KCTL_NAME)]

class sockaddr_dl(ctypes.Structure):
    _fields_ = [
        ('len', ctypes.c_byte),
        ('family', ctypes.c_byte),
        ('index', ctypes.c_ushort),
        ('type', ctypes.c_byte),
        ('nlen', ctypes.c_byte),
        ('alen', ctypes.c_byte),
        ('slen', ctypes.c_byte),
        ('data', ctypes.c_char*12),
    ]

class ifaddrs(ctypes.Structure):
    pass
ifaddrs._fields_ = [
    ('ifa_next', ctypes.POINTER(ifaddrs)),
    ('ifa_name', ctypes.c_char_p),
    ('ifa_flags', ctypes.c_uint),
    ('ifa_addr', ctypes.POINTER(sockaddr_dl)),
    ('ifa_netmask', ctypes.POINTER(sockaddr_dl)),
    ('ifa_dstaddr', ctypes.POINTER(sockaddr_dl)),
    ('ifa_data', ctypes.c_void_p),
]

class in6_addr(ctypes.Structure):
    _fields_ = [('__u6_addr8', ctypes.c_uint8*16)]

class sockaddr_in6(ctypes.Structure):
    _fields_ = [
        ('sin6_len', ctypes.c_uint8),
        ('sin6_family', ctypes.c_uint8),
        ('sin6_port', ctypes.c_uint16),
        ('sin6_flowinfo', ctypes.c_uint32),
        ('sin6_addr', in6_addr),
        ('sin6_scope_id', ctypes.c_uint32),
    ]

class in6_addrlifetime(ctypes.Structure):
    _fields_ = [
        ('ia6t_expire', ctypes.c_uint64),
        ('ia6t_preferred', ctypes.c_uint64),
        ('ia6t_vltime', ctypes.c_uint32),
        ('ia6t_pltime', ctypes.c_uint32),
    ]

class in6_aliasreq(ctypes.Structure):
    _fields_ = [
        ('ifra_name', ctypes.c_char*16),
        ('ifra_addr', sockaddr_in6),
        ('ifra_dstaddr', sockaddr_in6),
        ('ifra_prefixmask', sockaddr_in6),
        ('ifra_flags', ctypes.c_int),
        ('ifra_lifetime', in6_addrlifetime),
    ]

class ifk_data_t(ctypes.Union):
    _fields_ = [
        ('ifk_ptr', ctypes.c_void_p),
        ('ifk_value', ctypes.c_int),
    ]

class ifkpi(ctypes.Structure):
    _fields_ = [
        ('ifk_module_id', ctypes.c_uint),
        ('ifk_type', ctypes.c_uint),
        ('ifk_data', ifk_data_t),
    ]

class sockaddr(ctypes.Structure):
    _fields_ = [
        ('sa_len', ctypes.c_uint8),
        ('sa_family', ctypes.c_uint8),
        ('sa_data', ctypes.c_byte * 14),
    ]

class ifdevmtu(ctypes.Structure):
    _fields_ = [
        ('ifdm_current', ctypes.c_int),
        ('ifdm_min', ctypes.c_int),
        ('ifdm_max', ctypes.c_int),
    ]

class ifr_ifru_t(ctypes.Union):
    _fields_ = [
        ('ifru_addr', sockaddr),
        ('ifru_dstaddr', sockaddr),
        ('ifru_broadaddr', sockaddr),
        ('ifru_flags', ctypes.c_short),
        ('ifru_metric', ctypes.c_int),
        ('ifru_mtu', ctypes.c_int),
        ('ifru_phys', ctypes.c_int),
        ('ifru_media', ctypes.c_int),
        ('ifru_intval', ctypes.c_int),
        ('ifru_data', ctypes.c_void_p),
        ('ifru_devmtu', ifdevmtu),
        ('ifru_kpi', ifkpi),
        ('ifru_wake_flags', ctypes.c_uint32),
        ('ifru_route_refcnt', ctypes.c_uint32),
        ('ifru_cap', ctypes.c_int*2),
        ('ifru_functional_type', ctypes.c_uint32),
    ]

class ifreq(ctypes.Structure):
    _fields_ = [
        ('ifr_name', ctypes.c_char*16),
        ('ifr_ifru', ifr_ifru_t),
    ]

getifaddrs = ctypes.CDLL(None).getifaddrs
getifaddrs.restype = None
getifaddrs.argtypes = [ctypes.POINTER(ctypes.POINTER(ifaddrs))]

freeifaddrs = ctypes.CDLL(None).freeifaddrs
freeifaddrs.restype = None
freeifaddrs.argtypes = [ctypes.POINTER(ifaddrs)]

def getlinkaddrs():
    addrs = ctypes.POINTER(ifaddrs)()
    getifaddrs(ctypes.pointer(addrs))
    ifa = addrs
    result = {}
    while ifa:
        ifname = ifa.contents.ifa_name.decode()
        if bool(ifa.contents.ifa_addr) and ifa.contents.ifa_addr.contents.family == socket.AF_LINK:
            sdl = ifa.contents.ifa_addr
            if sdl.contents.alen and sdl.contents.type == IFT_ETHER:
                ether = (ctypes.c_char * sdl.contents.alen).from_address(ctypes.addressof(sdl.contents) + sockaddr_dl.data.offset + sdl.contents.nlen)[:]
                if ifname not in result:
                    result[ifname] = set()
                result[ifname].add(ether)
        ifa = ifa.contents.ifa_next
    freeifaddrs(addrs)
    return result
