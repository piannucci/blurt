import socket, struct, fcntl, ctypes, os, subprocess, time

libc = ctypes.CDLL('libc.dylib')
errno = ctypes.c_int.in_dll(libc, "errno")

PF_SYSTEM = 32
AF_SYS_CONTROL = 2
SYSPROTO_CONTROL = 2
UTUN_CONTROL_NAME = b"com.apple.net.utun_control"
MAX_KCTL_NAME = 96
CTLIOCGINFO = 0xc0000000 | 100 << 16 | ord('N') << 8 | 3
UTUN_OPT_IFNAME = 2

class ctl_info(ctypes.Structure):
    _fields_ = [("ctl_id", ctypes.c_uint32),
                ("ctl_name", ctypes.c_char * MAX_KCTL_NAME)]

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

# note the existence of /System/Library/Extensions/EAP-*.bundle

class utun:
    def __init__(self, nonblock=False, cloexec=True, mtu=150):
        self.fd = socket.socket(socket.PF_SYSTEM, socket.SOCK_DGRAM, SYSPROTO_CONTROL)
        info = ctl_info(0, UTUN_CONTROL_NAME)
        fcntl.ioctl(self.fd, CTLIOCGINFO, info)
        self.fd.connect((info.ctl_id, 0))
        self.iface = self.fd.getsockopt(SYSPROTO_CONTROL, UTUN_OPT_IFNAME, 256)[:-1].decode()
        if nonblock:
            fcntl.fcntl(self.fd, fcntl.F_SETFL, os.O_NONBLOCK)
        if cloexec:
            fcntl.fcntl(self.fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
        self.mtu = mtu
        # From ifconfig.8:
        ## Basic IPv6 node operation requires a link-local address on each
        ## interface configured for IPv6.  Normally, such an address is
        ## automatically configured by the kernel on each interface added to
        ## the system; this behaviour may be disabled by setting the sysctl MIB
        ## variable net.inet6.ip6.auto_linklocal to 0.
        ## If you delete such an address using ifconfig, the kernel may act very odd.  Do this at your own risk.
        # force generation of link-local address and routes
        if 0:
            subprocess.Popen(['ifconfig', self.iface, 'inet6', 'fe80::1/64', 'mtu', str(self.mtu)]).wait()
            subprocess.Popen(['ifconfig', self.iface, 'inet6', 'delete', 'fe80::1']).wait()
            time.sleep(.5)
            subprocess.Popen(['route', '-q', 'delete', '-inet6', '-net', 'fe80::%%%s/64' % self.iface]).wait()
            subprocess.Popen(['route', '-q', 'add', '-inet6', '-net', 'fe80::%%%s/64' % self.iface, '-interface', self.iface]).wait()
            time.sleep(.5)
            subprocess.Popen(['ifconfig', self.iface, 'inet6', 'fe80::1/64', 'mtu', str(self.mtu)]).wait()
            subprocess.Popen(['ifconfig', self.iface, 'inet6', 'delete', 'fe80::1']).wait()
            time.sleep(.5)
            subprocess.Popen(['route', '-q', 'delete', '-inet6', '-net', 'fe80::%%%s/64' % self.iface]).wait()
            subprocess.Popen(['route', '-q', 'add', '-inet6', '-net', 'fe80::%%%s/64' % self.iface, '-interface', self.iface]).wait()
            time.sleep(.5)
        else:
            # Okay, it seems like we want to invoke in6_ifattach_aliasreq in
            # the kernel.  We can do that via an undocumented ioctl (which is
            # used by pppd).  This results in the following control flow:
            # ioctl
            # ------ kernel ------
            # ioctl
            # fo_ioctl
            # fp->f_ops->fo_ioctl (soo_ioctl)
            # soioctl
            # ifioctllocked
            # ifioctl
            # so_proto->pr_usrreqs->pru_control (in6_control)
            # in6ctl_llstart
            # assuming (ifra->ifra_addr.sin6_family != AF_INET6),
            # in6_ifattach_aliasreq(ifp, NULL, NULL)
            # in6_if_up_dad_start(ifp);
            #     (mucks around with prefixes via in6_ifattach_linklocal?)
            #
            # Doesn't crash, but haven't confirmed that it does anything.
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM, 0) as s:
                SIOCLL_START = 0xc0806982
                fcntl.ioctl(s, SIOCLL_START, in6_aliasreq(self.iface.encode(), (ctypes.sizeof(sockaddr_in6), socket.AF_LINK)))
                SIOCSIFMTU = 0x80206934
                fcntl.ioctl(s, SIOCSIFMTU, ifreq(self.iface.encode(), ifr_ifru_t(ifru_mtu=mtu)))
        self.fileno = self.fd.fileno
    def ifconfig(self, *args):
        subprocess.Popen(['ifconfig', self.iface] + list(args)).wait()
    def write(self, buf):
        v6 = (buf[0]>>4) == 6
        protocol = ctypes.c_uint32(libc.htonl((socket.AF_INET, socket.AF_INET6)[v6]))
        count = os.writev(self.fd.fileno(), [protocol, buf])
        return max(count-4, 0) if (count>0) else count
    def read(self):
        buf = bytearray(self.mtu)
        protocol = ctypes.c_uint32()
        count = os.readv(self.fd.fileno(), [protocol, buf])
        if count < 0:
            raise OSError(errno.value)
        protocol = libc.ntohl(protocol)
        return bytes(buf[:count-4])
