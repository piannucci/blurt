import socket, struct, fcntl, ctypes, os, subprocess, time

libc = ctypes.CDLL('libc.dylib')
errno = ctypes.c_int.in_dll(libc, "errno")

PF_SYSTEM = AF_SYSTEM = 32
AF_SYS_CONTROL = 2
SYSPROTO_CONTROL = 2
UTUN_CONTROL_NAME = b"com.apple.net.utun_control"
MAX_KCTL_NAME = 96
CTLIOCGINFO = 0xC0000000 | 100 << 16 | ord('N') << 8 | 3
UTUN_OPT_IFNAME = 2

class ctl_info(ctypes.Structure):
    _fields_ = [("ctl_id", ctypes.c_uint32),
                ("ctl_name", ctypes.c_char * MAX_KCTL_NAME)]

class iovec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p),
                ("iov_len", ctypes.c_size_t)]

class utun:
    def __init__(self, nonblock=False, cloexec=True, mtu=150):
        self.fd = socket.socket(PF_SYSTEM, socket.SOCK_DGRAM, SYSPROTO_CONTROL)
        info = ctl_info(0, UTUN_CONTROL_NAME)
        fcntl.ioctl(self.fd, CTLIOCGINFO, info)
        self.fd.connect((info.ctl_id, 0))
        self.iface = self.fd.getsockopt(SYSPROTO_CONTROL, UTUN_OPT_IFNAME, 256)[:-1]
        if nonblock:
            fcntl.fcntl(self.fd, fcntl.F_SETFL, os.O_NONBLOCK)
        if cloexec:
            fcntl.fcntl(self.fd, fcntl.F_SETFD, fcntl.FD_CLOEXEC)
        self.mtu = mtu
        # force generation of link-local address and routes
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
        self.fileno = self.fd.fileno
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
