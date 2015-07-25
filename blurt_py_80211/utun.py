import sys, socket, struct, fcntl, ctypes, os, subprocess, select, binascii, time

libc = ctypes.CDLL('libc.dylib')
errno = ctypes.c_int.in_dll(libc, "errno")

def errorcode():
    os.errno.errorcode[errno.value]

PF_SYSTEM = AF_SYSTEM = 32
AF_SYS_CONTROL = 2
SYSPROTO_CONTROL = 2
UTUN_CONTROL_NAME = "com.apple.net.utun_control"
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
        addr = struct.pack('bbhii5i', 32, AF_SYSTEM, AF_SYS_CONTROL, info.ctl_id, 0, 0,0,0,0,0)
        libc.connect(self.fd.fileno(), addr, len(addr))
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
        v6 = (ord(buf[0])>>4) == 6
        protocol = ctypes.c_uint32(libc.htonl((socket.AF_INET, socket.AF_INET6)[v6]))
        iv = (iovec*2)()
        iv[0] = iovec(ctypes.cast(ctypes.pointer(protocol), ctypes.c_void_p),4)
        iv[1] = iovec(ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p),len(buf))
        count = libc.writev(self.fd.fileno(), iv, 2)
        return max(count-4, 0) if (count>0) else count
    def read(self):
        buf = '\0'*self.mtu
        protocol = ctypes.c_uint32()
        iv = (iovec*2)()
        iv[0] = iovec(ctypes.cast(ctypes.pointer(protocol), ctypes.c_void_p),4)
        iv[1] = iovec(ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p),len(buf))
        count = libc.readv(self.fd.fileno(), iv, 2)
        if count < 0:
            raise OSError(errno.value)
        protocol = libc.ntohl(protocol)
        return buf[:count-4]

node = (sys.argv[1] == '1')

from continuousTransceiver import AsynchronousTransciever, channels

u = utun()
xcvr = AsynchronousTransciever(channels[node], channels[~node])
xcvr.start()

p = select.poll()
p.register(u, select.POLLIN)
p.register(xcvr, select.POLLIN)

while True:
    for fd, event in p.poll():
        if fd == u.fileno() and event & select.POLLIN:
            print 'utun -> audio'
            xcvr.write(u.read())
        elif fd == xcvr.fileno() and event & select.POLLIN:
            print 'audio -> utun'
            u.write(xcvr.read())
