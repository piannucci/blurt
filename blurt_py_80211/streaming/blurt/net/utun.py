import socket
import fcntl
import ctypes, ctypes.util
import os
import subprocess
import time
import binascii
from ..mac import lowpan
from . import sockio

errno = ctypes.c_int.in_dll(ctypes.CDLL(ctypes.util.find_library('c')), "errno")

# note the existence of /System/Library/Extensions/EAP-*.bundle

def linkLocalIPv6(ll_addr):
    iid_chars = binascii.hexlify(lowpan.modEUI64(ll_addr)).decode()
    return 'fe80::' + ':'.join(iid_chars[i:i+4] for i in range(0,16,4))

class utun:
    def __init__(self, nonblock=False, cloexec=True, mtu=150, ll_addr=None):
        self.fd = socket.socket(socket.PF_SYSTEM, socket.SOCK_DGRAM, socket.SYSPROTO_CONTROL)
        info = sockio.ctl_info(0, sockio.UTUN_CONTROL_NAME)
        fcntl.ioctl(self.fd, sockio.CTLIOCGINFO, info)
        self.fd.connect((info.ctl_id, 0))
        self.iface = self.fd.getsockopt(socket.SYSPROTO_CONTROL, sockio.UTUN_OPT_IFNAME, 256)[:-1].decode()
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
            # Doesn't crash, but I haven't confirmed that it does anything.
            with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM, 0) as s:
                aliasreq = sockio.in6_aliasreq(self.iface.encode(), (ctypes.sizeof(sockio.sockaddr_in6), socket.AF_LINK))
                fcntl.ioctl(s, sockio.SIOCLL_START, aliasreq)
                ifreq = sockio.ifreq(self.iface.encode(), sockio.ifr_ifru_t(ifru_mtu=mtu))
                fcntl.ioctl(s, sockio.SIOCSIFMTU, ifreq)
        if ll_addr is not None:
            self.ifconfig('inet6', linkLocalIPv6(ll_addr))
        self.ll_addr = ll_addr
        self.fileno = self.fd.fileno
    def ifconfig(self, *args):
        subprocess.Popen(['ifconfig', self.iface] + list(args)).wait()
    def write(self, buf):
        v6 = (buf[0]>>4) == 6
        protocol = ctypes.c_uint32(socket.htonl((socket.AF_INET, socket.AF_INET6)[v6]))
        count = os.writev(self.fd.fileno(), [protocol, buf])
        return max(count-4, 0) if (count>0) else count
    def read(self):
        buf = bytearray(self.mtu)
        protocol = ctypes.c_uint32()
        count = os.readv(self.fd.fileno(), [protocol, buf])
        if count < 0:
            raise OSError(errno.value)
        protocol = socket.ntohl(protocol.value)
        return bytes(buf[:count-4])
    def __repr__(self):
        return '<utun %s>' % self.iface
