#!/usr/bin/env python3.4
import select
import utun
mtu = 76 # 150
u1 = utun.utun(mtu=mtu)
u2 = utun.utun(mtu=mtu)
u1.ifconfig('inet6', 'fe80::cafe:beef:1')
u2.ifconfig('inet6', 'fe80::cafe:beef:2')
try:
    while True:
        for fd in select.select([u1, u2], [], [], .01)[0]:
            datagram = fd.read()
            print('%d bytes' % len(datagram))
            u1.write(datagram)
            u2.write(datagram)
except KeyboardInterrupt:
    pass
