#!/usr/bin/env python3.7
import select
import binascii
import utun
import lowpan
import hashlib

u1 = utun.utun(mtu=1280)
u2 = utun.utun(mtu=1280)
ll_sa = binascii.unhexlify('0200c0f000d1')
ll_da = binascii.unhexlify('0200c0f000d2')
u1.ifconfig('inet6', lowpan.linkLocalIPv6(ll_sa))
u2.ifconfig('inet6', lowpan.linkLocalIPv6(ll_da))

class TesterPDB(lowpan.PDB):
    def dispatchIPv6PDU(self, p: lowpan.Packet):
        datagram = p.tail()
        print('%4d bytes <- tunnel, SHA1 %s' % (len(datagram), hashlib.sha1(datagram).hexdigest()))
        u1.write(datagram)
        u2.write(datagram)

pdb = TesterPDB()
pdb.ll_mtu = 76 - 12

try:
    while True:
        for fd in select.select([u1, u2], [], [], .01)[0]:
            datagram = fd.read()
            print('%4d bytes -> tunnel, SHA1 %s' % (len(datagram), hashlib.sha1(datagram).hexdigest()))
            fragments = pdb.compressIPv6Datagram(lowpan.Packet(ll_sa, ll_da, datagram))
            for f in fragments:
                pdb.dispatchFragmentedPDU(lowpan.Packet(ll_sa, ll_da, f))
except KeyboardInterrupt:
    pass
