#!/usr/bin/env python3.7
import select
import binascii
import utun
import lowpan

u1 = utun.utun(mtu=1280)
u2 = utun.utun(mtu=1280)
u1.ifconfig('inet6', 'fe80::cafe:beef:1')
u2.ifconfig('inet6', 'fe80::cafe:beef:2')

class TesterPDB(lowpan.PDB):
    def dispatchIPv6PDU(self, p: lowpan.Packet):
        self.dispatched_pdu = p.tail()

pdb = TesterPDB()
pdb.ll_mtu = 76 - 12
ll_sa = binascii.unhexlify('8c8590843fcc')
ll_da = binascii.unhexlify('000000000000')

try:
    while True:
        for fd in select.select([u1, u2], [], [], .01)[0]:
            datagram = fd.read()
            fragments = pdb.compressIPv6Datagram(lowpan.Packet(ll_sa, ll_da, datagram))
            for f in fragments:
                pdb.dispatchFragmentedPDU(lowpan.Packet(ll_sa, ll_da, f))
            if not hasattr(pdb, 'dispatched_pdu'):
                round_trip_successful = False
            else:
                round_trip_successful = pdb.dispatched_pdu == datagram
            print('%4d bytes %s' % (len(datagram), round_trip_successful))
            if not round_trip_successful:
                import pdb
                pdb.set_trace()
            u1.write(datagram)
            u2.write(datagram)
except KeyboardInterrupt:
    pass
