import time
from typing import Tuple, NamedTuple, List, Dict, Optional
import numpy as np
import binascii

class ReassemblyKey(NamedTuple):
    sa: bytes
    da: bytes
    size: int
    tag: int

class ReassemblyFragment(NamedTuple):
    offset: int
    data: bytes

class ReassemblyState(NamedTuple):
    start_time: float
    fragments: List[ReassemblyFragment]

class PDB:
    macaddr: bytes
    reassembly_buffers: Dict[ReassemblyKey, ReassemblyState]
    reassembly_timeout: float
    def purgeOlderThan(self, t):
        rb = self.reassembly_buffers
        for k,s in list(rb.items()):
            if s.start_time <= t:
                del rb[k]
    def consumeFragment(self, k: ReassemblyKey, f: ReassemblyFragment):
        t = time.time()
        self.purgeOlderThan(t - self.reassembly_timeout)
        rb = self.reassembly_buffers
        if k not in rb:
            rb[k] = ReassemblyState(t, [f])
            return
        f_start = f.offset * 8
        f_end = f_start + len(f.data)
        for ff in rb[k].fragments:
            ff_start = ff.offset * 8
            ff_end = ff_start + len(ff.data)
            if ff_start == f_start:
                if ff_end == f_end:
                    return # discard repeated fragment
                else:
                    break # imperfect overlap; purge
            elif ff_end > f_start and ff_start < f_end:
                break # imperfect overlap; purge
        else: # no overlap
            fragments = rb[k].fragments
            fragments.append(f)
            fragments = sorted(fragments)
            last_end = 0
            for offset, data in fragments:
                ff_start = offset * 8
                if ff_start != last_end:
                    break # gap
                last_end = ff_start + len(data)
            else: # no inline gaps
                if last_end == k.size: # no trailing gap
                    del rb[k]
                    return Packet(self, k.sa, k.da, b''.join(data for offset, data in fragments))
            # gap detected
            return
        # imperfect overlap detected
        rb[k] = ReassemblyState(t, [f]) # purge reassembly buffer
    def modEUI64(self, la):
        return bytes([la[0] ^ 0x02]) + la[1:3] + b'\xff\xfe' + la[3:6]

class Packet:
    pdb : PDB
    ll_sa : bytes
    ll_da : bytes
    data : bytes
    i : int # in bits
    def __init__(self, pdb, ll_sa, ll_da, data):
        self.pdb = pdb
        self.ll_sa = ll_sa
        self.ll_da = ll_da
        self.data = data
        self.i = 0
    def readBits(self, N):
        result = 0
        i = self.i
        j = 0
        self.i += N
        if i & 7:
            n = 8 - (i & 7)
            o = self.data[i>>3] & ((1<<n) - 1)
            if n >= N:
                return o >> (n-N)
            result |= o << (N-n)
            i += n
            j += n
        while j+8 <= N:
            result |= self.data[i>>3] << (N-j-8)
            i += 8
            j += 8
        if j < N:
            n = N-j
            o = (self.data[i>>3] >> (8-n)) & ((1<<n) - 1)
            result |= o
        return result
    def readOctets(self, N):
        i = self.i
        assert i & 7 == 0
        self.i += N << 3
        return self.data[i>>3:(i>>3)+N]
    def peekOctet(self):
        i = self.i
        assert i & 7 == 0
        return self.data[i>>3]
    def tail(self):
        i = self.i
        assert i & 7 == 0
        return self.data[i>>3:]

def dispatchIPv6PDU(p: Packet):
    print('Found IPv6 datagram: ' + binascii.hexlify(p.tail()).decode())

def dispatchFragmentedPDU(p: Packet):
    pdb = p.pdb
    dispatch = p.peekOctet()
    if dispatch & 0b11000000 == 0: # Not a LoWPAN frame
        return
    if dispatch == 0b01000001: # uncompressed IPv6 header
        p.readOctets(1)
        return dispatchIPv6PDU(p)
    if dispatch == 0b01000000: # ESC
        return
    if dispatch & 0b11100000 == 0b01100000: # LOWPAN_IPHC
        payload = decompressIPHCPDU(p, uncompressed_size=None)
        return dispatchIPv6PDU(Packet(p.pdb, p.ll_sa, p.ll_da, payload))
    if dispatch & 0b11011000 == 0b11000000: # FRAG1/FRAGN
        subsequent = (dispatch >> 5) & 1
        _, size, tag = p.readBits(5), p.readBits(11), p.readBits(16)
        offset = p.readBits(8) if subsequent else 0
        if subsequent:
            payload = p.tail()
        else:
            dispatch = p.peekOctet()
            if dispatch == 0b01000001: # uncompressed IPv6 header
                p.readOctets(1)
                payload = p.tail()
            elif dispatch == 0b01000010: # LOWPAN_HC1
                payload = decompressHC1PDU(p)
            elif dispatch & 0b11100000 == 0b01100000: # LOWPAN_IPHC
                payload = decompressIPHCPDU(p, uncompressed_size=size)
            else:
                return
        k = ReassemblyKey(p.ll_sa, p.ll_da, size, tag)
        f = ReassemblyFragment(offset, payload)
        p = pdb.consumeFragment(k, f)
        if p is not None:
            dispatchIPv6PDU(p)

def decompressHC1PDU(p: Packet):
    pass
    # def decompressAddress(la, mode, pdu, i):
    #     if (mode & 2) == 0:
    #         prefix, i = pdu[i:i+8], i+8
    #     else:
    #         prefix = b'\xfe\x80\0\0\0\0\0\0'
    #     if (mode & 1) == 0:
    #         interfaceIdentifier, i = pdu[i:i+8], i+8
    #     else:
    #         interfaceIdentifier = bytes([la[0] ^ 0x02]) + la[1:3] + b'\xff\xfe' + la[3:6]
    #     return prefix + interfaceIdentifier, i
    # def gotCompressedPDU(sa, da, pdu):
    #     i = 1
    #     encoding, i = pdu[i], i+1
    #     IPv6_HL, i = pdu[i], i+1
    #     IPv6_SA, i = decompressAddress(sa, (encoding >> 6) & 3, pdu, i)
    #     IPv6_DA, i = decompressAddress(da, (encoding >> 4) & 3, pdu, i)
    #     # from now on, i is in bits, not octets
    #     i *= 8
    #     if encoding & 8 == 0:
    #         IPv6_traffic_class, i = shiftOut(pdu, i, 8)
    #         IPv6_flow_label, i = shiftOut(pdu, i, 20)
    #     else:
    #         IPv6_traffic_class = 0
    #         IPv6_flow_label = 0
    #     next_header_encoding = (encoding & 6) >> 1
    #     if next_header_encoding == 0:   IPv6_next_header, i = shiftOut(pdu, i, 8)
    #     elif next_header_encoding == 1: IPv6_next_header = 0x11 # UDP
    #     elif next_header_encoding == 2: IPv6_next_header = 0x01 # ICMP
    #     elif next_header_encoding == 3: IPv6_next_header = 0x06 # TCP
    #     if encoding & 1:
    #         if IPv6_next_header == 0x11: # UDP
    #             UDP_
    #             IPv6_length = UDP_length = # XXX
    #     else:
    #         IPv6_length = len(pdu) - i
    #     i += -i % 8
    #     i //= 8
    #     IPv6_header = bytes([
    #         0x60 | (IPv6_traffic_class >> 4), (IPv6_traffic_class & 0xf) | (IPv6_flow_label >> 16), (IPv6_flow_label >> 8) & 0xff, IPv6_flow_label & 0xff,
    #         IPv6_length
    #     ])
    # Other non-compressed fields MUST follow the Hop Limit as implied by the "HC1
    # encoding" in the exact same order as shown above (Section 10.1): source address
    # prefix (64 bits) and/or interface identifier (64 bits), destination address
    # prefix (64 bits) and/or interface identifier (64 bits), Traffic Class (8 bits),
    # Flow Label (20 bits) and Next Header (8 bits).  The actual next header (e.g.,
    # UDP, TCP, ICMP, etc) follows the non-compressed fields.

def decompressIPHCPDU(p: Packet, uncompressed_size: Optional[int]):
    read, readOctets = p.readBits, p.readOctets
    _,TF,NH,HLIM = read(3), read(2), read(1), read(2)
    CID,SACSAM,MDACDAM = read(1), read(3), read(4)
    IPv6_version = 6
    if uncompressed_size is not None:
        IPv6_payload_length = uncompressed_size - 40
    else:
        IPv6_payload_length = None # we won't know until we're done decompressing the headers
    SCI, DCI = 0, 0
    if CID == 1: SCI, DCI = read(4), read(4)
    IPv6_ECN = IPv6_DSCP = IPv6_FL = 0
    if TF < 3: IPv6_ECN = read(2)
    if TF & 1 == 0: IPv6_DSCP = read(6)
    if TF == 0: read(4)
    if TF == 1: read(2)
    if TF < 2: IPv6_FL = read(20)
    IPv6_TC = IPv6_ECN | (IPv6_DSCP << 2)
    if NH == 0: IPv6_NH = read(8)
    else:       IPv6_NH = None
    if HLIM == 0: IPv6_HL = read(8)
    if HLIM == 1: IPv6_HL = 1
    if HLIM == 2: IPv6_HL = 64
    if HLIM == 3: IPv6_HL = 255
    if SACSAM & 4:          SA_prefix = bytes(8)
    elif SACSAM == 0:       SA_prefix = readOctets(8)
    else:                   SA_prefix = b'\xfe\x80\0\0\0\0\0\0'
    if SACSAM & 3 == 3:     SA_iid = p.pdb.modEUI64(p.ll_sa)
    elif SACSAM == 4:       SA_iid = bytes(8)
    elif SACSAM & 3 == 2:   SA_iid = b'\0\0\0\ff\fe\0' + readOctets(2)
    else:                   SA_iid = readOctets(8)
    IPv6_SA = SA_prefix + SA_iid
    if SACSAM > 4:
        IPv6_SA = p.pdb.context[SCI].rewriteSA(IPv6_SA)
    if MDACDAM & 7 == 0:
        IPv6_DA = readOctets(16)
    elif MDACDAM == 12:
        IPv6_DA = b'\xff' + readOctets(2) + p.pdb.context[DCI].unicastPrefixBasedMulticast + readOctets(4)
    else:
        if MDACDAM & 0xc == 4:      DA_prefix = bytes(8)
        elif MDACDAM & 0xc == 0:    DA_prefix = b'\xfe\x80\0\0\0\0\0\0'
        elif 8 < MDACDAM < 11:      DA_prefix = b'\xff' + readOctets(1) + b'\0\0\0\0\0\0'
        else:                       DA_prefix = b'\xff\x02\0\0\0\0\0\0'
        if MDACDAM & 11 == 1:       DA_iid = readOctets(8)
        elif MDACDAM & 11 == 2:     DA_iid = b'\0\0\0\xff\xfe\0' + readOctets(2)
        elif MDACDAM & 11 == 3:     DA_iid = p.pdb.modEUI64(p.ll_da)
        elif MDACDAM == 9:          DA_iid = bytes(3) + readOctets(5)
        elif MDACDAM == 10:         DA_iid = bytes(5) + readOctets(3)
        elif MDACDAM == 11:         DA_iid = bytes(7) + readOctets(1)
        IPv6_DA = DA_prefix + DA_iid
        if MDACDAM & 0xc == 4:
            IPv6_DA = p.pdb.context[DCI].rewriteDA(IPv6_DA)
    if NH == 1: # LOWPAN_NHC
        p.ip_sa = IPv6_SA
        p.ip_da = IPv6_DA
        IPv6_NH, payload = decompressNHCPDU(p, IPv6_payload_length)
        IPv6_payload_length = len(payload)
    else:
        payload = p.tail()
        if IPv6_payload_length is None: # iff not fragmented
            IPv6_payload_length = len(payload)
    IPv6_header = bytes([
        (IPv6_version << 4) | (IPv6_TC >> 4), ((IPv6_TC & 0xf) << 4) | (IPv6_FL >> 16), (IPv6_FL >> 8) & 0xff, IPv6_FL & 0xff,
        (IPv6_payload_length >> 8) & 0xff, IPv6_payload_length & 0xff, IPv6_NH, IPv6_HL
    ]) + IPv6_SA + IPv6_DA
    return IPv6_header + payload

def decompressNHCPDU(p: Packet, uncompressed_size: Optional[int]):
    read, readOctets = p.readBits, p.readOctets
    encoding = read(8)
    if encoding & 0b11111000 == 0b10110000:
        pass # Extension header GHC                [RFC7400]
    if encoding & 0b11111000 == 0b11010000:
        pass # UDP GHC                             [RFC7400]
    if encoding & 0b11111111 == 0b11011111:
        pass # ICMPv6 GHC                          [RFC7400]
    if encoding & 0x11110000 == 0b11100000: # IPv6 extension header
        EID = (encoding >> 1) & 7
        if EID == 7: # IPHC
            this_header = 41
            if uncompressed_size is not None:
                uncompressed_size -= 1
            return this_header, decompressIPHCPDU(p, uncompressed_size)
        NH = encoding & 1
        ext_NH = read(8) if NH == 0 else None # LOWPAN_NHC
        ext_length_octets = read(8)
        ext_length = (ext_length_octets + 7) // 8
        ext_tail = readOctets(ext_length_octets)
        if EID == 0: # Hop-by-Hop Options Header      [RFC6282][RFC2460]
            this_header = 0
            # re-introduce trailing PadN option if needed
        if EID == 1: # Routing Header                 [RFC6282][RFC2460]
            this_header = 43
        if EID == 2: # Fragment Header                [RFC6282][RFC2460]
            this_header = 44
            ext_length = 0
            assert ext_length_octets == 6
        if EID == 3: # Destinations Options Header    [RFC6282][RFC2460]
            this_header = 60
            # re-introduce trailing PadN option if needed
        if EID == 4: # Mobility Header                [RFC6282][RFC6275]
            this_header = 135
        if NH == 1:
            if uncompressed_size is not None:
                uncompressed_size -= 2 + len(ext_tail)
            ext_NH, payload = decompressNHCPDU(p, uncompressed_size)
        else:
            payload = p.tail()
        ext_header = bytes([ext_NH, ext_length]) + ext_tail
        return this_header, ext_header + payload
    if encoding & 0b11111000 == 0b11110000: # 11110CPP  UDP Header  [RFC6282]
        this_header = 17
        P = encoding & 3
        C = (encoding >> 2) & 1
        if P == 3:
            UDP_SP = 0xf0b0 | read(4)
            UDP_DP = 0xf0b0 | read(4)
        else:
            UDP_SP = read(16) if P & 2 == 0 else 0xf000 | read(8)
            UDP_DP = read(16) if P & 1 == 0 else 0xf000 | read(8)
        payload = p.tail()
        UDP_length = uncompressed_size if uncompressed_size is not None else len(payload)
        if C == 0:
            UDP_checksum = read(16)
        else:
            pseudo_header = p.ip_sa + p.ip_da + bytes([
                0, 0, (UDP_length >> 8) & 0xff, UDP_length & 0xff,
                0, 0, 0, 17,
                (UDP_SP >> 8) & 0xff, UDP_SP & 0xff, (UDP_DP >> 8) & 0xff, UDP_DP & 0xff,
                (UDP_length >> 8) & 0xff, UDP_length & 0xff, 0, 0
            ]) + payload
            if len(pseudo_header) % 2:
                pseudo_header += b'\0'
            UDP_checksum = np.fromstring(pseudo_header, dtype=np.uint16).sum(dtype=np.uint32)
            UDP_checksum %= 0xffff
            UDP_checksum ^= 0xffff
        UDP_header = bytes([(UDP_SP >> 8) & 0xff, UDP_SP & 0xff, (UDP_DP >> 8) & 0xff, UDP_DP & 0xff,
                            (UDP_length >> 8) & 0xff, UDP_length & 0xff, (UDP_checksum >> 8) & 0xff, UDP_checksum & 0xff])
        return this_header, UDP_header + payload

    # Any header that cannot fit within the first fragment MUST NOT be compressed.
    # 
    # LOWPAN_NHC MUST NOT be used to encode IPv6 Extension Headers that have more than
    # 255 octets following the Length field after compression.
    # 
    # Don't use 0xf0bX ports without an integrity check b/c they're overloaded.
    # 
    # The Length field contained in a compressed IPv6 Extension Header indicates the
    # number of octets that pertain to the (compressed) extension header following the
    # Length field.  Note that this changes the Length field definition in [RFC2460]
    # from indicating the header size in 8-octet units, not including the first 8
    # octets.  Changing the Length field to be in units of octets removes wasteful
    # internal fragmentation.
    # 
    # IPv6 Hop-by-Hop and Destination Options Headers may use a trailing Pad1 or PadN
    # to achieve 8-octet alignment.  When there is a single trailing Pad1 or PadN
    # option of 7 octets or less and the containing header is a multiple of 8 octets,
    # the trailing Pad1 or PadN option MAY be elided by the compressor.  A
    # decompressor MUST ensure that the containing header is padded out to a multiple
    # of 8 octets in length, using a Pad1 or PadN option if necessary.  Note that Pad1
    # and PadN options that appear in locations other than the end MUST be carried
    # in-line as they are used to align subsequent options.


