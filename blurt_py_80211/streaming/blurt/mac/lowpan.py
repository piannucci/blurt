import time
from typing import Tuple, NamedTuple, List, Dict, Optional
import numpy as np
import binascii

link_local_prefix = b'\xfe\x80\0\0\0\0\0\0'

def matches(a: bytes, mask: bytes, value: bytes):
    return all(aa&mm==vv for aa,mm,vv in zip(a,mask,value))

def modEUI64(la):
    return bytes([la[0] ^ 0x02]) + la[1:3] + b'\xff\xfe' + la[3:6]

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

class DummyContext:
    def __init__(self):
        self.unicastPrefixBasedMulticast = bytes(10)
    def rewriteSA(self, IPv6_SA):
        return IPv6_SA
    def rewriteDA(self, IPv6_DA):
        return IPv6_DA

class Packet:
    ll_sa : bytes
    ll_da : bytes
    data : bytes
    i : int # in bits
    def __init__(self, ll_sa, ll_da, data):
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
    def peekOctets(self, N):
        i = self.i
        assert i & 7 == 0
        return self.data[i>>3:(i>>3)+N]
    def tail(self):
        i = self.i
        assert i & 7 == 0
        return self.data[i>>3:]
    def __len__(self):
        return len(self.data)

class PDB:
    macaddr: bytes
    reassembly_buffers: Dict[ReassemblyKey, ReassemblyState]
    reassembly_timeout: float

    def __init__(self):
        self.next_tag = 0
        self.context = [DummyContext() for i in range(16)]
        self.reassembly_timeout = 60
        self.reassembly_buffers = {}

    def purgeOlderThan(self, t):
        rb = self.reassembly_buffers
        for k,s in list(rb.items()):
            if s.start_time <= t:
                del rb[k]

    def dispatchFragment(self, k: ReassemblyKey, f: ReassemblyFragment):
        t = time.monotonic()
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
                    return self.dispatchIPv6PDU(Packet(k.sa, k.da, b''.join(data for offset, data in fragments)))
            # gap detected
            return
        # imperfect overlap detected
        rb[k] = ReassemblyState(t, [f]) # purge reassembly buffer

    def nextFragmentTag(self):
        self.next_tag += 1
        return (self.next_tag - 1) & 0xffff

    def dispatchIPv6PDU(self, p: Packet):
        print('Found IPv6 datagram: ' + binascii.hexlify(p.tail()).decode())

    def compressIPv6Datagram(self, p: Packet):
        read, readOctets = p.readBits, p.readOctets
        header_stack = [] # contains tuples (encoding for NH=0, encoding for NH=1, uncompressed encoding, length of uncompressed data)

        IPv6_NH = 41
        while True:
            if IPv6_NH == 41:
                uncompressed = p.peekOctets(40)
                # IPv6 header
                IPv6_version, IPv6_DSCP, IPv6_ECN, IPv6_FL = read(4), read(6), read(2), read(20)
                IPv6_payload_length, IPv6_NH, IPv6_HL = read(16), read(8), read(8)
                IPv6_SA, IPv6_DA = readOctets(16), readOctets(16)
                assert IPv6_version == 6
                if IPv6_DSCP == 0 and IPv6_ECN == 0 and IPv6_FL == 0:
                    TF = 3
                    TF_inline = bytes()
                elif IPv6_FL == 0:
                    TF = 2
                    TF_inline = bytes([(IPv6_ECN << 6) | IPv6_DSCP])
                elif IPv6_DSCP == 0:
                    TF = 1
                    TF_inline = bytes([(IPv6_ECN << 6) | ((IPv6_FL >> 16) & 0xf), (IPv6_FL >> 8) & 0xff, IPv6_FL & 0xff])
                else:
                    TF = 0
                    TF_inline = bytes([(IPv6_ECN << 6) | IPv6_DSCP, (IPv6_FL >> 16) & 0xf, (IPv6_FL >> 8) & 0xff, IPv6_FL & 0xff])
                HLIM_inline = bytes()
                if IPv6_HL == 1:
                    HLIM = 1
                elif IPv6_HL == 64:
                    HLIM = 2
                elif IPv6_HL == 255:
                    HLIM = 3
                else:
                    HLIM = 0
                    HLIM_inline = bytes([IPv6_HL])
                SACSAM, SCI, SA_inline = self.compressIPv6SA(IPv6_SA, p)
                MDACDAM, DCI, DA_inline = self.compressIPv6DA(IPv6_DA, p)
                if SCI > 0 or DCI > 0:
                    CID = 1
                    CID_inline = bytes([(SCI << 4) | DCI])
                else:
                    CID = 0
                    CID_inline = bytes()
                if header_stack:
                    # include an LOWPAN_NHC Encoding if this is not the first IPv6 header
                    NHC_encoding = bytes([0b11101110])
                    uncompressed_dispatch = bytes()
                else:
                    # include a LOWPAN_IPHC Dispatch if this is the first header
                    NHC_encoding = bytes()
                    uncompressed_dispatch = bytes([0b01000001])
                # compute two versions; we don't know yet which one we need
                LOWPAN_IPHC_NH_0 = NHC_encoding + bytes([
                    0b01100000 | (TF << 3) | (0 << 2) | HLIM,
                    (CID << 7) | (SACSAM << 4) | MDACDAM
                ]) + CID_inline + TF_inline + bytes([IPv6_NH]) + HLIM_inline + SA_inline + DA_inline
                LOWPAN_IPHC_NH_1 = NHC_encoding + bytes([
                    0b01100000 | (TF << 3) | (1 << 2) | HLIM,
                    (CID << 7) | (SACSAM << 4) | MDACDAM
                ]) + CID_inline + TF_inline + bytes() + HLIM_inline + SA_inline + DA_inline
                # choice depends on whether the *next* header is compressed, which
                # depends on whether it will fit, which depends on its compressed size...
                header_stack.append((LOWPAN_IPHC_NH_0, LOWPAN_IPHC_NH_1, uncompressed_dispatch + uncompressed, 40))
            elif IPv6_NH in (0, 43, 44, 60, 135):
                ext_NH, ext_length = p.peekOctets(2)
                if IPv6_NH == 0: # hop-by-hop options
                    EID = 0
                elif IPv6_NH == 43: # routing header
                    EID = 1
                elif IPv6_NH == 44: # fragment header
                    EID = 2
                    ext_length = 1
                elif IPv6_NH == 60: # destination options
                    EID = 3
                elif IPv6_NH == 135: # mobility header
                    EID = 4
                uncompressed = readOctets((ext_length+1)*8)
                # I think the only compression we do here is removing trailing pad
                # for EID 0 and 3
                compressed = uncompressed[2:]
                if EID in (0, 3):
                    # A single trailing Pad1 or PadN option may be elided.
                    j = 0
                    while j < len(compressed):
                        tag = compressed[j]
                        if tag == 0:
                            k = j + 1
                        else:
                            k = j + 2 + compressed[j+1]
                        if k == len(compressed) and tag < 2:
                            compressed = compressed[:j]
                            break
                        j = k
                if len(compressed) <= 255:
                    LOWPAN_NHC_NH_0 = bytes([
                        0b11100000 | (EID << 1) | 0, ext_NH, len(compressed)
                    ]) + compressed
                    LOWPAN_NHC_NH_1 = bytes([
                        0b11100000 | (EID << 1) | 1, len(compressed)
                    ]) + compressed
                else:
                    LOWPAN_NHC_NH_0 = LOWPAN_NHC_NH_1 = None
                header_stack.append((LOWPAN_NHC_NH_0, LOWPAN_NHC_NH_1, uncompressed, len(uncompressed)))
                IPv6_NH = ext_NH
            elif IPv6_NH == 17:
                # UDP
                uncompressed = p.peekOctets(8)
                UDP_SP, UDP_DP, UDP_length, UDP_checksum = read(16), read(16), read(16), read(16)
                if UDP_SP & 0xfff0 == 0xf0b0 and UDP_DP & 0xfff0 == 0xf0b0:
                    P, P_inline = 3, bytes([(UDP_SP & 0xf) << 4 | (UDP_DP & 0xf)])
                elif UDP_SP & 0xff00 == 0xf000:
                    P, P_inline = 2, bytes([UDP_SP & 0xff, (UDP_DP >> 8) & 0xff, UDP_DP & 0xff])
                elif UDP_DP & 0xff00 == 0xf000:
                    P, P_inline = 1, bytes([(UDP_SP >> 8) & 0xff, UDP_SP & 0xff, UDP_DP & 0xff])
                else:
                    P, P_inline = 0, bytes([(UDP_SP >> 8) & 0xff, UDP_SP & 0xff, (UDP_DP >> 8) & 0xff, UDP_DP & 0xff])
                C, C_inline = 1, bytes() # always elide UDP checksum; rely on link-layer FCS
                LOWPAN_NHC = bytes([0b11110000 | (C << 2) | P]) + P_inline + C_inline
                header_stack.append((LOWPAN_NHC, None, uncompressed, 8))
                break # no more headers
            else:
                break

        payload = p.tail()

        # use as many compressed headers as possible

        # A proposal is a natural number.
        # The atomic_length of a proposal p is
        # p>0 ? (sum(len(header_stack[i][1]) for i in range(p-1)) + len(header_stack[p-1][0])) : 1
        # The full_length of a proposal p is
        # atomic_length(p) + sum(len(header_stack[i][2]) for i in range(p, len(header_stack))) + len(payload)
        # A proposal p is acceptable iff atomic_length(p) <= self.ll_mtu: we can't use compression
        # in fragments after the first.
        # We want the acceptable proposal with the least full_length.

        # Sanity check that we can make forward progress:
        # In the first fragment, 4 bytes FRAG1 header + 1 byte IPv6 dispatch + 8 bytes (fragment offset quantum)
        # In subsequent fragments, 5 bytes FRAGN header + 8 bytes (fragment offset quantum)
        # While we could get away with less for the first fragment if it has compressed headers, the requirement
        # still binds for later fragments.
        assert self.ll_mtu >= 13

        # XXX check for logic and off-by-one errors
        l = np.array([[len(xx) if xx is not None else np.inf for xx in x[:3]] for x in header_stack])
        cl = np.r_[[[0,0,0]], l.cumsum(0)]
        atomic_length = np.r_[0, cl[:-1,1]+l[:,0]]
        full_length = atomic_length + cl[-1,2] - cl[:,2] + len(payload)
        requires_fragmentation = full_length > self.ll_mtu
        atomic_length += requires_fragmentation * 4 # FRAG1 header is four bytes
        acceptable = atomic_length <= self.ll_mtu
        i = acceptable.nonzero()[0][full_length[acceptable].argmin()] # is this still correct when we haven't included FRAGn headers?
        atomic_part = b''.join(h[1] for h in header_stack[:i-1]) + header_stack[i-1][0]
        divisible_part = b''.join(h[2] for h in header_stack[i:]) + payload
        divisible_part_offset = sum(h[3] for h in header_stack[:i])
        fragment = len(atomic_part) + len(divisible_part) > self.ll_mtu
        if not fragment:
            return [atomic_part + divisible_part]
        else:
            slack = self.ll_mtu-len(atomic_part)-4
            assert slack >= 0
            split = slack & ~7
            datagram_size = len(p)
            datagram_tag = self.nextFragmentTag()
            frag1 = bytes([
                0b11000000 | (datagram_size >> 8) & 0x7, datagram_size & 0xff,
                (datagram_tag >> 8) & 0xff, datagram_tag & 0xff
            ]) + atomic_part + divisible_part[:split]
            per_fragment = (self.ll_mtu-5) & ~7
            fragN = [
                bytes([
                    0b11100000 | (datagram_size >> 8) & 0x7, datagram_size & 0xff,
                    (datagram_tag >> 8) & 0xff, datagram_tag & 0xff,
                    (divisible_part_offset+split+i*per_fragment)//8
                ]) + divisible_part[split+per_fragment*i:split+per_fragment*(i+1)]
                for i in range((len(divisible_part)-split+per_fragment-1)//per_fragment)
            ]
            return [frag1] + fragN

    def compressIPv6SA(self, IPv6_SA: bytes, p: Packet):
        # 0-byte options
        if IPv6_SA == bytes(16):
            return 0b100, 0, bytes()
        if IPv6_SA == link_local_prefix + modEUI64(p.ll_sa):
            return 0b011, 0, bytes()
        for SCI in range(15):
            context_free = bytes(8) + modEUI64(p.ll_sa)
            contextualized = self.context[SCI].rewriteSA(context_free)
            if IPv6_SA == contextualized:
                return 0b111, SCI, bytes()
        # 2-byte options
        if IPv6_SA[:14] == link_local_prefix + b'\0\0\0\xff\xfe\0':
            return 0b010, 0, IPv6_SA[14:]
        for SCI in range(15):
            contextualized_0 = self.context[SCI].rewriteSA(bytes(8) + b'\0\0\0\xff\xfe\0\0\0')
            contextualized_1 = self.context[SCI].rewriteSA(bytes(8) + b'\0\0\0\xff\xfe\0\xff\xff')
            context_mask = bytes([0xff^aa^bb for aa,bb in zip(contextualized_0, contextualized_1)])
            if matches(IPv6_SA, context_mask, contextualized_0):
                return 0b110, SCI, IPv6_SA[14:]
        # 8-byte options
        if IPv6_SA[:8] == link_local_prefix:
            return 0b001, 0, IPv6_SA[8:]
        for SCI in range(15):
            contextualized_0 = self.context[SCI].rewriteSA(bytes(8) + b'\0\0\0\0\0\0\0\0')
            contextualized_1 = self.context[SCI].rewriteSA(bytes(8) + b'\xff\xff\xff\xff\xff\xff\xff\xff')
            context_mask = bytes([0xff^aa^bb for aa,bb in zip(contextualized_0, contextualized_1)])
            if matches(IPv6_SA, context_mask, contextualized_0):
                return 0b101, SCI, IPv6_SA[8:]
        # 16-byte options
        return 0b000, 0, IPv6_SA

    def compressIPv6DA(self, IPv6_DA: bytes, p: Packet):
        if IPv6_DA[0] != 0xff:
            # unicast
            # 0-byte options
            if IPv6_DA == link_local_prefix + modEUI64(p.ll_da):
                return 0b0011, 0, bytes()
            for DCI in range(15):
                context_free = bytes(8) + modEUI64(p.ll_da)
                contextualized = self.context[DCI].rewriteDA(context_free)
                if IPv6_DA == contextualized:
                    return 0b0111, DCI, bytes()
            # 2-byte options
            if IPv6_DA[:14] == link_local_prefix + b'\0\0\0\xff\xfe\0':
                return 0b0010, 0, IPv6_DA[14:]
            for DCI in range(15):
                contextualized_0 = self.context[DCI].rewriteDA(bytes(8) + b'\0\0\0\xff\xfe\0\0\0')
                contextualized_1 = self.context[DCI].rewriteDA(bytes(8) + b'\0\0\0\xff\xfe\0\xff\xff')
                context_mask = bytes([0xff^aa^bb for aa,bb in zip(contextualized_0, contextualized_1)])
                if matches(IPv6_DA, context_mask, contextualized_0):
                    return 0b0110, DCI, IPv6_DA[14:]
            # 8-byte options
            if IPv6_DA[:8] == link_local_prefix:
                return 0b0001, 0, IPv6_DA[8:]
            for DCI in range(15):
                contextualized_0 = self.context[DCI].rewriteDA(bytes(8) + b'\0\0\0\0\0\0\0\0')
                contextualized_1 = self.context[DCI].rewriteDA(bytes(8) + b'\xff\xff\xff\xff\xff\xff\xff\xff')
                context_mask = bytes([0xff^aa^bb for aa,bb in zip(contextualized_0, contextualized_1)])
                if matches(IPv6_DA, context_mask, contextualized_0):
                    return 0b0101, DCI, IPv6_DA[8:]
            # 16-byte options
            return 0b0000, 0, IPv6_DA
        else:
            # multicast
            # 1-byte options
            if IPv6_DA[:15] == b'\xff\x02\0\0\0\0\0\0\0\0\0\0\0\0\0':
                return 0b1011, 0, IPv6_DA[15:]
            # 4-byte options
            if IPv6_DA[:1] == b'\xff' and IPv6_DA[2:13] == bytes(11):
                return 0b1010, 0, IPv6_DA[1:2] + IPv6_DA[13:]
            # 6-byte options
            if IPv6_DA[:1] == b'\xff' and IPv6_DA[2:11] == bytes(9):
                return 0b1001, 0, IPv6_DA[1:2] + IPv6_DA[11:]
            for DCI in range(15):
                if IPv6_DA[:1] == b'\xff' and IPv6_DA[2:12] == self.context[DCI].unicastPrefixBasedMulticast:
                    return 0b1100, DCI, IPv6_DA[1:3] + IPv6_DA[12:]
            # 16-byte options
            return 0b1000, 0, IPv6_DA

    def dispatchFragmentedPDU(self, p: Packet):
        dispatch = p.peekOctet()
        if dispatch & 0b11000000 == 0: # Not a LoWPAN frame
            return
        if dispatch == 0b01000001: # uncompressed IPv6 header
            p.readOctets(1)
            return self.dispatchIPv6PDU(p)
        if dispatch == 0b01000000: # ESC
            return
        if dispatch & 0b11100000 == 0b01100000: # LOWPAN_IPHC
            payload = self.decompressIPHCPDU(p, uncompressed_size=None)
            return self.dispatchIPv6PDU(Packet(p.ll_sa, p.ll_da, payload))
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
                    payload = self.decompressHC1PDU(p)
                elif dispatch & 0b11100000 == 0b01100000: # LOWPAN_IPHC
                    payload = self.decompressIPHCPDU(p, uncompressed_size=size)
                else:
                    return
            k = ReassemblyKey(p.ll_sa, p.ll_da, size, tag)
            f = ReassemblyFragment(offset, payload)
            self.dispatchFragment(k, f)

    def decompressHC1PDU(self, p: Packet):
        pass
        # def decompressAddress(la, mode, pdu, i):
        #     if (mode & 2) == 0:
        #         prefix, i = pdu[i:i+8], i+8
        #     else:
        #         prefix = link_local_prefix
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

    def decompressIPHCPDU(self, p: Packet, uncompressed_size: Optional[int]):
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
        else:                   SA_prefix = link_local_prefix
        if SACSAM & 3 == 3:     SA_iid = modEUI64(p.ll_sa)
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
            elif MDACDAM & 0xc == 0:    DA_prefix = link_local_prefix
            elif 8 < MDACDAM < 11:      DA_prefix = b'\xff' + readOctets(1) + b'\0\0\0\0\0\0'
            else:                       DA_prefix = b'\xff\x02\0\0\0\0\0\0'
            if MDACDAM & 11 == 1:       DA_iid = readOctets(8)
            elif MDACDAM & 11 == 2:     DA_iid = b'\0\0\0\xff\xfe\0' + readOctets(2)
            elif MDACDAM & 11 == 3:     DA_iid = modEUI64(p.ll_da)
            elif MDACDAM == 9:          DA_iid = bytes(3) + readOctets(5)
            elif MDACDAM == 10:         DA_iid = bytes(5) + readOctets(3)
            elif MDACDAM == 11:         DA_iid = bytes(7) + readOctets(1)
            IPv6_DA = DA_prefix + DA_iid
            if MDACDAM & 0xc == 4:
                IPv6_DA = p.pdb.context[DCI].rewriteDA(IPv6_DA)
        if NH == 1: # LOWPAN_NHC
            p.ip_sa = IPv6_SA
            p.ip_da = IPv6_DA
            IPv6_NH, payload = self.decompressNHCPDU(p, IPv6_payload_length)
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

    def decompressNHCPDU(self, p: Packet, uncompressed_size: Optional[int]):
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
            if EID == 1: # Routing Header                 [RFC6282][RFC2460]
                this_header = 43
            if EID == 2: # Fragment Header                [RFC6282][RFC2460]
                this_header = 44
                ext_length = 0
                assert ext_length_octets == 6
            if EID == 3: # Destinations Options Header    [RFC6282][RFC2460]
                this_header = 60
            if EID == 4: # Mobility Header                [RFC6282][RFC6275]
                this_header = 135
            if EID == 0 or EID == 3:
                padbytes = (-2-ext_length_octets) % 8
                if padbytes == 1:
                    ext_tail += b'\0'
                elif padbytes > 1:
                    ext_tail += bytes([1, padbytes-2] + [0]*(N-2))
            if NH == 1:
                if uncompressed_size is not None:
                    uncompressed_size -= 2 + len(ext_tail)
                ext_NH, payload = self.decompressNHCPDU(p, uncompressed_size)
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
                UDP_checksum = ((UDP_checksum % 0xffff) ^ 0xffff) or 0xffff
            UDP_header = bytes([(UDP_SP >> 8) & 0xff, UDP_SP & 0xff, (UDP_DP >> 8) & 0xff, UDP_DP & 0xff,
                                (UDP_length >> 8) & 0xff, UDP_length & 0xff, (UDP_checksum >> 8) & 0xff, UDP_checksum & 0xff])
            return this_header, UDP_header + payload

# Don't use 0xf0bX ports without an integrity check b/c they're overloaded.

def roundtrip_test(datagram: bytes):
    pdb = PDB()
    ll_sa = binascii.unhexlify('8c8590843fcc')
    ll_da = binascii.unhexlify('000000000000')
    pdb.ll_mtu = 65536
    compressed = pdb.compressIPv6Datagram(Packet(ll_sa, ll_da, datagram))[0]
    decompressed = pdb.decompressIPHCPDU(Packet(ll_sa, ll_da, compressed), None)
    return decompressed == datagram

if __name__ == '__main__':
    for d in [
        '6000000000240001fe800000000000008e8590fffe843fccff0200000000000000000000000000163a000100050200008f0010430000000106000000ff0200000000000000000001ffef0001',
        '60000000004c0001fe800000000000008e8590fffe843fccff0200000000000000000000000000163a000100050200008f009a880000000304000000ff0200000000000000000002ff6930cb04000000ff0200000000000000000001ff843fcc06000000ff0200000000000000000001ffef0002',
        '6000000000240001fe800000000000008e8590fffe843fccff0200000000000000000000000000163a000100050200008f0012430000000104000000ff0200000000000000000001ffef0001',
        '6000000000380001fe800000000000008e8590fffe843fccff0200000000000000000000000000163a000100050200008f00cfd70000000204000000ff0200000000000000000001ff843fcc04000000ff0200000000000000000001ffef0002',
        '6000000000240001fe800000000000008e8590fffe843fccff0200000000000000000000000000163a000100050200008f0012420000000104000000ff0200000000000000000001ffef0002',
        ]:
        print(roundtrip_test(binascii.unhexlify(d)))
