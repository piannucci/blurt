import numpy as np
from . import util

protocols = dict([(lambda a:(int(a[1]), a[0]))(l.split('\t')[:2]) for l in open('/etc/protocols', 'r').readlines() if not l.startswith('#')])

def process(msdu):
    dsap, ssap, control = msdu[:3]
    protocol_id = util.shiftin(util.shiftout(msdu[3:6][::-1], 8), 24)[0]
    if not dsap == 0xaa and ssap == 0xaa and control == 3 and protocol_id == 0:
        return {'error': 'Not using IPv4 over SNAP in LLC (RFC 1042)'}
    ethertype = util.shiftin(util.shiftout(msdu[6:8][::-1], 8), 16)[0]
    result = dict(ethertype='%04x' % ethertype)
    if ethertype == 0x0800:
        llc_payload = msdu[8:]
        version, ihl = util.shiftin(util.shiftout(llc_payload[0], 8), 4)[::-1]
        if not version == 4:
            return dict(result, error='Not using IPv4')
        ip_header = llc_payload[:ihl*4]
        src_ip, dst_ip = ['.'.join(map(str, ip_header[offset:offset+4])) for offset in range(12, 20, 4)]
        result.update(ethertype='ipv4', src_ip=src_ip, dst_ip=dst_ip)
        ip_payload = llc_payload[ihl*4:]
        proto = protocols.get(ip_header[9], 'unknown')
        result.update(proto=proto)
        if proto in ('tcp', 'udp'):
            result.update(
                src_port=util.shiftin(util.shiftout(ip_payload[0:2][::-1], 8), 16)[0],
                dst_port=util.shiftin(util.shiftout(ip_payload[2:4][::-1], 8), 16)[0],
            )
        if proto == 'tcp':
            data_offset = util.shiftin(util.shiftout(ip_payload[12], 8)[4:8], 4)[0]
            result.update(payload=bytes(ip_payload[data_offset*4:].astype(np.uint8)))
        elif proto == 'udp':
            result.update(payload=bytes(ip_payload[8:].astype(np.uint8)))
        return result
    elif ethertype == 0x0806:
        return dict(result, ethertype='arp')
    elif ethertype == 0x8035:
        return dict(result, ethertype='rarp')
    elif ethertype == 0x86DD:
        return dict(result, ethertype='ipv6')
    else:
        return dict(result, error='Unknown ethertype')
