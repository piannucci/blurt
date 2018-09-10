import numpy as np
from . import crc
from . import util
from . import msdu

def process(mpdu, HT):
    if not crc.CRC32_802_11_FCS.check(util.shiftout(mpdu, 8)):
        return {'error': 'MPDU checksum failed'}
    frame_control = util.shiftout(mpdu[:2], 8)
    protocol_version = frame_control[:2]
    if not (protocol_version == 0).all():
        return {'error': 'Unrecognized protocol version'}
    type = util.shiftin(frame_control[2:4], 2)[0]
    type_string = ['Management', 'Control', 'Data', 'Reserved'][type]
    subtype = util.shiftin(frame_control[4:8], 4)[0]
    subtype_string = {
        0: ['Association request', 'Association response', 'Reassociation request', 'Reassociation response', 'Probe request', 'Probe response', 'Timing Advertisement', 'Reserved',
            'Beacon', 'ATIM', 'Disassociation', 'Authentication', 'Deauthentication', 'Action', 'Action No Ack', 'Reserved'],
        1: ['Reserved']*7 + ['Control Wrapper', 'Block Ack Request', 'Block Ack', 'PS-Poll', 'RTS', 'CTS', 'ACK', 'CF-End', 'CF-End + CF-Ack'],
        2: ['Data', 'Data + CF-Ack', 'Data + CF-Poll', 'Data + CF-Ack + CF-Poll', 'Null', 'CF-Ack', 'CF-Poll', 'CF-Ack + CF-Poll',
            'QoS Data', 'QoS Data + CF-Ack', 'QoS Data + CF-Poll', 'QoS Data + CF-Ack + CF-Poll', 'QoS Null', 'Reserved', 'QoS CF-Poll', 'QoS CF-Ack + CF-Poll'],
        3: ['Reserved']*16,
    }[type][subtype]
    to_ds, from_ds, more_fragments, retry, power_management, more_data, protected_frame, order = frame_control[8:]
    duration_id = util.shiftout(mpdu[2:4], 8)
    if type in (0, 2):
        address_offsets = [4, 10, 16] + ([24] if (to_ds == 1 and from_ds == 1) else [])
    elif type == 1:
        address_offsets = [4, 10][:{7:1, 8:2, 9:2, 10:2, 11:2, 12:1, 13:1, 14:2, 15:2}[subtype]]
    else:
        address_offsets = []
    address_count = len(address_offsets)
    addresses = [':'.join('%02x' % b for b in mpdu[i:i+6]) for i in address_offsets]
    result = dict(type=type_string,
                  subtype=subtype_string,
                  to_ds=to_ds,
                  from_ds=from_ds,
                  more_fragments=more_fragments,
                  addresses=addresses)
    sequence_control = mpdu[22:24]
    j = 6 + address_count*6
    qos_control_present = bool(subtype & 8)
    a_msdu_present = False
    if qos_control_present:
        if j >= mpdu.size:
            return result
        qos_control = util.shiftout(mpdu[j:j+2], 8)
        a_msdu_present = qos_control[7]
    j += 2*qos_control_present
    ht_control_present = HT and order
    j += 4*ht_control_present
    frame_body = mpdu[j:-4]
    if type == 2:
        if a_msdu_present:
            msdus = []
            while j < frame_body.size:
                da = frame_body[j:j+6]
                sa = frame_body[j+6:j+12]
                msdu_length = util.shiftin(util.shiftout(frame_body[j+12:j+14], 8), 16)[0]
                if msdu_length:
                    msdus.append(frame_body[j+14:j+14+msdu_length])
                j += 14+msdu_length
        elif len(frame_body):
            msdus = [frame_body]
        else:
            msdus = []
        result.update(msdu=[msdu.process(x) for x in msdus])
    else:
        result.update(frame_body=bytes(frame_body.astype(np.uint8)))
    return result
