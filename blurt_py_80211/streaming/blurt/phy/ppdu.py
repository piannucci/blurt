from . import crc
from . import util
from . import mpdu

def process(ppdu, aggregation, HT):
    # disaggregate
    if aggregation:
        i = 0
        mpdus = []
        while i < ppdu.size:
            mpdu_delimiter = util.shiftout(ppdu[i:i+4], 8)
            if not (crc.CRC8_HT_SIG.compute(mpdu_delimiter[:16]) == mpdu_delimiter[16:24]).all():
                return {'error': 'MPDU delimiter checksum failed'}
            if not chr(ppdu[i+3]) == 'N':
                return {'error': 'MPDU delimiter signature incorrect'}
            mpdu_length = util.shiftin(mpdu_delimiter[4:16], 12)[0]
            if mpdu_length:
                mpdus.append(ppdu[i+4:i+4+mpdu_length])
            i += 4+mpdu_length
            i = (i+3)//4*4
    else:
        mpdus = [ppdu]
    return dict(mpdu=[mpdu.process(x, HT) for x in mpdus])
