#!/usr/bin/env python3.4
from ..net import sockio

print(':'.join('%02x' % x for x in next(iter(sockio.getlinkaddrs()['en0']))))
