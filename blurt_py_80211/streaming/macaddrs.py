#!/usr/bin/env python3.4
import sockio

print(':'.join('%02x' % x for x in next(iter(sockio.getlinkaddrs()['en0']))))
