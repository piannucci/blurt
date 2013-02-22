#!/usr/bin/env python
from pylab import *
from numpy import *
import sys

results = []

for fn in sys.argv[1:]:
    with open(fn, 'r') as f:
        results.extend(eval(f.read()))

results = array(results)

figure()
subplot(211)
hist(results[:,0], bins=100, normed=True, cumulative=True, histtype='step')
xlim(0,amax(results[:,0]))
subplot(212)
hist(results[:,1], bins=100, normed=True, cumulative=True, histtype='step')
xlim(0,amax(results[:,0]))

figure()
hist(results[:,1]/results[:,0], bins=100, normed=True, cumulative=True, histtype='step')
xlim(0,1)

ion()
draw()
