#!/usr/bin/env python
import numpy as np

scrambler_y = np.zeros((127,128), np.uint8)
s = np.arange(128, dtype=np.uint8)
for i in range(127):
    s = np.uint16((s << 1) ^ (1 & ((s >> 3) ^ (s >> 6))))
    scrambler_y[i] = s&1

def scrambler(initial):
    y = scrambler_y[:,np.uint8(initial)]
    i = 0
    while True:
        yield y[i]
        i = (i+1)%127

def scramble(input, multipleof, scramblerState=0x7F):
    sc = scrambler_y[:,np.uint8(scramblerState)]
    pad_bits = 6
    if multipleof != None:
        stragglers = (input.size + 6) % multipleof
        if stragglers:
            pad_bits += multipleof - stragglers
    output = np.r_[np.array(input, int)&1, np.zeros(pad_bits, int)]
    output ^= np.tile(sc, (output.size+126)/127)[:output.size]
    output[input.size:input.size+6] = 0
    return output
