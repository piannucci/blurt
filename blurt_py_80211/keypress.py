#!/usr/bin/env python

import ctypes, ctypes.util

carbon = ctypes.CDLL(ctypes.util.find_library('Carbon'))

kCGSessionEventTap = ctypes.c_uint32(1)

carbon.CGEventCreateKeyboardEvent.restype = ctypes.c_void_p
carbon.CGEventCreateKeyboardEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_bool]
carbon.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
carbon.CFRelease.argtypes = [ctypes.c_void_p]

kb = {'0': 0x1D, '1': 0x12, '2': 0x13, '3': 0x14, '4': 0x15, '5': 0x17, '6': 0x16, '7': 0x1A, '8': 0x1C, '9': 0x19,
      'A': 0x00, 'B': 0x0B, 'C': 0x08, 'D': 0x02, 'E': 0x0E, 'F': 0x03, 'G': 0x05, 'H': 0x04, 'I': 0x22, 'J': 0x26,
      'K': 0x28, 'L': 0x25, 'M': 0x2E, 'N': 0x2D, 'O': 0x1F, 'P': 0x23, 'Q': 0x0C, 'R': 0x0F, 'S': 0x01, 'T': 0x11,
      'U': 0x20, 'V': 0x09, 'W': 0x0D, 'X': 0x07, 'Y': 0x10, 'Z': 0x06, '\n': 0x24}

def type(s):
    for c in s.upper():
        vk = kb[c]
        e = carbon.CGEventCreateKeyboardEvent(0, vk, 1);
        carbon.CGEventPost(kCGSessionEventTap, e);
        carbon.CFRelease(e);
        e = carbon.CGEventCreateKeyboardEvent(0, vk, 0);
        carbon.CGEventPost(kCGSessionEventTap, e);
        carbon.CFRelease(e);
