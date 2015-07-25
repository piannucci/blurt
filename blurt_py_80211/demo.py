#!/usr/bin/env python
import sys
import numpy as np
import audio, audioLoopback, wifi80211

wifi = wifi80211.WiFi_802_11()

class ContinuousReceiver(audioLoopback.AudioBuffer):
    def init(self):
        packetLength = wifi.encode(np.zeros(100, int), 0).size
        self.kwargs['maximum'] = int(packetLength*4)
        self.kwargs['trigger'] = int(packetLength*1)
        self.dtype = np.complex64
        super(ContinuousReceiver, self).init()
        Fc = self.kwarg('Fc')
        Fs = self.kwarg('Fs')
        upsample_factor = self.kwarg('upsample_factor')
        self.inputProcessor = audioLoopback.InputProcessor(Fs, Fc, upsample_factor)
    def trigger_received(self):
        sys.stdout.write('.')
        sys.stdout.flush()
        input = self.peek(self.maximum)
        endIndex = 0
        try:
            results, _ = wifi.decode(input, False, False)
            for payload, startIndex, endIndex, lsnr_estimate in results:
                print(repr(''.join(map(chr, payload))) + (' @ %.3f dB' % lsnr_estimate))
        except Exception, e:
            print(repr(e))
        if endIndex:
            return endIndex
        else:
            return self.trigger/2

bandwidth = None

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
            description='Simple command-line Blurt transmitter/receiver',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--Fs', '-s', metavar='fs', type=float,
            help='audio sample rate in Hz', default=48000.)
    parser.add_argument('--Fc', '-c', metavar='fc', type=float,
            help='carrier (center) frequency in Hz', default=19000.)
    parser.add_argument('--rate', '-r', metavar='rate', type=int,
            help='modulation and coding scheme (MCS) index, from 0 to 7', choices=range(8), default=0)
    class BandwidthAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            namespace.upsample_factor = int(round(namespace.Fs/values))
    parser.add_argument('-b', '--bandwidth', metavar='bw', type=float, action=BandwidthAction,
        help='desired bandwidth in Hz; if used with a non-default --Fs, this option must follow that one', dest='bandwidth', default=3000.0)
    parser.add_argument('--rx', action='store_true', default=False, help='listen for incoming Blurts')
    parser.add_argument('--tx', action='store_false', dest='rx', help='send a Blurt')
    parser.add_argument('message', nargs='?', default='Hello, world!')
    args = parser.parse_args(namespace=argparse.Namespace(upsample_factor=16))

    bandwidth = args.Fs / args.upsample_factor
    if args.Fc + bandwidth*.5 > args.Fs * .5:
        parser.error('Center frequency plus half of bandwidth cannot exceed half of sampling frequency (Nyquist criterion)')

    if args.rx:
        print('Listening for transmissions with a center frequency of %.0f Hz and a bandwidth of %.0f Hz (sample rate %.0f)' % \
            (args.Fc, args.Fs / args.upsample_factor, args.Fs))
        audio.record(ContinuousReceiver(Fs=args.Fs, Fc=args.Fc, upsample_factor=args.upsample_factor), args.Fs)
    else:
        print('Transmitting %r with a center frequency of %.0f Hz and a bandwidth of %.0f Hz (sample rate %.0f)' % \
            (args.message, args.Fc, args.Fs / args.upsample_factor, args.Fs))
        input_octets = np.array(map(ord, args.message), dtype=np.uint8)
        output = wifi.encode(input_octets, args.rate)
        audioLoopback.audioOut(output, args.Fs, args.Fc, args.upsample_factor, None)
