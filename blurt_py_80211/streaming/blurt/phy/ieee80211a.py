import collections
import itertools
import numpy as np
from ..graph import Output, Input, Block
from .iir import IIRFilter
from . import cc
from . import rates
from . import ofdm
from .interleaver import interleave
from .crc import CRC32_802_11_FCS as FCS
from . import scrambler
from . import correlator
from .graph_adapter import GenericDecoderBlock

Channel = collections.namedtuple('Channel', ['Fs', 'Fc', 'upsample_factor'])

preemphasisOrder = 0

############################ OFDM ############################

class Clause18Decoder:
    def __init__(self, i, nChannelsPerFrame, oversample):
        self.oversample = oversample
        self.mtu = 1500
        self.start = i
        self.j = 0
        self.nChannelsPerFrame = nChannelsPerFrame
        max_coded_bits = ((2 + self.mtu + 4) * 8 + 6) * 2
        max_data_symbols = (max_coded_bits+ofdm.L.Nsc-1) // ofdm.L.Nsc
        max_samples = ofdm.L.N_training_samples + ofdm.L.nsym * (1 + max_data_symbols)
        self.y = np.full((max_samples, nChannelsPerFrame), np.inf, complex)
        self.size = 0
        self.trained = False
        self.result = None
    def process(self, frames, k):
        if self.result is not None:
            return self.result
        if k < self.start:
            frames = frames[self.start-k:]
        nsym = ofdm.L.nsym
        self.y[self.size:self.size+frames.shape[0]] = frames[:self.y.shape[0]-self.size]
        self.size += frames.shape[0]
        if not self.trained:
            if self.size > ofdm.L.N_training_samples:
                self.training_data, self.i = ofdm.L.train(self.y)
                syms = self.y[self.i:self.i+(self.y.shape[0]-self.i)//nsym*nsym]
                self.syms = ofdm.L.ekfDecoder(syms.reshape(-1, nsym, self.nChannelsPerFrame), self.i, self.training_data)
                self.trained = True
            else:
                return
        j_valid = (self.size - self.i) // nsym
        if self.j == 0:
            if j_valid > 0:
                lsig = next(self.syms)
                SIGNAL_bits = cc.decode(interleave(lsig.real, ofdm.L.Nsc, 1, reverse=True))
                if not int(SIGNAL_bits.sum()) & 1 == 0:
                    self.result = ()
                    return self.result
                SIGNAL_bits = (SIGNAL_bits[:18] << np.arange(18)).sum()
                self.rate = rates.L_rate(SIGNAL_bits & 0xf)
                if self.rate is None:
                    self.result = ()
                    return self.result
                length_octets = (SIGNAL_bits >> 5) & 0xfff
                if length_octets > self.mtu:
                    self.result = ()
                    return self.result
                self.length_coded_bits = (length_octets*8 + 16+6)*2
                self.Ncbps = ofdm.L.Nsc * self.rate.Nbpsc
                self.length_symbols = int((self.length_coded_bits+self.Ncbps-1) // self.Ncbps)
                SIGNAL_coded_bits = interleave(cc.encode((SIGNAL_bits >> np.arange(24)) & 1), ofdm.L.Nsc, 1)
                self.dispersion = (lsig-(SIGNAL_coded_bits*2.-1.)).var()
                self.j = 1
            else:
                return
        if j_valid < self.length_symbols + 1:
            return
        syms = np.array(list(itertools.islice(self.syms, self.length_symbols)))
        demapped_bits = self.rate.constellation[0].demap(syms, self.dispersion)
        deinterleaved_bits = interleave(demapped_bits, self.Ncbps, self.rate.Nbpsc, reverse=True)
        llr = self.rate.depuncture(deinterleaved_bits)[:self.length_coded_bits]
        output_bits = scrambler.descramble(cc.decode(llr))[16:-6]
        if not FCS.check(output_bits):
            self.result = ()
            return self.result
        output_bytes = (output_bits[:-32].reshape(-1, 8) << np.arange(8)).sum(1)
        self.result = output_bytes.astype(np.uint8).tostring(), 10*np.log10(1/self.dispersion)
        return self.result

class IEEE80211aDecoderBlock(GenericDecoderBlock):
    def __init__(self, channel):
        super().__init__(channel, correlator.Clause18Detector, Clause18Decoder)

class IEEE80211aEncoderBlock(Block):
    inputs = [Input(())]
    outputs = [Output(np.float32, ('nChannelsPerFrame',))]

    def __init__(self, channel, nChannelsPerFrame=2):
        super().__init__()
        self.channel = channel
        self.nChannelsPerFrame = 2
        self.oversample = 4
        self.preferredRate = 6

    def start(self):
        self.k = 0 # LO phase
        self.lp = IIRFilter.lowpass(
            0.5*self.oversample/self.channel.upsample_factor,
            axis=0,
            shape=(None,self.nChannelsPerFrame,)
        )

    def process(self):
        channel = self.channel
        baseRate = rates.L_rate(0xb)
        for datagram, in self.input():
            octets = np.frombuffer(datagram, np.uint8)
            # prepare header and payload bits
            rateEncoding = {6:0xb, 9:0xf, 12:0xa, 18:0xe, 24:0x9, 36:0xd, 48:0x8, 54:0xc}[self.preferredRate]
            rate = rates.L_rate(rateEncoding)
            data_bits = (octets[:,None] >> np.arange(8)[None,:]).flatten() & 1
            data_bits = np.r_[np.zeros(16, int), data_bits, FCS.compute(data_bits)]
            SIGNAL_bits = ((rateEncoding | ((octets.size+4) << 5)) >> np.arange(18)) & 1
            SIGNAL_bits[-1] = SIGNAL_bits.sum() & 1
            # OFDM modulation
            scrambler_state = np.random.randint(1,127)
            parts = (ofdm.L.subcarriersFromBits(SIGNAL_bits, baseRate, 0),
                     ofdm.L.subcarriersFromBits(data_bits, rate, scrambler_state))
            oversample = self.oversample
            output = ofdm.L.encode(parts, oversample, self.nChannelsPerFrame)
            # inter-frame space
            output = np.concatenate((
                output,
                np.zeros((round(ofdm.L.IFS*oversample), self.nChannelsPerFrame))
            ))
            # upsample
            output = self.lp(np.repeat(output, channel.upsample_factor // oversample, 0))
            # upconvert
            Omega = 2*np.pi*channel.Fc/channel.Fs
            output = (output * np.exp(1j* Omega * np.r_[self.k:self.k+output.shape[0]])[:,None]).real
            self.k += output.shape[0]
            if 0:
                # pre-emphasize TODO make this work on output with shape[time,spatial stream]
                for i in range(preemphasisOrder):
                    output = np.diff(np.r_[output,0])
                output *= abs(np.exp(1j*Omega)-1)**-preemphasisOrder
            # crest control
            output /= abs(output).max()
            self.output((output,))
