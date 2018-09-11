import collections
import itertools
import typing
import numpy as np
from ..graph import Output, Input, Block
from .iir import IIRFilter
from . import cc
from . import rates
from . import ofdm
from .interleaver import interleave
from .crc import CRC32_802_11_FCS as FCS
from . import scrambler

Channel = collections.namedtuple('Channel', ['Fs', 'Fc', 'upsample_factor'])

stereoDelay = .005
preemphasisOrder = 0
IFS = 2.5 * 80 * 8 / 96e3 # inter-frame space

nfft = ofdm.L.nfft
ncp = ofdm.L.ncp
sts_freq = ofdm.L.sts_freq
lts_freq = ofdm.L.lts_freq
ts_reps = ofdm.L.ts_reps
dataSubcarriers = ofdm.L.dataSubcarriers
pilotSubcarriers = ofdm.L.pilotSubcarriers
pilotTemplate = ofdm.L.pilotTemplate
Nsc = ofdm.L.Nsc
Nsc_used = ofdm.L.Nsc_used

############################ OFDM ############################

N_sts_period = nfft // 4
N_sts_samples = ts_reps * (ncp + nfft)
N_training_samples = N_sts_samples + ts_reps * (ncp + nfft) + 8

class Autocorrelator:
    def __init__(self, nChannelsPerFrame, oversample):
        self.y_hist = np.zeros((0, nChannelsPerFrame))
        self.results = []
        self.nChannelsPerFrame = nChannelsPerFrame
        self.quantum = oversample * N_sts_period
        self.width = N_sts_samples // N_sts_period
        self.history = self.quantum * self.width
    def feed(self, y):
        y = np.r_[self.y_hist, y]
        count_needed = y.shape[0] // self.quantum * self.quantum
        count_consumed = count_needed - self.history
        if count_consumed <= 0:
            self.y_hist = y
        else:
            self.y_hist = y[count_consumed:]
            y = y[:count_needed].reshape(-1, self.quantum, self.nChannelsPerFrame)
            corr_sum = np.abs((y[:-1].conj() * y[1:]).sum(1)).cumsum(0)
            self.results.append((corr_sum[self.width-1:] - corr_sum[:-self.width+1]).mean(-1))
    def __iter__(self):
        while self.results:
            yield self.results.pop(0)

class PeakDetector:
    def __init__(self, l):
        self.y_hist = np.zeros(l)
        self.l = l
        self.i = 0
        self.results = []
    def feed(self, y):
        y = np.r_[self.y_hist, y]
        count_needed = y.size
        count_consumed = count_needed - 2*self.l
        if count_consumed <= 0:
            self.y_hist = y
        else:
            self.y_hist = y[count_consumed:]
            stripes_shape = (2*self.l+1, count_needed-2*self.l)
            stripes_strides = (y.strides[0],)*2
            stripes = np.lib.stride_tricks.as_strided(y, stripes_shape, stripes_strides)
            self.results.extend((stripes.argmax(0) == self.l).nonzero()[0] + self.i)
            self.i += count_consumed
    def __iter__(self):
        while self.results:
            yield self.results.pop(0)

def estimate_cfo(y, overlap, span):
    return np.angle((y[:-overlap].conj() * y[overlap:]).sum()) / span

def remove_cfo(y, k, theta):
    return y * np.exp(-1j*theta*np.r_[k:k+y.shape[0]])[:,None]

def train(y):
    Nrx_streams = y.shape[1]
    i = 0
    theta = estimate_cfo(y[i:i+N_sts_samples], N_sts_period, N_sts_period)
    i += N_sts_samples + ncp*ts_reps
    lts = np.fft.fft(remove_cfo(y[i:i+nfft*ts_reps], i, theta).reshape(-1, nfft, Nrx_streams), axis=1)
    theta += estimate_cfo(lts * (lts_freq != 0)[:,None], 1, nfft)
    def wienerFilter(i):
        lts = np.fft.fft(remove_cfo(y[i:i+nfft*ts_reps], i, theta).reshape(-1, nfft, Nrx_streams), axis=1)
        X = lts_freq[:,None]
        Y = lts.sum(0)
        YY = (lts[:,:,:,None] * lts[:,:,None,:].conj()).sum(0)
        YY_inv = np.linalg.inv(YY)
        G = np.einsum('ij,ik,ikl->ijl', X, Y.conj(), YY_inv)
        snr = Nrx_streams/(abs(np.einsum('ijk,lik->lij', G, lts) - X[None,:,:])**2).mean()
        return snr, i + nfft*ts_reps, G
    snr, i, G = max(map(wienerFilter, range(i-8, i+8)))
    var_input = y[:N_training_samples].var()
    var_n = var_input / (snr / Nrx_streams * Nsc_used / Nsc + 1)
    var_x = var_input - var_n
    var_y = 2*var_n*var_x + var_n**2
    uncertainty = np.arctan(var_y**.5 / var_x) / nfft**.5
    var_ni = var_x/Nsc_used*Nrx_streams/snr
    return (G, uncertainty, var_ni, theta), i

def decodeOFDM(syms, i, training_data):
    G, uncertainty, var_ni, theta_cfo = training_data
    Np = pilotSubcarriers.size
    sigma_noise = Np*var_ni*.5
    sigma = sigma_noise + Np*np.sin(uncertainty)**2
    P = np.diag([sigma, sigma, uncertainty**2])
    x = Np * np.array([[1.,0.,0.]]).T
    R = np.diag([sigma_noise, sigma_noise])
    Q = P * 0.1
    for j, y in enumerate(syms):
        sym = np.einsum('ijk,ik->ij', G, np.fft.fft(remove_cfo(y[ncp:], i+ncp, theta_cfo), axis=0))[:,0]
        i += nfft+ncp
        pilot = (sym[pilotSubcarriers]*pilotTemplate).sum() * float(1-2*scrambler.pilot_sequence[j%127])
        re,im,theta = x[:,0]
        c, s = np.cos(theta), np.sin(theta)
        F = np.array([[c, -s, -s*re - c*im], [s, c, c*re - s*im], [0, 0, 1]])
        x[0,0] = c*re - s*im
        x[1,0] = c*im + s*re
        P = F.dot(P).dot(F.T) + Q
        S = P[:2,:2] + R
        K = np.linalg.solve(S, P[:2,:]).T
        x += K.dot(np.array([[pilot.real], [pilot.imag]]) - x[:2,:])
        P -= K.dot(P[:2,:])
        u = x[0,0] - x[1,0]*1j
        yield sym[dataSubcarriers] * (u/abs(u))

class OneShotDecoder:
    def __init__(self, i, nChannelsPerFrame, oversample):
        self.oversample = oversample
        self.mtu = 200 # XXX
        self.start = i
        self.j = 0
        self.nChannelsPerFrame = nChannelsPerFrame
        max_coded_bits = ((2 + self.mtu + 4) * 8 + 6) * 2
        max_data_symbols = (max_coded_bits+Nsc-1) // Nsc
        max_samples = N_training_samples + (ncp+nfft) * (1 + max_data_symbols)
        self.y = np.full((max_samples, nChannelsPerFrame), np.inf, complex)
        self.size = 0
        self.trained = False
        self.result = None
    def feed(self, frames, k):
        if self.result is not None:
            return
        if k < self.start:
            frames = frames[self.start-k:]
        self.y[self.size:self.size+frames.shape[0]] = frames[:self.y.shape[0]-self.size]
        self.size += frames.shape[0]
        if not self.trained:
            if self.size > N_training_samples:
                self.training_data, self.i = train(self.y)
                syms = self.y[self.i:self.i+(self.y.shape[0]-self.i)//(nfft+ncp)*(nfft+ncp)]
                self.syms = decodeOFDM(syms.reshape(-1, (nfft+ncp), self.nChannelsPerFrame), self.i, self.training_data)
                self.trained = True
            else:
                return
        j_valid = (self.size - self.i) // (nfft + ncp)
        if self.j == 0:
            if j_valid > 0:
                lsig = next(self.syms)
                SIGNAL_bits = cc.decode(interleave(lsig.real, Nsc, 1, reverse=True))
                if not int(SIGNAL_bits.sum()) & 1 == 0:
                    self.result = ()
                    return
                SIGNAL_bits = (SIGNAL_bits[:18] << np.arange(18)).sum()
                self.rate = rates.L_rate(SIGNAL_bits & 0xf)
                if self.rate is None:
                    self.result = ()
                    return
                length_octets = (SIGNAL_bits >> 5) & 0xfff
                if length_octets > self.mtu:
                    self.result = ()
                    return
                self.length_coded_bits = (length_octets*8 + 16+6)*2
                self.Ncbps = Nsc * self.rate.Nbpsc
                self.length_symbols = int((self.length_coded_bits+self.Ncbps-1) // self.Ncbps)
                SIGNAL_coded_bits = interleave(cc.encode((SIGNAL_bits >> np.arange(24)) & 1), Nsc, 1)
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
            return
        output_bytes = (output_bits[:-32].reshape(-1, 8) << np.arange(8)).sum(1)
        self.result = output_bytes.astype(np.uint8).tostring(), 10*np.log10(1/self.dispersion)

class IEEE80211aDecoderBlock(Block):
    inputs = [Input(('nChannelsPerFrame',))]
    outputs = [Output(typing.Tuple[np.ndarray, float], ())]

    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.intermediate_upsample = 1

    def start(self):
        self.autocorrelator = Autocorrelator(self.nChannelsPerFrame, self.intermediate_upsample)
        self.peakDetector = PeakDetector(9) # 25
        self.decoders = []
        self.lookback = collections.deque()
        self.k_current = 0
        self.k_lookback = 0
        self.lp = IIRFilter.lowpass(.45/self.channel.upsample_factor, shape=(None, self.nChannelsPerFrame), axis=0)
        self.i = 0

    def process(self):
        channel = self.channel
        Omega = 2*np.pi*channel.Fc/channel.Fs
        upsample_factor = channel.upsample_factor
        for (y, inputTime, now), in self.input():
            nFrames = y.shape[0]
            y = self.lp(y * np.exp(-1j*Omega * np.r_[self.i:self.i+nFrames])[:,None])
            y = y[-self.i%channel.upsample_factor::channel.upsample_factor]
            self.i += nFrames
            self.autocorrelator.feed(y)
            for corr in self.autocorrelator:
                self.peakDetector.feed(corr)
            for peak in self.peakDetector:
                k = (peak+1) * N_sts_period * self.intermediate_upsample
                d = OneShotDecoder(k, self.nChannelsPerFrame, self.intermediate_upsample)
                k = self.k_lookback
                for y_old in self.lookback:
                    d.feed(y_old, k)
                    k += y_old.shape[0]
                self.decoders.append(d)
            self.lookback.append(y)
            for d in list(self.decoders):
                d.feed(y, self.k_current)
                if d.result is not None:
                    if d.result:
                        self.output((d.result,))
                    self.decoders.remove(d)
            self.k_current += y.shape[0]
            while self.lookback:
                if self.k_lookback + self.lookback[0].shape[0] >= self.k_current - 1024:
                    break
                self.k_lookback += self.lookback.popleft().shape[0]

def subcarriersFromBits(bits, rate, scramblerState):
    # adds tail bits and any needed padding to form a full symbol; does not add SERVICE
    Ncbps = Nsc * rate.Nbpsc
    Nbps = Ncbps * rate.ratio[0] // rate.ratio[1]
    pad_bits = 6 + -(bits.size + 6) % Nbps
    scrambled = scrambler.scramble(np.r_[bits, np.zeros(pad_bits, int)], scramblerState)
    scrambled[bits.size:bits.size+6] = 0
    punctured = cc.encode(scrambled)[np.resize(rate.puncturingMatrix, scrambled.size*2)]
    interleaved = interleave(punctured, Nsc * rate.Nbpsc, rate.Nbpsc)
    grouped = (interleaved.reshape(-1, rate.Nbpsc) << np.arange(rate.Nbpsc)).sum(1)
    return rate.constellation[0].symbols[grouped].reshape(-1, Nsc)

class IEEE80211aEncoderBlock(Block):
    inputs = [Input(())]
    outputs = [Output(np.float32, ('nChannelsPerFrame',))]

    def __init__(self, channel, nChannelsPerFrame=2):
        super().__init__()
        self.channel = channel
        self.nChannelsPerFrame = 2
        self.intermediate_upsample = 4

    def start(self):
        self.k = 0 # LO phase
        self.lp = IIRFilter.lowpass(0.5*self.intermediate_upsample/self.channel.upsample_factor)

    def process(self):
        channel = self.channel
        baseRate = rates.L_rate(0xb)
        for datagram, in self.input():
            octets = np.frombuffer(datagram, np.uint8)
            # prepare header and payload bits
            rateEncoding = 0xb
            rate = rates.L_rate(rateEncoding)
            data_bits = (octets[:,None] >> np.arange(8)[None,:]).flatten() & 1
            data_bits = np.r_[np.zeros(16, int), data_bits, FCS.compute(data_bits)]
            SIGNAL_bits = ((rateEncoding | ((octets.size+4) << 5)) >> np.arange(18)) & 1
            SIGNAL_bits[-1] = SIGNAL_bits.sum() & 1
            # OFDM modulation
            scrambler_state = np.random.randint(1,127)
            parts = (subcarriersFromBits(SIGNAL_bits, baseRate, 0   ),
                     subcarriersFromBits(data_bits,   rate,     scrambler_state))
            oversample = self.intermediate_upsample
            output = ofdm.L.encode(parts, oversample)
            # inter-frame space
            output = np.r_[output, np.zeros(round(IFS * channel.Fs*oversample/channel.upsample_factor))]
            # upsample
            output = np.vstack((output, np.zeros(((channel.upsample_factor // oversample)-1, output.size), output.dtype)))
            output = output.T.flatten()*(channel.upsample_factor / oversample)
            output = self.lp(output)
            # modulation and pre-emphasis
            Omega = 2*np.pi*channel.Fc/channel.Fs
            output = (output * np.exp(1j* Omega * np.r_[self.k:self.k+output.size])).real
            self.k += output.size
            for i in range(preemphasisOrder):
                output = np.diff(np.r_[output,0])
            output *= abs(np.exp(1j*Omega)-1)**-preemphasisOrder
            output /= abs(output).max()
            # stereo beamforming reduction
            delay = np.zeros(int(stereoDelay*channel.Fs))
            frames = np.vstack((np.r_[delay, output], np.r_[output, delay])).T
            self.output((frames,))
