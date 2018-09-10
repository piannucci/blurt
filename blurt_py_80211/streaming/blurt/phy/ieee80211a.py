import warnings
import collections
import itertools
import queue
import typing
import numpy as np
from . import kernels
from ..graph import Output, Input, Block, OverrunWarning
from .iir import IIRFilter

Channel = collections.namedtuple('Channel', ['Fs', 'Fc', 'upsample_factor'])

stereoDelay = .005
preemphasisOrder = 0
IFS = 2.5 * 80 * 8 / 96e3 # inter-frame space
nfft = 64
ncp = 16
sts_freq = np.zeros(64, complex)
sts_freq[[4, 8, 12, 16, 20, 24, -24, -20, -16, -12, -8, -4]] = \
    np.array([-1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1]) * (13./6.)**.5 * (1+1j)
lts_freq = np.array([0, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,
                     1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,
                     1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1])
ts_reps = 2
dataSubcarriers = np.r_[-26:-21,-20:-7,-6:0,1:7,8:21,22:27]
pilotSubcarriers = np.array([-21,-7,7,21])
pilotTemplate = np.array([1,1,1,-1])

############################ Scrambler ############################

scrambler = np.zeros((128,127), np.uint8)
s = np.arange(128, dtype=np.uint8)
for i in range(127):
    s = np.uint8((s << 1) ^ (1 & ((s >> 3) ^ (s >> 6))))
    scrambler[:,i] = s & 1

############################ CRC ############################

def CRC(x):
    a = np.r_[np.r_[np.zeros(x.size, int), np.ones(32, int)] ^ \
              np.r_[np.zeros(32, int), x[::-1]], np.zeros(-x.size%16, int)]
    a = (a.reshape(-1, 16) << np.arange(16)).sum(1)[::-1]
    return kernels.crc(a, lut)

lut = np.arange(1<<16, dtype=np.uint64) << 32
for i in range(47,31,-1):
    lut ^= (0x104c11db7 << (i-32)) * ((lut >> i) & 1)
lut = lut.astype(np.uint32)

############################ CC ############################

def mul(a, b):
    return np.bitwise_xor.reduce(a[:,None,None] * (b[:,None] & (1<<np.arange(7))), -1)

output_map = 1 & (mul(np.array([109, 79]), np.arange(128)) >> 6)
output_map_soft = output_map * 2 - 1

def encode(y):
    return kernels.encode(y, output_map)

def decode(llr):
    N = llr.size//2
    return kernels.decode(N, (llr[:N*2].reshape(-1,2,1)*output_map_soft).sum(1))

############################ Rates ############################

class Rate:
    def __init__(self, Nbpsc, ratio):
        self.Nbpsc = Nbpsc
        if Nbpsc == 1:
            self.symbols = np.array([-1,1])
        else:
            n = Nbpsc//2
            grayRevCode = sum(((np.arange(1<<n) >> i) & 1) << (n-1-i) for i in range(n))
            grayRevCode ^= grayRevCode >> 1
            grayRevCode ^= grayRevCode >> 2
            symbols = (2*grayRevCode+1-(1<<n)) * (1.5 / ((1<<Nbpsc) - 1))**.5
            self.symbols = np.tile(symbols, 1<<n) + 1j*np.repeat(symbols, 1<<n)
        self.puncturingMatrix = np.bool_({(1,2):[1,1], (2,3):[1,1,1,0], (3,4):[1,1,1,0,0,1],
                                          (5,6):[1,1,1,0,0,1,1,0,0,1]}[ratio])
        self.ratio = ratio
    def depuncture(self, y):
        output_size = (y.size + self.ratio[1]-1) // self.ratio[1] * self.ratio[0] * 2
        output = np.zeros(output_size, y.dtype)
        output[np.resize(self.puncturingMatrix, output.size)] = y
        return output
    def demap(self, y, dispersion):
        n = self.Nbpsc
        squared_distance = np.abs(self.symbols - y.flatten()[:,None])**2
        ll = -np.log(np.pi * dispersion) - squared_distance / dispersion
        ll -= np.logaddexp.reduce(ll, 1)[:,None]
        j = np.arange(1<<n)
        llr = np.zeros((y.size, n), int)
        for i in range(n):
            llr[:,i] = 10 * (np.logaddexp.reduce(ll[:,0 != (j & (1<<i))], 1) - \
                             np.logaddexp.reduce(ll[:,0 == (j & (1<<i))], 1))
        return np.clip(llr, -1e4, 1e4)

rates = {0xb: Rate(1, (1,2)), 0xf: Rate(2, (1,2)), 0xa: Rate(2, (3,4)), 0xe: Rate(4, (1,2)),
         0x9: Rate(4, (3,4)), 0xd: Rate(6, (2,3)), 0x8: Rate(6, (3,4)), 0xc: Rate(6, (5,6))}

############################ OFDM ############################

Nsc = dataSubcarriers.size
Nsc_used = dataSubcarriers.size + pilotSubcarriers.size
N_sts_period = nfft // 4
N_sts_samples = ts_reps * (ncp + nfft)
N_sts_reps = N_sts_samples // N_sts_period
N_training_samples = N_sts_samples + ts_reps * (ncp + nfft) + 8

def interleave(input, Ncbps, Nbpsc, reverse=False):
    s = max(Nbpsc//2, 1)
    j = np.arange(Ncbps)
    if reverse:
        i = (Ncbps//16) * (j%16) + (j//16)
        p = s*(i//s) + (i + Ncbps - (16*i//Ncbps)) % s
    else:
        i = s*(j//s) + (j + (16*j//Ncbps)) % s
        p = 16*i - (Ncbps - 1) * (16*i//Ncbps)
    return input[...,p].flatten()

class Autocorrelator:
    def __init__(self, nChannelsPerFrame):
        self.y_hist = np.zeros((0, nChannelsPerFrame))
        self.results = []
        self.nChannelsPerFrame = nChannelsPerFrame
    def feed(self, y):
        y = np.r_[self.y_hist, y]
        count_needed = y.shape[0] // N_sts_period * N_sts_period
        count_consumed = count_needed - N_sts_reps * N_sts_period
        if count_consumed <= 0:
            self.y_hist = y
        else:
            self.y_hist = y[count_consumed:]
            y = y[:count_needed].reshape(-1, N_sts_period, self.nChannelsPerFrame)
            corr_sum = np.abs((y[:-1].conj() * y[1:]).sum(1)).cumsum(0)
            self.results.append((corr_sum[N_sts_reps-1:] - corr_sum[:-N_sts_reps+1]).mean(-1))
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
        pilot = (sym[pilotSubcarriers]*pilotTemplate).sum() * (1.-2.*scrambler[0x7F,j%127])
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
    def __init__(self, i, nChannelsPerFrame):
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
    def feed(self, sequence, k):
        if self.result is not None:
            return
        if k < self.start:
            sequence = sequence[self.start-k:]
        self.y[self.size:self.size+sequence.shape[0]] = sequence[:self.y.shape[0]-self.size]
        self.size += sequence.shape[0]
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
                lsig_bits = decode(interleave(lsig.real, Nsc, 1, True))
                if not int(lsig_bits.sum()) & 1 == 0:
                    self.result = ()
                    return
                lsig_bits = (lsig_bits[:18] << np.arange(18)).sum()
                if not lsig_bits & 0xF in rates:
                    self.result = ()
                    return
                self.rate = rates[lsig_bits & 0xF]
                length_octets = (lsig_bits >> 5) & 0xFFF
                if length_octets > self.mtu:
                    self.result = ()
                    return
                self.length_coded_bits = (length_octets*8 + 16+6)*2
                self.Ncbps = Nsc * self.rate.Nbpsc
                self.length_symbols = int((self.length_coded_bits+self.Ncbps-1) // self.Ncbps)
                plcp_coded_bits = interleave(encode((lsig_bits >> np.arange(24)) & 1), Nsc, 1)
                self.dispersion = (lsig-(plcp_coded_bits*2.-1.)).var()
                self.j = 1
            else:
                return
        if j_valid < self.length_symbols + 1:
            return
        syms = np.array(list(itertools.islice(self.syms, self.length_symbols)))
        demapped_bits = self.rate.demap(syms, self.dispersion).reshape(-1, self.Ncbps)
        deinterleaved_bits = interleave(demapped_bits, self.Ncbps, self.rate.Nbpsc, True)
        llr = self.rate.depuncture(deinterleaved_bits)[:self.length_coded_bits]
        output_bits = (decode(llr) ^ np.resize(scrambler[0x5d], llr.size//2))[16:-6]
        if not CRC(output_bits) == 0xc704dd7b:
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

    def start(self):
        self.autocorrelator = Autocorrelator(self.nChannelsPerFrame)
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
                d = OneShotDecoder(peak * N_sts_period + 16, self.nChannelsPerFrame)
                k = self.k_lookback
                for y_old in self.lookback:
                    d.feed(y_old, k)
                    k += y_old.shape[0]
                self.decoders.append(d)
            self.lookback.append(y)
            for d in self.decoders:
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
    Ncbps = Nsc * rate.Nbpsc
    Nbps = Ncbps * rate.ratio[0] // rate.ratio[1]
    pad_bits = 6 + -(bits.size + 6) % Nbps
    scrambled = np.r_[bits, np.zeros(pad_bits, int)] ^ \
                np.resize(scrambler[scramblerState], bits.size + pad_bits)
    scrambled[bits.size:bits.size+6] = 0
    punctured = encode(scrambled)[np.resize(rate.puncturingMatrix, scrambled.size*2)]
    interleaved = interleave(punctured.reshape(-1, Ncbps), Nsc * rate.Nbpsc, rate.Nbpsc)
    grouped = (interleaved.reshape(-1, rate.Nbpsc) << np.arange(rate.Nbpsc)).sum(1)
    return rate.symbols[grouped].reshape(-1, Nsc)

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
        baseRate = rates[0xb]
        for datagram, in self.input():
            octets = np.frombuffer(datagram, np.uint8)
            # prepare header and payload bits
            rateEncoding = 0xb
            rate = rates[rateEncoding]
            data_bits = (octets[:,None] >> np.arange(8)[None,:]).flatten() & 1
            data_bits = np.r_[np.zeros(16, int), data_bits,
                              (~CRC(data_bits) >> np.arange(32)[::-1]) & 1]
            plcp_bits = ((rateEncoding | ((octets.size+4) << 5)) >> np.arange(18)) & 1
            plcp_bits[-1] = plcp_bits.sum() & 1
            # OFDM modulation
            subcarriers = np.vstack((subcarriersFromBits(plcp_bits, baseRate, 0   ),
                                     subcarriersFromBits(data_bits, rate,     0x5d)))
            pilotPolarity = np.resize(scrambler[0x7F], subcarriers.shape[0])
            symbols = np.zeros((subcarriers.shape[0],nfft), complex)
            symbols[:,dataSubcarriers] = subcarriers
            symbols[:,pilotSubcarriers] = pilotTemplate * (1. - 2.*pilotPolarity)[:,None]
            ts_tile_shape = (ncp*ts_reps+nfft-1)//nfft + ts_reps + 1
            symbols_tile_shape = (1, ncp//nfft + 3)
            intermediate_upsample = self.intermediate_upsample
            sts_freq_ius = np.r_[sts_freq[:nfft//2], np.zeros(nfft*(intermediate_upsample-1)), sts_freq[nfft//2:]]
            lts_freq_ius = np.r_[lts_freq[:nfft//2], np.zeros(nfft*(intermediate_upsample-1)), lts_freq[nfft//2:]]
            symbols_ius  = np.concatenate((symbols[:,:nfft//2], np.zeros((symbols.shape[0], nfft*(intermediate_upsample-1))), symbols[:,nfft//2:]), axis=1)
            sts_time = np.tile(np.fft.ifft(sts_freq_ius)*intermediate_upsample, ts_tile_shape)
            lts_time = np.tile(np.fft.ifft(lts_freq_ius)*intermediate_upsample, ts_tile_shape)
            symbols  = np.tile(np.fft.ifft(symbols_ius)*intermediate_upsample, symbols_tile_shape)
            sts_time = sts_time[(-ncp*ts_reps%nfft)*intermediate_upsample:(-nfft+1)*intermediate_upsample]
            lts_time = lts_time[(-ncp*ts_reps%nfft)*intermediate_upsample:(-nfft+1)*intermediate_upsample]
            symbols  = symbols[:,(-ncp%nfft)*intermediate_upsample:(-nfft+1)*intermediate_upsample]
            # temporal smoothing
            subsequences = [sts_time, lts_time] + list(symbols)
            output = np.zeros(sum(map(len, subsequences)) - (len(subsequences) - 1) * intermediate_upsample, complex)
            i = 0
            ramp = np.linspace(0,1,2+intermediate_upsample)[1:-1]
            for x in subsequences:
                weight = np.ones(x.size)
                weight[-1:-intermediate_upsample-1:-1] = weight[:intermediate_upsample] = ramp
                output[i:i+len(x)] += weight * x
                i += len(x) - intermediate_upsample
            # inter-frame space
            output = np.r_[output, np.zeros(round(IFS * channel.Fs*intermediate_upsample/channel.upsample_factor))]
            # upsample
            output = np.vstack((output, np.zeros(((channel.upsample_factor // intermediate_upsample)-1, output.size), output.dtype)))
            output = output.T.flatten()*(channel.upsample_factor / intermediate_upsample)
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