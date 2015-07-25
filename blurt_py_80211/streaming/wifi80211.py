#!/usr/bin/env python
import numpy as np, scipy.linalg, scipy.weave, collections, itertools
import iir

Fs, Fc, upsample_factor = 96e3, 17e3, 32

nfft = 64
ncp = 64 # 16
lts_freq = np.array([0, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1])
ts_reps = 6 # 2
dataSubcarriers = np.r_[-26:-21,-20:-7,-6:0,1:7,8:21,22:27]
pilotSubcarriers = np.array([-21,-7,7,21])
Nsc = dataSubcarriers.size
Nsc_used = dataSubcarriers.size + pilotSubcarriers.size
N_sts_period = nfft / 4
N_sts_samples = ts_reps * (ncp + nfft)
N_sts_reps = N_sts_samples // N_sts_period
N_training_samples = N_sts_samples + ts_reps * (ncp + nfft) + 8
pilotTemplate = np.array([1,1,1,-1])

############################ Scrambler ############################

scrambler_y = np.zeros((128,127), np.uint8)
s = np.arange(128, dtype=np.uint8)
for i in range(127):
    s = np.uint8((s << 1) ^ (1 & ((s >> 3) ^ (s >> 6))))
    scrambler_y[:,i] = s & 1

############################ CRC ############################

def shiftout(x, n):
    return (x >> np.arange(n)[::-1]) & 1

def remainder(a):
    a = a[::-1]
    if a.size % 16:
        a = np.r_[a, np.zeros(-a.size%16, int)]
    a = (a.reshape(-1, 16) << np.arange(16)).sum(1)[::-1]
    code = """
    uint32_t r = 0, A=a.extent(blitz::firstDim);
    for (int i=0; i<A; i++)
        r = ((r << 16) & 0xffffffff) ^ a(i) ^ lut(r >> 16);
    return_val = r;
    """
    return shiftout(scipy.weave.inline(code, ['a', 'lut'], type_converters=scipy.weave.converters.blitz), 32)

def FCS(calculationFields):
    return 1 & ~remainder(np.r_[np.ones(32, int), np.zeros(calculationFields.size, int)] ^ np.r_[calculationFields, np.zeros(32, int)])

def checkFCS(frame):
    return (remainder(np.r_[np.ones(32, int), np.zeros(frame.size, int)] ^ np.r_[frame, np.zeros(32, int)]) == correct_remainder).all()

G = shiftout(0x104c11db7, 33).astype(bool)
correct_remainder = shiftout(0xc704dd7b, 32).astype(bool)
lut,a,b,c = np.empty(1<<16, np.uint32), np.empty(48, np.uint8), np.arange(32), np.arange(16)[::-1]
d = a[:-33:-1]
for j in xrange(1<<16):
    a[:16] = (j >> c) & 1
    a[16:] = 0
    for i in xrange(16):
        if a[i]:
            a[i:i+33] ^= G
    lut[j] = (d << b).sum()

############################ CC ############################

def mul(a, b):
    c = 0
    for i in xrange(7):
        c ^= (a * np.array((b>>i)&1, bool)) << i
    return c

states            = np.arange(128, dtype=np.uint8)
output_map_1      = np.uint8(1 & (mul(109, states) >> 6))
output_map_2      = np.uint8(1 & (mul( 79, states) >> 6))
state_inv_map     = ((states << 1) & 127)[:,None] + np.array([0,1])
state_inv_map_tag = np.tile(states[:,None] >> 6, (1,2))
output_map_1_soft = output_map_1.astype(int) * 2 - 1
output_map_2_soft = output_map_2.astype(int) * 2 - 1

def encode(y):
    output, x, sh = np.empty(y.size*2, np.uint8), y << 6, 0
    for i in xrange(x.size):
        sh = (sh>>1) ^ x[i]
        output[2*i+0] = output_map_1[sh]
        output[2*i+1] = output_map_2[sh]
    return output

def decode(llr):
    N = llr.size/2
    bt = np.empty((N, 128), np.uint8);
    x = llr[0:2*N:2,None]*output_map_1_soft + llr[1:2*N:2,None]*output_map_2_soft
    msg = np.empty(N, np.uint8)
    code = """
    const int M = 128;
    int64_t *cost = new int64_t [M*2];
    int64_t *scores = new int64_t [M];
    for (int i=0; i<M; i++)
        scores[i] = 0;
    for (int k=0; k<N; k++) {
        for (int i=0; i<M; i++) {
            cost[2*i+0] = scores[state_inv_map(i, 0)] + x(k, i);
            cost[2*i+1] = scores[state_inv_map(i, 1)] + x(k, i);
        }
        for (int i=0; i<M; i++) {
            int a, b;
            a = cost[2*i+0];
            b = cost[2*i+1];
            bt(k, i) = (a<b) ? 1 : 0;
            scores[i] = (a<b) ? b : a;
        }
    }
    int i = (scores[0] < scores[1]) ? 1 : 0;
    for (int k=N-1; k>=0; k--) {
        int j = bt(k, i);
        msg(k) = state_inv_map_tag(i,j);
        i = state_inv_map(i,j);
    }
    delete [] cost;
    delete [] scores;
    """
    scipy.weave.inline(code, ['N','state_inv_map', 'x','bt','state_inv_map_tag', 'msg'], type_converters=scipy.weave.converters.blitz)
    return msg

############################ Rates ############################

def grayRevToBinary(n):
    x = np.arange(1<<n)
    y = np.zeros_like(x)
    for i in range(n):
        y <<= 1
        y |= x&1
        x >>= 1
    shift = 1
    while shift<n:
        y ^= y >> shift
        shift<<=1
    return y

class Rate:
    def __init__(self, Nbpsc, ratio):
        self.Nbpsc = Nbpsc
        if Nbpsc == 1:
            self.symbols = np.array([-1,1])
        else:
            n = Nbpsc/2
            symbols = (2*grayRevToBinary(n)+1-(1<<n)) * (1.5 / ((1<<Nbpsc) - 1))**.5
            self.symbols = np.tile(symbols, 1<<n) + 1j*np.repeat(symbols, 1<<n)
        self.puncturingMatrix = {(1,2):np.array([1,1]), (2,3):np.array([1,1,1,0]), (3,4):np.array([1,1,1,0,0,1]), (5,6):np.array([1,1,1,0,0,1,1,0,0,1])}[ratio]
        self.Ncbps = Nsc * self.Nbpsc

    def depuncture(self, input):
        m = self.puncturingMatrix
        output = np.zeros(((input.size + m.sum() - 1) / m.sum()) * m.size, input.dtype)
        output[np.resize(m, output.size).astype(bool)] = input
        return output

    def demap(self, y, dispersion):
        n = self.Nbpsc
        squared_distance = np.abs(self.symbols - y[:,None])**2
        ll = -np.log(np.pi * dispersion) - squared_distance / dispersion
        ll -= np.logaddexp.reduce(ll, 1)[:,None]
        j = np.arange(1<<n)
        ll0 = np.zeros((y.size, n), float)
        ll1 = np.zeros((y.size, n), float)
        for i in xrange(n):
            idx0 = np.where(0 == (j & (1<<i)))[0]
            idx1 = np.where(j & (1<<i))[0]
            ll0[:,i] = np.logaddexp.reduce(ll[:,idx0], 1)
            ll1[:,i] = np.logaddexp.reduce(ll[:,idx1], 1)
        return np.int64(np.clip(10.*(ll1-ll0), -1e4, 1e4)).flatten()

rates = {0xb: Rate(1, (1,2)), 0xf: Rate(2, (1,2)), 0xa: Rate(2, (3,4)), 0xe: Rate(4, (1,2)), 0x9: Rate(4, (3,4)), 0xd: Rate(6, (2,3)), 0x8: Rate(6, (3,4)), 0xc: Rate(6, (5,6))}

############################ OFDM ############################

def interleave(input, Ncbps, Nbpsc, reverse=False):
    s = max(Nbpsc/2, 1)
    j = np.arange(Ncbps)
    if reverse:
        i = (Ncbps/16) * (j%16) + (j/16)
        p = s*(i/s) + (i + Ncbps - (16*i/Ncbps)) % s
    else:
        i = s*(j/s) + (j + (16*j/Ncbps)) % s
        p = 16*i - (Ncbps - 1) * (16*i/Ncbps)
    return input[...,p].flatten()

# y is a slice of the input stream
# k is the index of the first sample in y
# theta is the CFO in radians/sample
def remove_cfo(y, k, theta):
   return y * np.exp(-1j*theta*np.r_[k:k+y.size])

estimate_cfo = lambda arr, overlap, span: np.angle((arr[:-overlap].conj()*arr[overlap:]).sum())/span
TrainingData = collections.namedtuple('TrainingData', ['G', 'uncertainty', 'var_n', 'theta'])

def autocorrelate(source):
    y_hist = np.zeros(0)
    for y in source:
        y = np.r_[y_hist, y]
        count_needed = y.size // N_sts_period * N_sts_period
        count_consumed = count_needed - N_sts_reps * N_sts_period
        if count_consumed <= 0:
            y_hist = y
            continue
        else:
            y_hist = y[count_consumed:]
            y = y[:count_needed].reshape(-1, N_sts_period)
            corr_sum = np.abs((y[:-1].conj() * y[1:]).sum(1)).cumsum()
            yield corr_sum[N_sts_reps-1:] - corr_sum[:-N_sts_reps+1]

def peakDetect(source, l):
    y_hist = np.zeros(0)
    i = l
    for y in source:
        y = np.r_[y_hist, y]
        count_needed = y.size
        count_consumed = count_needed - 2*l
        if count_consumed <= 0:
            y_hist = y
            continue
        else:
            y_hist = y[count_consumed:]
            for j in (np.lib.stride_tricks.as_strided(y, (2*l+1,count_needed-2*l), (y.strides[0],)*2).argmax(0) == l).nonzero()[0]:
                yield j + i
            i += count_consumed

def stsDetect(source):
    for i in peakDetect(autocorrelate(source), 25):
        yield i * N_sts_period + 16

def train(y):
    i = 0
    theta = estimate_cfo(y[i:i+N_sts_samples], N_sts_period, N_sts_period)
    i += N_sts_samples + ncp*ts_reps
    theta += estimate_cfo(np.fft.fft(remove_cfo(y[i:i+nfft*ts_reps], i, theta).reshape(-1, nfft), axis=1) * (lts_freq != 0), 1, nfft)
    def wienerFilter(i):
        lts = np.fft.fft(remove_cfo(y[i:i+nfft*ts_reps], i, theta).reshape(-1, nfft), axis=1)
        Y = lts.mean(0)
        S_Y = (np.abs(lts)**2).mean(0)
        G = Y.conj()*lts_freq / S_Y
        snr = 1./(G*lts - lts_freq).var()
        return G, snr, i + nfft*ts_reps
    G, snr, i = max([wienerFilter(i+offset) for offset in np.arange(-8, 8)], key=lambda r:r[1])
    var_input = y.var()
    var_n = var_input / (snr * Nsc_used / Nsc + 1)
    var_x = var_input - var_n
    var_y = 2*var_n*var_x + var_n**2
    uncertainty = np.arctan(var_y**.5 / var_x) / nfft**.5
    var_ni = var_x/Nsc_used/snr
    return TrainingData(G, uncertainty, var_ni, theta), i, 10*np.log10(snr)

def ofdm_symbols(source, i, training_data):
    pilotPolarity = (1. - 2.*x for x in itertools.cycle(scrambler_y[0x7F]))
    sigma_noise = 4*training_data.var_n*.5
    sigma = sigma_noise + 4*np.sin(training_data.uncertainty)**2
    P, x, R = np.diag([sigma, sigma, training_data.uncertainty**2]), np.array([[4.,0.,0.]]).T, np.diag([sigma_noise, sigma_noise])
    Q = P * 0.1
    for y in source:
        sym = np.fft.fft(remove_cfo(y[ncp:], i+ncp, training_data.theta)) * training_data.G
        i += nfft+ncp
        pilot = np.sum(sym[pilotSubcarriers] * next(pilotPolarity) * pilotTemplate)
        re,im,theta = x[:,0]
        c, s = np.cos(theta), np.sin(theta)
        F = np.array([[c, -s, -s*re - c*im], [s,  c,  c*re - s*im], [0,  0,  1]])
        x[0,0] = c*re - s*im
        x[1,0] = c*im + s*re
        P = F.dot(P).dot(F.T) + Q
        S = P[:2,:2] + R
        K = np.linalg.solve(S, P[:2,:]).T
        x += K.dot(np.array([[pilot.real], [pilot.imag]]) - x[:2,:])
        P -= K.dot(P[:2,:])
        u = x[0,0] - x[1,0]*1j
        u /= np.abs(u)
        yield sym[dataSubcarriers] * u

def downconvert(source):
    i = 0
    lowpass = iir.lowpass(.8/upsample_factor, order=6, continuous=True, dtype=np.complex128)
    s = -1j*2*np.pi*Fc/Fs
    for y in source:
        yield lowpass(y * np.exp(s * np.arange(i,i+y.size)))[-i%upsample_factor::upsample_factor]
        i += y.size

def advance(source, y, k, i):
    while k+y.size <= i:
        k += y.size
        y = next(source)
    return y, k

# preconditions: 0 <= k <= i, y = source[k:k+y.size], length > 0
# postcondition: result = source[i:i+length]
def chunk(source, y, k, i, length):
    result = np.empty(length, y.dtype)
    while True:
        y, k = advance(source, y, k, i)
        result[:y.size-(i-k)] = y[i-k:i-k+length]
        while k+y.size < i+length:
            k += y.size
            y = next(source)
            result[k-i:k-i+y.size] = y[:length-(k-i)]
        yield result
        i += length

def decodeBlurt(source):
    j = 0 # ignore upto cursor
    k = 0 # next sample cursor
    y = np.empty(0)
    source,a = itertools.tee(downconvert(source))
    for i in stsDetect(a):
        if i < j:
            continue
        try:
            y, k = advance(source, y, k, i)
            source,a = itertools.tee(source)
            training_data, training_advance, lsnr = train(next(chunk(a, y, k, i, N_training_samples)))
            i += training_advance
            y, k = advance(source, y, k, i)
            source,a = itertools.tee(source)
            syms = ofdm_symbols(chunk(a, y, k, i, nfft+ncp), training_advance, training_data)
            lsig = next(syms)
            lsig_bits = decode(interleave(lsig.real, Nsc, 1, True))
            assert int(lsig_bits.sum()) & 1 == 0
            lsig_bits = (lsig_bits[:18] << np.arange(18)).sum()
            rate = rates[lsig_bits & 0xF]
            length_coded_bits = (((lsig_bits >> 5) & 0xFFF)*8 + 16+6)*2
            length_symbols = (length_coded_bits+rate.Ncbps-1) // rate.Ncbps
            plcp_coded_bits = interleave(encode(np.r_[(lsig_bits >> np.arange(18)) & 1, 0,0,0,0,0,0]), Nsc, 1).astype(int)
            dispersion = (lsig-(plcp_coded_bits*2-1)).var()
            demapped_bits = np.array([rate.demap(next(syms), dispersion) for _ in xrange(length_symbols)])
            llr = rate.depuncture(interleave(demapped_bits, rate.Ncbps, rate.Nbpsc, True))[:length_coded_bits]
            output_bits = (decode(llr) ^ np.resize(scrambler_y[0x5d], llr.size/2))[16:-6]
            assert checkFCS(output_bits)
            j = i + (nfft+ncp)*(length_symbols+1)
            yield (output_bits[:-32].reshape(-1, 8) << np.arange(8)).sum(1).astype(np.uint8).tostring(), 10*np.log10(1/dispersion)
        except:
            pass
