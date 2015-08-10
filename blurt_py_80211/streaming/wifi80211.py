#!/usr/bin/env python
import numpy as np, weave, itertools, queue
import iir, audio

############################ Parameters ############################

Fs, Fc, upsample_factor = 96e3, 20e3, 32

mtu = 150

audioInputFrameSize = 2048
audioInputQueueLength = .33 # seconds
packetQueueDepth = 64

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

stereoDelay = .005
preemphasisOrder = 13

############################ Scrambler ############################

scrambler_y = np.zeros((128,127), np.uint8)
s = np.arange(128, dtype=np.uint8)
for i in range(127):
    s = np.uint8((s << 1) ^ (1 & ((s >> 3) ^ (s >> 6))))
    scrambler_y[:,i] = s & 1

############################ CRC ############################

def CRC(x):
    a = np.r_[np.r_[np.zeros(x.size, int), np.ones(32, int)] ^ \
              np.r_[np.zeros(32, int), x[::-1]], np.zeros(-x.size%16, int)]
    a = (a.reshape(-1, 16) << np.arange(16)).sum(1)[::-1]
    return weave.inline("""
    uint32_t r = 0, A=a.extent(blitz::firstDim);
    for (int i=0; i<A; i++)
        r = (r << 16) ^ a(i) ^ lut(r >> 16);
    return_val = r;
    """, ['a','lut'], type_converters=weave.converters.blitz)

lut = np.arange(1<<16, dtype=np.uint64) << 32
for i in range(47,31,-1):
    lut ^= (0x104c11db7 << (i-32)) * ((lut >> i) & 1)
lut = lut.astype(np.uint32)

############################ CC ############################

def mul(a, b):
    c = 0
    for i in range(7):
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
    output = np.empty(y.size*2, np.uint8)
    weave.inline("""
    int sh = 0, N = y.extent(blitz::firstDim);
    for (int i=0; i<N; i++) {
        sh = (sh>>1) ^ ((int)y(i) << 6);
        output(2*i+0) = output_map_1(sh);
        output(2*i+1) = output_map_2(sh);
    }
    """, ['y','output','output_map_1','output_map_2'],
         type_converters=weave.converters.blitz)
    return output

def decode(llr):
    N = llr.size//2
    x = llr[0:2*N:2,None]*output_map_1_soft + llr[1:2*N:2,None]*output_map_2_soft
    msg = np.empty(N, np.uint8)
    weave.inline("""
    const int M = 128;
    int64_t *cost = new int64_t [M*2];
    int64_t *scores = new int64_t [M];
    uint8_t bt[N][M];
    for (int i=0; i<M; i++)
        scores[i] = 0;
    for (int k=0; k<N; k++) {
        for (int i=0; i<M; i++) {
            cost[2*i+0] = scores[state_inv_map(i, 0)] + x(k, i);
            cost[2*i+1] = scores[state_inv_map(i, 1)] + x(k, i);
        }
        for (int i=0; i<M; i++) {
            int a = cost[2*i+0];
            int b = cost[2*i+1];
            bt[k][i] = (a<b) ? 1 : 0;
            scores[i] = (a<b) ? b : a;
        }
    }
    int i = (scores[0] < scores[1]) ? 1 : 0;
    for (int k=N-1; k>=0; k--) {
        int j = bt[k][i];
        msg(k) = state_inv_map_tag(i,j);
        i = state_inv_map(i,j);
    }
    delete [] cost;
    delete [] scores;
    """, ['N','state_inv_map','x','state_inv_map_tag','msg'],
         type_converters=weave.converters.blitz)
    return msg

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
            stripes_shape = (2*l+1, count_needed-2*l)
            stripes_strides = (y.strides[0],)*2
            stripes = np.lib.stride_tricks.as_strided(y, stripes_shape, stripes_strides)
            yield from (stripes.argmax(0) == l).nonzero()[0] + i
            i += count_consumed

def estimate_cfo(y, overlap, span):
    return np.angle((y[:-overlap].conj() * y[overlap:]).sum()) / span

def remove_cfo(y, k, theta):
    return y * np.exp(-1j*theta*np.r_[k:k+y.size])

def train(y):
    i = 0
    theta = estimate_cfo(y[i:i+N_sts_samples], N_sts_period, N_sts_period)
    i += N_sts_samples + ncp*ts_reps
    lts = np.fft.fft(remove_cfo(y[i:i+nfft*ts_reps], i, theta).reshape(-1, nfft), axis=1)
    theta += estimate_cfo(lts * (lts_freq != 0), 1, nfft)
    def wienerFilter(i):
        lts = np.fft.fft(remove_cfo(y[i:i+nfft*ts_reps], i, theta).reshape(-1, nfft), axis=1)
        Y = lts.mean(0)
        S_Y = (np.abs(lts)**2).mean(0)
        G = Y.conj()*lts_freq / S_Y
        snr = 1./(G*lts - lts_freq).var()
        return snr, i + nfft*ts_reps, G
    snr, i, G = max(map(wienerFilter, range(i-8, i+8)))
    var_input = y.var()
    var_n = var_input / (snr * Nsc_used / Nsc + 1)
    var_x = var_input - var_n
    var_y = 2*var_n*var_x + var_n**2
    uncertainty = np.arctan(var_y**.5 / var_x) / nfft**.5
    var_ni = var_x/Nsc_used/snr
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
        sym = np.fft.fft(remove_cfo(y[ncp:], i+ncp, theta_cfo)) * G
        i += nfft+ncp
        pilot = (sym[pilotSubcarriers]*pilotTemplate).sum() * (1.-2.*scrambler_y[0x7F,j%127])
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

class StreamIndexer(object):
    def __init__(self, source, bufferSize=96000*10):
        self.bufferSize = bufferSize
        self.source = source
        y = next(self.source)
        self.dtype = y.dtype
        self.y = np.empty(self.bufferSize, self.dtype)
        self.y[:y.size] = y
        self.k = 0
        self.size = y.size
    def grow(self, i):
        if i - self.k > self.bufferSize:
            raise ValueError('stream buffer size limit exceeded')
        # load indices < i
        while self.k + self.size < i:
            y = next(self.source)
            start = (self.k + self.size) % self.bufferSize
            stop = start + y.size
            if stop <= self.bufferSize:
                self.y[start:stop] = y
            else:
                self.y[start:] = y[:self.bufferSize-start]
                self.y[:stop-self.bufferSize] = y[self.bufferSize-start:]
            self.size += y.size
    def shrink(self, i):
        # discard indices < i
        self.grow(i)
        delta = max(i-self.k, 0)
        self.k += delta
        self.size -= delta
    def __getitem__(self, sl):
        if isinstance(sl, slice):
            start = sl.start
            if start is None:
                start = 0
            if start < self.k:
                raise IndexError('stream index out of range')
            stop = sl.stop
            if stop is None:
                raise IndexError('stream index out of range')
            step = sl.step
            if step is None:
                step = 1
            if step < 0:
                raise ValueError('slice step must be positive')
            self.grow(stop)
            size = stop - start
            start %= self.bufferSize
            stop = start + size
            if stop <= self.bufferSize:
                return self.y[start:stop:step]
            else:
                return np.r_[self.y[start:], self.y[:stop-self.bufferSize]][::step]
        else:
            if sl < self.k:
                raise IndexError('stream index out of range')
            self.grow(sl+1)
            return self.y[sl % self.bufferSize]

def downconvert(source):
    i = 0
    lp = iir.lowpass(.45/upsample_factor, order=6)
    for y in source:
        smoothed = lp(y * np.exp(-1j*2*np.pi*Fc/Fs * np.r_[i:i+y.size]))
        yield smoothed[-i%upsample_factor::upsample_factor]
        i += y.size

def decodeBlurt(source):
    j = 0 # ignore upto cursor
    s1,s2 = itertools.tee(downconvert(source))
    c = StreamIndexer(s2)
    for k in peakDetect(autocorrelate(s1), 25):
        i = k * N_sts_period + 16
        if i < j:
            continue
        c.shrink(i)
        training_data, training_advance = train(c[i:i+N_training_samples])
        i += training_advance
        syms = (c[i+j*(nfft+ncp):i+(j+1)*(nfft+ncp)] for j in itertools.count())
        syms = decodeOFDM(syms, training_advance, training_data)
        lsig = next(syms)
        lsig_bits = decode(interleave(lsig.real, Nsc, 1, True))
        if not int(lsig_bits.sum()) & 1 == 0:
            continue
        lsig_bits = (lsig_bits[:18] << np.arange(18)).sum()
        if not lsig_bits & 0xF in rates:
            continue
        rate = rates[lsig_bits & 0xF]
        length_octets = (lsig_bits >> 5) & 0xFFF
        if length_octets > mtu:
            continue
        length_coded_bits = (length_octets*8 + 16+6)*2
        Ncbps = Nsc * rate.Nbpsc
        length_symbols = int((length_coded_bits+Ncbps-1) // Ncbps)
        plcp_coded_bits = interleave(encode((lsig_bits >> np.arange(24)) & 1), Nsc, 1)
        dispersion = (lsig-(plcp_coded_bits*2.-1.)).var()
        syms = np.array(list(itertools.islice(syms, length_symbols)))
        demapped_bits = rate.demap(syms, dispersion).reshape(-1, Ncbps)
        deinterleaved_bits = interleave(demapped_bits, Ncbps, rate.Nbpsc, True)
        llr = rate.depuncture(deinterleaved_bits)[:length_coded_bits]
        output_bits = (decode(llr) ^ np.resize(scrambler_y[0x5d], llr.size//2))[16:-6]
        if not CRC(output_bits) == 0xc704dd7b:
            continue
        j = i + (nfft+ncp)*(length_symbols+1)
        output_bytes = (output_bits[:-32].reshape(-1, 8) << np.arange(8)).sum(1)
        yield output_bytes.astype(np.uint8).tostring(), 10*np.log10(1/dispersion)

def subcarriersFromBits(bits, rate, scramblerState):
    Ncbps = Nsc * rate.Nbpsc
    Nbps = Ncbps * rate.ratio[0] // rate.ratio[1]
    pad_bits = 6 + -(bits.size + 6) % Nbps
    scrambled = np.r_[bits, np.zeros(pad_bits, int)] ^ \
                np.resize(scrambler_y[scramblerState], bits.size + pad_bits)
    scrambled[bits.size:bits.size+6] = 0
    punctured = encode(scrambled)[np.resize(rate.puncturingMatrix, scrambled.size*2)]
    interleaved = interleave(punctured.reshape(-1, Ncbps), Nsc * rate.Nbpsc, rate.Nbpsc)
    grouped = (interleaved.reshape(-1, rate.Nbpsc) << np.arange(rate.Nbpsc)).sum(1)
    return rate.symbols[grouped].reshape(-1, Nsc)

def encodeBlurt(source):
    cutoff = (Nsc_used/2 + .5)/nfft
    lp1 = iir.lowpass(cutoff/upsample_factor, order=6, method='Ch', ripple=-.021)
    lp2 = lp1.copy()
    baseRate = rates[0xb]
    k = 0
    for octets in source:
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
        pilotPolarity = np.resize(scrambler_y[0x7F], subcarriers.shape[0])
        symbols = np.zeros((subcarriers.shape[0],nfft), complex)
        symbols[:,dataSubcarriers] = subcarriers
        symbols[:,pilotSubcarriers] = pilotTemplate * (1. - 2.*pilotPolarity)[:,None]
        ts_tile_shape = (ncp*ts_reps+nfft-1)//nfft + ts_reps + 1
        symbols_tile_shape = (1, (ncp+nfft-1)//nfft + 1 + 1)
        sts_time = np.tile(np.fft.ifft(sts_freq), ts_tile_shape)[-ncp*ts_reps%nfft:-nfft+1]
        lts_time = np.tile(np.fft.ifft(lts_freq), ts_tile_shape)[-ncp*ts_reps%nfft:-nfft+1]
        symbols  = np.tile(np.fft.ifft(symbols ), symbols_tile_shape)[:,-ncp%nfft:-nfft+1]
        # temporal smoothing
        subsequences = [sts_time, lts_time] + list(symbols)
        output = np.zeros(sum(map(len, subsequences)) - len(subsequences) + 1, complex)
        i = 0
        for x in subsequences:
            j = i + len(x)-1
            output[i] += .5*x[0]
            output[i+1:j] += x[1:-1]
            output[j] += .5*x[-1]
            i = j
        output = np.vstack((output, np.zeros((upsample_factor-1, output.size), output.dtype)))
        output = output.T.flatten()*upsample_factor
        output = lp2(lp1(np.r_[np.zeros(200), output, np.zeros(200)]))
        # modulation and pre-emphasis
        output = (output * np.exp(1j*2*np.pi*Fc/Fs * np.r_[k:k+output.size])).real
        k += output.size
        for i in range(preemphasisOrder):
            output = np.diff(np.r_[output,0])
        output *= abs(np.exp(1j*2*np.pi*Fc/Fs)-1)**-preemphasisOrder
        # stereo beamforming reduction
        delay = np.zeros(stereoDelay*Fs)
        yield np.vstack((np.r_[delay, output], np.r_[output, delay])).T

############################ Audio ############################

class QueueStream(audio.stream.StreamArray):
    def __init__(self):
        super().__init__()
        self.in_queue = queue.Queue(int(np.ceil(audioInputQueueLength * Fs /
                                                audioInputFrameSize)))
        self.inBufSize = audioInputFrameSize
        self.vu = 1e-10
    def consume(self, sequence):
        try:
            self.in_queue.put_nowait(sequence)
        except queue.Full:
            print('QueueStream overrun')
        except Exception as e:
            print('QueueStream exception %s' % repr(e))
        self.vu = (sequence**2).max()
    def __iter__(self):
        while True:
            yield self.in_queue.get()

class IteratorStream(audio.stream.ThreadedStream):
    def __init__(self, source):
        super().__init__(channels=2)
        self.source = source
    def thread_produce(self):
        return next(self.source)

if __name__ == '__main__':
    import sys, time, traceback, threading
    if len(sys.argv) > 1 and sys.argv[1] == '--tx':
        def packets():
            for i in itertools.count():
                yield np.r_[np.random.random_integers(ord('A'),ord('Z'),26),
                            np.fromstring('%06d' % i, np.uint8)]
        audio.play(IteratorStream(encodeBlurt(packets())), Fs)
    elif len(sys.argv) > 1 and sys.argv[1] == '--rx':
        ap = audio.AudioInterface(None)
        qs = QueueStream()
        q = queue.Queue(packetQueueDepth)
        def decoderThread():
            try:
                for packet in decodeBlurt(qs):
                    q.put(packet)
            except Exception as e:
                print('decoderThread exception')
                traceback.print_exc()
        t = threading.Thread(target=decoderThread, daemon=True)
        t.start()
        try:
            ap.record(qs, Fs)
            while t.is_alive():
                try:
                    payload, lsnr = q.get(timeout=.01)
                    sys.stdout.write('\r\x1b[2K')
                    print(payload, lsnr)
                except queue.Empty:
                    pass
                sys.stdout.write('\r\x1b[2K' + '.' * int(max(0, 80 + 10*np.log10(qs.vu))))
                sys.stdout.flush()
        except KeyboardInterrupt:
            pass
        sys.stdout.write('\r\x1b[2K')
        ap.stop()
