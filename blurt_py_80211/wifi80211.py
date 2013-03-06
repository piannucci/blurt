#!/usr/bin/env python
import numpy as np
import util, cc, ofdm, scrambler, interleaver, qam, crc
import audio.stream
import pylab as pl

code = cc.ConvolutionalCode()
ofdm = ofdm.OFDM(ofdm.LT)

class Rate:
    def __init__(self, encoding, (Nbpsc, constellation), (puncturingRatio, puncturingMatrix)):
        self.encoding = encoding
        self.Nbpsc = Nbpsc
        self.constellation = constellation
        self.puncturingRatio = puncturingRatio
        self.puncturingMatrix = puncturingMatrix
        self.Nbps = ofdm.format.Nsc * Nbpsc * puncturingRatio[0] / puncturingRatio[1]
        self.Ncbps = ofdm.format.Nsc * Nbpsc

class WiFi_802_11:
    def __init__(self):
        self.rates = [Rate(0xd, qam.bpsk , cc.puncturingSchedule[(1,2)]),
                      Rate(0xf, qam.qpsk , cc.puncturingSchedule[(1,2)]),
                      Rate(0x5, qam.qpsk , cc.puncturingSchedule[(3,4)]),
                      Rate(0x7, qam.qam16, cc.puncturingSchedule[(1,2)]),
                      Rate(0x9, qam.qam16, cc.puncturingSchedule[(3,4)]),
                      Rate(0xb, qam.qam64, cc.puncturingSchedule[(2,3)]),
                      Rate(0x1, qam.qam64, cc.puncturingSchedule[(3,4)]),
                      Rate(0x3, qam.qam64, cc.puncturingSchedule[(5,6)])]

    def plcp_bits(self, rate, octets):
        plcp_rate = util.rev(rate.encoding, 4)
        plcp = plcp_rate | (octets << 5)
        parity = (util.mul(plcp, 0x1FFFF) >> 16) & 1
        plcp |= parity << 17
        return util.shiftout(np.array([plcp]), 18)

    def subcarriersFromBits(self, bits, rate, scramblerState):
        # scrambled includes tail & padding
        scrambled = scrambler.scramble(bits, rate.Nbps, scramblerState=scramblerState)
        coded = code.encode(scrambled)
        punctured = code.puncture(coded, rate.puncturingMatrix)
        interleaved = interleaver.interleave(punctured, rate.Ncbps, rate.Nbpsc)
        return qam.encode(interleaved, rate, ofdm.format.Nsc)

    def encode(self, input_octets, rate_index):
        service_bits = np.zeros(16, int)
        data_bits = util.shiftout(input_octets, 8)
        data_bits = np.r_[service_bits, data_bits, crc.FCS(data_bits)]
        signal_subcarriers = self.subcarriersFromBits(self.plcp_bits(self.rates[rate_index], input_octets.size+4), self.rates[0], 0)
        data_subcarriers = self.subcarriersFromBits(data_bits, self.rates[rate_index], 0x5d)
        return ofdm.encode(signal_subcarriers, data_subcarriers)

    def train(self, input, lsnr):
        """
        Recover OFDM timing and frequency-domain deconvolution filter.
        """
        snr = 10.**(.1*lsnr)
        Fs = ofdm.format.Fs
        nfft = ofdm.format.nfft
        Nsc_used, Nsc = ofdm.format.Nsc_used, ofdm.format.Nsc
        ts_reps = ofdm.format.ts_reps
        # First, obtain a coarse frequency offset estimate from the short training sequences
        N_sts_period = nfft / 4
        t_sts_period = N_sts_period/Fs
        N_sts_reps = int(((ts_reps+.5) * nfft) / N_sts_period)
        sts = input[:N_sts_period*N_sts_reps]
        freq_off_estimate = -np.angle(np.sum(sts[:-N_sts_period] * sts[N_sts_period:].conj()))/(2*np.pi*t_sts_period)
        input *= np.exp(-2*np.pi*1j*freq_off_estimate*np.arange(input.size)/Fs)
        if 0:
            err = abs(freq_off_estimate - freq_offset)
            print 'Coarse frequency estimation error: %.0f Hz (%5.3f bins, %5.3f cyc/sym)' % (err, err / (Fs/64), err * 4e-6)
        offset = 8 + 16
        offset += N_sts_reps*N_sts_period
        # Next, obtain a fine frequency offset estimate from the long training sequences, and estimate
        # how uncertain this estimate is.
        N_lts_period = nfft
        t_lts_period = N_lts_period/Fs
        N_lts_reps = ts_reps
        lts = np.fft.fft(input[offset:offset+N_lts_period*N_lts_reps].reshape(N_lts_reps, N_lts_period), axis=1)
        lts[:, np.where(ofdm.format.lts_freq == 0)] = 0.
        # We have multiple receptions of the same signal, with independent noise.
        # We model the second reception as ... XXX
        # Now consider the random variable y.
        # y = (x+n1) * (a x+n2).conj()
        # E[y]
        # = E[abs(x)**2*a.conj() + x*n2.conj() + n1*x.conj() + n1*n2]
        # = var(x) * a.conj()
        # so the negative angle of this is the arg of a
        # var(y)
        # = var(x*n2. + n1*x. + n1*n2)
        # = E[(x n2. + n1 x. + n1 n2)(x. n2 + n1. x + n1. n2.)]
        # = 2var(n)var(x) + var(n)^2
        additional_freq_off_estimate = -np.angle((lts[:-1]*lts[1:].conj()).sum())/(2*np.pi*t_lts_period)
        # if each subcarrier has SNR=snr, then var(input) = ((snr+1) num_used_sc + num_unused_sc) var(n_i)
        # var(n) = var(input) / (snr num_used_sc/num_sc + 1)
        # var(x_i) = (var(input) - var(n)) / num_used_sc
        var_input = input.var()
        var_n = var_input / (float(snr * Nsc_used) / Nsc + 1)
        var_x = var_input - var_n
        var_y = 2*var_n*var_x + var_n**2
        freq_off_estimate += additional_freq_off_estimate
        input *= np.exp(-2*np.pi*1j*additional_freq_off_estimate*np.arange(input.size)/Fs)
        uncertainty = np.arctan((var_y**.5) / var_x) / (2*np.pi*t_lts_period) / nfft**.5
        if 0:
            err = abs(freq_off_estimate - freq_offset)
            # first print error, then print 1.5 sigma for the "best we can really expect"
            print 'Fine frequency estimation error: %.0f +/- %.0f Hz (%5.3f bins, %5.3f cyc/sym)' % (err, 1.5*uncertainty, err / (Fs/64), err * 4e-6)
        lts = np.fft.fft(input[offset:offset+N_lts_period*N_lts_reps].reshape(N_lts_reps, N_lts_period), axis=1)
        Y = lts.mean(0)
        S_Y = (np.abs(lts)**2).mean(0)
        S_X = np.abs(ofdm.format.lts_freq)**2
        H = np.where(S_X, Y/np.where(S_X, ofdm.format.lts_freq, 1.), 0)
        S_N = np.ones(64) * np.var(Y) / (1+snr)
        if 1:
            # Wiener deconvolution using estimated H
            #G = H.conj()*S_X / (np.abs(Y)**2 + S_N)
            G = Y.conj()*ofdm.format.lts_freq / S_Y
        else:
            # Directly invert estimated H
            G = np.where(H, 1./np.where(H, H, 1.), 0)
        var_x = np.var(input)/(64./52./snr + 1)/52.
        var_n = var_x/snr
        offset += 160-16
        return input[offset:], (G, uncertainty, var_n), offset

    def kalman_init(self, uncertainty, var_n):
        std_theta = 2*np.pi*uncertainty*4e-6 # convert from Hz to radians/symbol
        sigma_noise = 4*var_n*.5 # var_n/2 noise per channel, times 4 pilots
        sigma_re = sigma_noise + 4*np.sin(std_theta)**2 # XXX suspect
        sigma_im = sigma_noise + 4*np.sin(std_theta)**2 # XXX suspect
        sigma_theta = std_theta**2
        P = np.diag(np.array([sigma_re, sigma_im, sigma_theta])) # PAI 2013-02-12 calculation of P[0|0] verified
        x = np.array([[4.,0.,0.]]).T # PAI 2013-02-12 calculation of x[0|0] verified
        Q = P * 0.1 # XXX PAI 2013-02-12 calculation of Q[k] suspect
        R = np.diag(np.array([sigma_noise, sigma_noise])) # PAI 2013-02-12 calculation of R[k] verified
        return (P, x, Q, R)

    def kalman_update(self, (P, x, Q, R), pilot):
        # extended kalman filter
        re,im,theta = x[:,0]
        c = np.cos(theta)
        s = np.sin(theta)
        F = np.array([[c, -s, -s*re - c*im], [s,  c,  c*re - s*im], [0,  0,  1]]) # PAI 2013-02-12 calculation of F[k-1] verified
        x[0,0] = c*re - s*im # PAI 2013-02-12 calculation of x[k|k-1] verified
        x[1,0] = c*im + s*re
        P = F.dot(P).dot(F.T) + Q # PAI 2013-02-12 calculation of P[k|k-1] verified
        z = np.array([[pilot.real], [pilot.imag]]) # PAI 2013-02-12 calculation of z[k] verified
        y = z - x[:2,:] # PAI 2013-02-12 calculation of y[k] verified
        S = P[:2,:2] + R # PAI 2013-02-12 calculation of S[k] verified
        try:
            # K = P.dot(H.T).dot(np.linalg.inv(S)) # PAI 2013-02-12 calculation of K[k] verified
            # K = P H.T S.I
            # S.T K.T = H P.T
            # but S, P symmetric, so S K.T = H P
            K = np.linalg.solve(S, P[:2,:]).T
        except np.linalg.LinAlgError:
            # singular S means P has become negative definite
            K = 0
            # oh well, abs() its eigenvalues :-P
            U, V = np.linalg.eigh(P)
            P = V.dot(np.diag(np.abs(U))).dot(V.T.conj())
            print >> sys.stderr, 'Singular K'
        x += K.dot(y) # PAI 2013-02-12 calculation of x[k|k] verified
        P -= K.dot(P[:2,:]) # PAI 2013-02-12 calculation of P[k|k] verified
        u = x[0,0] - x[1,0]*1j
        u /= np.abs(u)
        return (P, x, Q, R), u

    def demodulate(self, input, (G, uncertainty, var_n), visualize=False):
        nfft = ofdm.format.nfft
        ncp = ofdm.format.ncp
        kalman_state = self.kalman_init(uncertainty, var_n)
        pilotPolarity = ofdm.pilotPolarity()
        demapped_bits = []
        j = ncp - 16
        i = 0
        initializedPlot = False
        length_symbols = 0
        while input.size-j > nfft and i <= length_symbols:
            sym = np.fft.fft(input[j:j+nfft])*G
            data = sym[ofdm.format.dataSubcarriers]
            pilots = sym[ofdm.format.pilotSubcarriers] * pilotPolarity.next() * ofdm.format.pilotTemplate
            kalman_state, kalman_u = self.kalman_update(kalman_state, np.sum(pilots))
            data *= kalman_u
            pilots *= kalman_u
            if i==0: # signal
                signal_bits = data.real>0
                signal_bits = interleaver.interleave(signal_bits, ofdm.format.Nsc, 1, reverse=True)
                scrambled_plcp_estimate = code.decode(signal_bits*2-1, 18)
                plcp_estimate = scrambler.scramble(scrambled_plcp_estimate, int(ofdm.format.Nsc*.5), scramblerState=0)
                parity = (np.sum(plcp_estimate) & 1) == 0
                if not parity:
                    return None, None, 0
                plcp_estimate = util.shiftin(plcp_estimate[:18], 18)[0]
                try:
                    encoding_estimate = util.rev(plcp_estimate & 0xF, 4)
                    rate_estimate = [r.encoding == encoding_estimate for r in self.rates].index(True)
                except ValueError:
                    return None, None, 0
                r_est = self.rates[rate_estimate]
                Nbpsc, constellation_estimate = r_est.Nbpsc, r_est.constellation
                min_dist = np.diff(np.unique(sorted(constellation_estimate.real)))[0]
                Ncbps, Nbps = r_est.Ncbps, r_est.Nbps
                length_octets = (plcp_estimate >> 5) & 0xFFF
                length_bits = length_octets * 8
                length_coded_bits = (length_bits+16+6)*2
                length_symbols = (length_coded_bits+Ncbps-1) // Ncbps
                signal_bits = code.encode(scrambled_plcp_estimate)
                dispersion = data - qam.bpsk[1][interleaver.interleave(signal_bits, ofdm.format.Nsc, 1)]
                dispersion = np.var(dispersion)
            else:
                if visualize:
                    if not initializedPlot:
                        pl.clf()
                        pl.axis('scaled')
                        pl.xlim(-1.5,1.5)
                        pl.ylim(-1.5,1.5)
                        initializedPlot = True
                    pl.scatter(data.real, data.imag, c=np.arange(data.size))
                ll = qam.demapper(data, constellation_estimate, min_dist, dispersion, Nbpsc)
                demapped_bits.append(ll.flatten())
            j += nfft+ncp
            i += 1
        if len(demapped_bits) == 0:
            return None, None, 0
        punctured_bits_estimate = interleaver.interleave(np.concatenate(demapped_bits), Ncbps, Nbpsc, True)
        coded_bits = code.depuncture(punctured_bits_estimate, r_est.puncturingMatrix)
        if coded_bits.size < length_coded_bits:
            return None, None, 0
        if visualize:
            pl.draw()
        return coded_bits[:length_coded_bits], length_bits, j

    def decodeFromLLR(self, llr, length_bits):
        scrambled_bits = code.decode(llr, length_bits+16)
        return scrambler.scramble(scrambled_bits, None, scramblerState=0x5d)[:length_bits+16]

    def autocorrelate(self, input):
        Nperiod = ofdm.format.nfft / 4
        autocorr = input[Nperiod:] * input[:-Nperiod].conj()
        Noutputs = autocorr.size // Nperiod
        autocorr = autocorr[:Nperiod*Noutputs].reshape(Noutputs, Nperiod).sum(1)
        corr_sum = np.abs(np.r_[np.zeros(5), autocorr]).cumsum()
        Nreps = int(((ofdm.format.ts_reps+.5) * ofdm.format.nfft) / Nperiod)
        return corr_sum[Nreps-1:] - corr_sum[:-Nreps+1]

    def decode(self, input, lsnr=None, visualize=False):
        score = self.autocorrelate(input)
        #score2 = -score * np.r_[0, np.diff(score, 2), 0]
        startIndex = max(0, 16*np.argmax(score)-64)
        input = input[startIndex:]
        if input.size <= 328:
            return None
        input, training_data, used_samples_training = self.train(input, lsnr if lsnr is not None else 10.)
        llr, length_bits, used_samples_data = self.demodulate(input, training_data, visualize)
        if llr is None:
            return None
        output_bits = self.decodeFromLLR(llr, length_bits)
        if not crc.checkFCS(output_bits[16:]):
            return None
        return util.shiftin(output_bits[16:-32], 8), startIndex + used_samples_training + used_samples_data

# produces 4 outputs before first output of autocorrelate()
class Autocorrelator:
    def __init__(self, next=None):
        self.input_fragment = np.zeros(16, np.complex128)
        self.corr_fragment = np.zeros(8, np.float64)
        self.next = next if next is not None else []
    def consume(self, input):
        # we only process input in 16-sample chunks
        stream = np.r_[self.input_fragment, input]
        n = 16*((stream.size-16)//16)
        self.input_fragment = stream[n:]
        if n:
            input = stream[:n+16]
            corr = stream[16:n+16] * stream[:n].conj()
            corr = corr.reshape(n//16, 16).sum(1)
            corr = np.r_[self.corr_fragment, corr]
            self.corr_fragment = corr[-9:]
            corr_sum = corr.cumsum()
            output = np.abs(corr_sum[9:] - corr_sum[:-9])
            self.next.consume(output)

