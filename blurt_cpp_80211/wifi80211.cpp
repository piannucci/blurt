#include "wifi80211.h"
#include "qam.h"
#include "util.h"
#include "scrambler.h"
#include "interleave.h"
#include "fft.h"
#include <limits>
#include "kalman.h"

WiFi80211::WiFi80211() : ofdm(audioLTFormat()), code(7, 0133, 0171)
{
    rates.push_back(Rate(0xd, QAM(1), PuncturingMask(1,2), ofdm.format));
    rates.push_back(Rate(0xf, QAM(2), PuncturingMask(1,2), ofdm.format));
    rates.push_back(Rate(0x5, QAM(2), PuncturingMask(3,4), ofdm.format));
    rates.push_back(Rate(0x7, QAM(4), PuncturingMask(1,2), ofdm.format));
    rates.push_back(Rate(0x9, QAM(4), PuncturingMask(3,4), ofdm.format));
    rates.push_back(Rate(0xb, QAM(6), PuncturingMask(2,3), ofdm.format));
    rates.push_back(Rate(0x1, QAM(6), PuncturingMask(3,4), ofdm.format));
    rates.push_back(Rate(0x3, QAM(6), PuncturingMask(5,6), ofdm.format));
}

void WiFi80211::plcp_bits(const Rate &rate, int octets, bitvector &output) {
    int plcp_rate = rev(rate.encoding, 4);
    int plcp = plcp_rate | (octets << 5);
    int parity = (mul(plcp, 0x1FFFF) >> 16) & 1;
    plcp |= parity << 17;
    std::vector<int> input(1);
    input[0] = plcp;
    shiftout(input, 18, output);
}

void WiFi80211::subcarriersFromBits(const bitvector &bits, const Rate &rate, int scramblerState, std::vector<complex> &output) {
    // scrambled includes tail & padding
    bitvector scrambled, coded, punctured, interleaved;
    Scrambler::scramble(bits, rate.Nbps, scrambled, scramblerState);
    code.encode(scrambled, coded);
    rate.puncturingMask.puncture(coded, punctured);
    interleave(punctured, rate.Ncbps, rate.Nbpsc, false, interleaved);
    rate.constellation.map(interleaved, output);
}

void WiFi80211::encode(const std::vector<int> &input_octets, int rate_index, std::vector<complex> &output) {
    bitvector service_bits(16, 0);
    bitvector data_bits, fcs_bits;
    shiftout(input_octets, 8, data_bits);
    crc.FCS(data_bits, fcs_bits);
    data_bits.insert(data_bits.begin(), service_bits.begin(), service_bits.end());
    data_bits.insert(data_bits.end(), fcs_bits.begin(), fcs_bits.end());
    std::vector<complex> signal_subcarriers, data_subcarriers;
    bitvector plcp_bits_output;
    plcp_bits(rates[rate_index], input_octets.size()+4, plcp_bits_output);
    subcarriersFromBits(plcp_bits_output, rates[0], 0, signal_subcarriers);
    subcarriersFromBits(data_bits, rates[rate_index], 0x5d, data_subcarriers);
    signal_subcarriers.insert(signal_subcarriers.end(), data_subcarriers.begin(), data_subcarriers.end());
    ofdm.encode(signal_subcarriers, output);
}

void WiFi80211::autocorrelate(const std::vector<complex> &input, std::vector<float> &output) {
    int nfft = ofdm.format.nfft;
    int ncp = ofdm.format.ncp;
    int ts_reps = ofdm.format.ts_reps;
    int Nperiod = nfft / 4;
    int Noutputs = (input.size() - Nperiod) / Nperiod;
    output.resize(Noutputs);
    std::vector<float> corr_sum(5 + Noutputs, 0);
    int Nreps = ts_reps * (nfft + ncp) / Nperiod;
    for (int i=0; i<Noutputs; i++) {
        int k = i * Nperiod;
        complex acc = 0;
        for (int j=0; j<Nperiod; j++)
            acc += input[k+j+Nperiod] * conj(input[k+j]);
        corr_sum[5+i] = corr_sum[4+i] + abs(acc);
    }
    for (int i=0; i<Noutputs; i++)
        output[i] = corr_sum[Nreps-1+i] - corr_sum[i];
}

void WiFi80211::synchronize(const std::vector<complex> &input, std::vector<int> &startIndices) {
    std::vector<float> score;
    autocorrelate(input, score);
    startIndices.clear();
    // look for points outstanding in their neighborhood by explicit comparison
    const int l = 25;
    int N = score.size();
    for (int j=0; j < N + 2*l; j++) {
        bool pass = true;
        for (int i=0; i<l; i++)
            pass = pass && (((0 <= j-i && j-i < N) ? score[j-i] : 0) < ((0 <= j-l < N) ? score[j-l] : 0));
        for (int i=l+1; i<2*l+1; i++)
            pass = pass && (((0 <= j-i && j-i < N) ? score[j-i] : 0) < ((0 <= j-l < N) ? score[j-l] : 0));
        if (pass) {
            int startIndex = 16*(j - l) - 64;
            if (startIndex < 0)
                startIndex = 0;
            startIndices.push_back(startIndex);
        }
    }
}

void WiFi80211::wienerFilter(const std::vector<complex> &lts, std::vector<complex> &G, float &snr, float &lsnr_estimate) {
    std::vector<complex> lts_freq(lts), Y(ofdm.format.nfft, 0);
    std::vector<float> S_Y(ofdm.format.nfft, 0);
    complex *p = &lts_freq[0];
    for (int i=0; i<ofdm.format.ts_reps; i++) {
        fft(p, ofdm.format.nfft);
        for (int j=0; j<ofdm.format.nfft; j++) {
            Y[j] += p[j];
            S_Y[j] += norm(p[j]);
        }
        p += ofdm.format.nfft;
    }
    float scale = 1.f/ofdm.format.ts_reps;
    G.resize(ofdm.format.nfft);
    for (int j=0; j<ofdm.format.nfft; j++) {
        Y[j] *= scale;
        S_Y[j] *= scale;
        // Wiener deconvolution
        G[j] = conj(Y[j])*ofdm.format.lts_freq[j] / S_Y[j];
    }
    // noise estimation via residuals
    complex resid_sum = 0;
    float resid_sum_sq = 0;
    p = &lts_freq[0];
    int count = 0;
    for (int i=0; i<ofdm.format.ts_reps; i++) {
        for (int j=0; j<ofdm.format.nfft; j++) {
            complex resid = G[j] * p[j] - ofdm.format.lts_freq[j];
            resid_sum += resid;
            resid_sum_sq += norm(resid);
            count++;
        }
        p += ofdm.format.nfft;
    }
    float resid_var = resid_sum_sq / count - norm(resid_sum)/count/count;
    snr = 1./resid_var;
    lsnr_estimate = 10*log10(snr);
}

float var(const std::vector<complex> &input) {
    complex sum = 0;
    float sum_sq = 0;
    int count = 0;
    for (int i=0; i<input.size(); i++) {
        sum += input[i];
        sum_sq += norm(input[i]);
        count++;
    }
    return sum_sq / count - norm(sum)/count/count;
}

void WiFi80211::train(std::vector<complex> &input, std::vector<complex> &G, float &uncertainty, float &var_ni, int &offset, float &lsnr_estimate) {
    const int nfft = ofdm.format.nfft;
    const int Nsc_used = ofdm.format.Nsc_used, Nsc = ofdm.format.Nsc;
    const int ts_reps = ofdm.format.ts_reps;
    const int ncp = ofdm.format.ncp;
    // First, obtain a coarse frequency offset estimate from the short training sequences
    const int N_sts_period = nfft / 4;
    const int N_sts_reps = (ts_reps * (nfft+ncp)) / N_sts_period;
    G.clear();
    lsnr_estimate = -std::numeric_limits<float>::infinity();
    if (input.size() < N_sts_period*N_sts_reps)
        return;
    std::vector<complex> sts(input.begin(), input.begin() + N_sts_period*N_sts_reps);
    complex acc = 0;
    for (int i=0; i<N_sts_period*(N_sts_reps-1); i++)
        acc += sts[i] * conj(sts[N_sts_period+i]);
    float angle_off_estimate = arg(acc)/N_sts_period;
    for (int i=0; i<input.size(); i++)
        input[i] = input[i] * exp(complex(0,angle_off_estimate*i));
    // Next, obtain a fine frequency offset estimate from the long training sequences, and estimate
    // how uncertain this estimate is.
    const int N_lts_period = nfft;
    const int N_lts_reps = ts_reps;
    const int lts_cp = ncp * N_lts_reps;
    offset = N_sts_reps*N_sts_period;
    if (input.size() < offset + N_lts_period*N_lts_reps)
        return;
    std::vector<complex> lts(input.begin() + offset, input.begin() + offset + N_lts_period*N_lts_reps);
    complex *p = &lts[0];
    for (int i=0; i<N_lts_reps; i++)
    {
        fft(p, nfft);
        for (int j=0; j<nfft; j++) {
            if (ofdm.format.lts_freq[j] == complex(0,0))
                p[j] = 0;
        }
        p += N_lts_period;
    }
    // We have multiple receptions of the same signal, with independent noise.
    // We model reception 1 as differing from reception 2 by a complex unit multiplier a.
    // Now consider the random variable y:
    // y = (x+n1) * (a x+n2).conj()
    // E[y]
    // = E[abs(x)**2*a.conj() + x*n2.conj() + n1*x.conj() + n1*n2]
    // = var(x) * a.conj()
    // so the negative angle of this is the arg of a
    // var(y)
    // = var(x*n2. + n1*x. + n1*n2)
    // = E[(x n2. + n1 x. + n1 n2)(x. n2 + n1. x + n1. n2.)]
    // = 2var(n)var(x) + var(n)^2
    // std(angle(y)) ~ arctan(std(y) / abs(E[y]))
    acc = 0;
    for (int i=0; i<N_lts_reps-1; i++) {
        int k = i*N_lts_period;
        for (int j=0; j<N_lts_period; j++)
            acc += lts[k+j]*conj(lts[k+N_lts_period+j]);
    }
    float additional_freq_off_estimate = arg(acc)/N_lts_period;
    for (int i=0; i<input.size(); i++)
        input[i] = input[i] * exp(complex(0, additional_freq_off_estimate*i));
    // if each subcarrier has SNR=snr, then var(input) = ((snr+1) num_used_sc + num_unused_sc) var(n_i)
    // var(n) = var(input) / (snr num_used_sc/num_sc + 1)
    // var(x_i) = (var(input) - var(n)) / num_used_sc
    std::vector<std::vector<complex> > Gs;
    std::vector<float> snrs, lsnr_estimates;
    std::vector<int> offsets;
    int offset_index = -1;
    for (int i=0; i<16; i++) {
        int off = offset + lts_cp + i-8;
        offsets.push_back(off);
        Gs.push_back(std::vector<complex>());
        snrs.push_back(0);
        lsnr_estimates.push_back(-std::numeric_limits<float>::infinity());
        lts.assign(input.begin()+off, input.begin()+off+N_lts_period*N_lts_reps);
        wienerFilter(lts, Gs.back(), snrs.back(), lsnr_estimates.back());
        if (lsnr_estimates.back() > lsnr_estimate) {
            offset_index = i;
            lsnr_estimate = lsnr_estimates.back();
        }
    }
    // pick the offset that gives the highest SNR
    float snr = snrs[offset_index];
    G = Gs[offset_index];
    offset = offsets[offset_index];
    float var_input = var(input);
    float var_n = var_input / (float(snr * Nsc_used) / Nsc + 1);
    float var_x = var_input - var_n;
    float var_y = 2*var_n*var_x + var_n*var_n;
    uncertainty = atan(sqrt(var_y) / var_x) * (nfft+ncp) / N_lts_period / sqrt(nfft);
    var_ni = var_x/Nsc_used/snr;
    offset += N_lts_reps * nfft;
}

void WiFi80211::demodulate(const std::vector<complex> &input, const std::vector<complex> &G, float uncertainty, float var_n,
                           std::vector<int> &coded_bits, int &length_bits, int &offset) {
    int nfft = ofdm.format.nfft;
    int ncp = ofdm.format.ncp;
    KalmanPilotTracker kalman(uncertainty, var_n);
    PilotPolarity pilotPolarity;
    std::vector<int> demapped_bits;
    offset += ncp;
    int i = 0;
    int length_symbols = 0, length_coded_bits;
    float dispersion;
    Rate &r_est = rates[0];
    while (input.size()-offset > nfft && i <= length_symbols) {
        std::vector<complex> sym(input.begin()+offset, input.begin()+offset+nfft);
        fft(&sym[0], nfft);
        for (int j=0; j<nfft; j++)
            sym[j] *= G[j];
        std::vector<complex> data(ofdm.format.Nsc);
        for (int j=0; j<data.size(); j++)
            data[j] = sym[ofdm.format.dataSubcarriers[j]];
        std::vector<complex> pilots(ofdm.format.pilotSubcarriers.size());
        complex pilot_sum = 0;
        complex polarity = pilotPolarity.next();
        for (int j=0; j<pilots.size(); j++) {
            pilots[j] = sym[ofdm.format.pilotSubcarriers[j]] * polarity * ofdm.format.pilotTemplate[j];
            pilot_sum += pilots[j];
        }
        complex kalman_u;
        kalman.update(pilot_sum, kalman_u);
        for (int j=0; j<data.size(); j++)
            data[j] = data[j] * kalman_u;
        for (int j=0; j<pilots.size(); j++)
            pilots[j] = pilots[j] * kalman_u;
        if (i==0) {
            // signal
            std::vector<int> signal_bits(data.size());
            for (int j=0; j<data.size(); j++)
                signal_bits[j] = (real(data[j]) > 0) ? 1 : -1;
            std::vector<int> signal_bits_deinterleaved;
            interleave(signal_bits, ofdm.format.Nsc, 1, true, signal_bits_deinterleaved);
            bitvector scrambled_plcp_estimate, plcp_estimate;
            code.decode(signal_bits_deinterleaved, 18, scrambled_plcp_estimate);
            Scrambler::scramble(scrambled_plcp_estimate, ofdm.format.Nsc/2, plcp_estimate, 0);
            int parity = 0;
            for (int j=0; j<plcp_estimate.size(); j++)
                parity ^= plcp_estimate[j] & 1;
            parity = (parity == 0);
            if (!parity) {
                coded_bits.clear();
                length_bits = 0;
                return;
            }
            std::vector<int> plcp_estimate_shifted_in;
            plcp_estimate.resize(18);
            shiftin(plcp_estimate, 18, plcp_estimate_shifted_in);
            int plcp_estimate_value = plcp_estimate_shifted_in[0];
            int encoding_estimate = rev(plcp_estimate_value & 0xf, 4);
            int rate_estimate;
            for (rate_estimate=0; rate_estimate<rates.size(); rate_estimate++)
                if (rates[rate_estimate].encoding == encoding_estimate)
                    break;
            if (rate_estimate == rates.size()) {
                coded_bits.clear();
                length_bits = 0;
                return;
            }
            r_est = rates[rate_estimate];
            int Ncbps = r_est.Ncbps;
            int length_octets = (plcp_estimate_value >> 5) & 0xfff;
            length_bits = length_octets * 8;
            length_coded_bits = (length_bits+16+6)*2;
            length_symbols = (length_coded_bits+Ncbps-1) / Ncbps;
            bitvector signal_bits_encoded, signal_bits_interleaved;
            code.encode(scrambled_plcp_estimate, signal_bits_encoded);
            interleave(signal_bits_encoded, ofdm.format.Nsc, 1, false, signal_bits_interleaved);
            std::vector<complex> residual;
            rates[0].constellation.map(signal_bits_interleaved, residual);
            for (int j=0; j<residual.size(); j++)
                residual[j] = residual[j] - data[j];
            dispersion = var(residual);
        } else {
            std::vector<int> ll;
            r_est.constellation.demap(data, dispersion, ll);
            demapped_bits.insert(demapped_bits.end(), ll.begin(), ll.end());
        }
        offset += nfft+ncp;
        i++;
    }
    if (demapped_bits.size() == 0) {
        coded_bits.clear();
        length_bits = 0;
        return;
    }
    std::vector<int> punctured_bits_estimate;
    interleave(demapped_bits, r_est.Ncbps, r_est.Nbpsc, true, punctured_bits_estimate);
    r_est.puncturingMask.depuncture(punctured_bits_estimate, coded_bits);
    if (coded_bits.size() < length_coded_bits) {
        coded_bits.clear();
        length_bits = 0;
        return;
    }
    coded_bits.resize(length_coded_bits);
}

void WiFi80211::decodeFromLLR(const std::vector<int> &input, int length_bits, bitvector &output) {
    bitvector scrambled_bits;
    code.decode(input, length_bits+16, scrambled_bits);
    scrambled_bits.resize(length_bits+16);
    Scrambler::scramble(scrambled_bits, 0, output, 0x5d);
}

void WiFi80211::decode(const std::vector<complex> &input, std::vector<DecodeResult> &output) {
    output.clear();
    int endIndex = 0;
    std::vector<complex> working_buffer;
    working_buffer.reserve(input.size());
    int minSize = ofdm.format.preambleLength + ofdm.format.ncp + ofdm.format.nfft;
    std::vector<int> startIndices;
    synchronize(input, startIndices);
    for (int i=0; i<startIndices.size(); i++) {
        int startIndex = startIndices[i];
        if (startIndex >= input.size())
            break;
        if (startIndex < endIndex) // we already successfully decoded this packet
            continue;
        working_buffer.assign(input.begin()+startIndex, input.end());
        if (working_buffer.size() <= minSize)
            continue;
        std::vector<complex> G;
        float uncertainty, var_ni, lsnr;
        int offset = 0;
        train(working_buffer, G, uncertainty, var_ni, offset, lsnr);
        if (G.size() == 0)
            continue;
        std::vector<int> llr;
        int length_bits;
        demodulate(working_buffer, G, uncertainty, var_ni, llr, length_bits, offset);
        if (llr.size() == 0)
            continue;
        bitvector output_bits;
        decodeFromLLR(llr, length_bits, output_bits);
        output_bits.erase(output_bits.begin(), output_bits.begin()+16);
        output_bits.resize(length_bits);
        if (!crc.checkFCS(output_bits))
            continue;
        output_bits.resize(output_bits.size()-32);
        DecodeResult result;
        shiftin(output_bits, 8, result.payload);
        endIndex = startIndex + offset;
        result.startIndex = startIndex;
        result.endIndex = endIndex;
        result.lsnr = lsnr;
        output.push_back(result);
    }
}
