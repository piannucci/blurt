#include "wifi80211.h"
#include "qam.h"
#include "util.h"
#include "scrambler.h"
#include "interleave.h"
#include "fft.h"
#include <limits>
#include <cassert>
#include <valgrind/memcheck.h>

void OFDMFrame::wienerFilter(const it & begin, const it & end, v &G, float &snr, float &lsnr_estimate) const {
    v lts_freq(begin, end), Y(ofdm.format.nfft, 0);
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
        if (S_Y[j] == complex(0.,0.)) {
            G.clear();
            return;
        }
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

size_t OFDMFrame::try_coarse_train(const it begin, const it end) {
    const int N_sts_period = ofdm.format.nfft / 4;
    const int N_sts_reps = (ofdm.format.ts_reps * (ofdm.format.nfft+ofdm.format.ncp)) / N_sts_period;
    const int N_sts_samples = N_sts_period*N_sts_reps;

    // obtain a coarse frequency offset estimate from the short training sequences
    if (end - begin < N_sts_samples)
        return 0;

    complex acc = 0;
    for (int i=0; i<N_sts_samples-N_sts_period; i++)
        acc += begin[i] * conj(begin[N_sts_period+i]);
    freq_off_estimate = arg(acc)/N_sts_period;

    return N_sts_samples;
}

size_t OFDMFrame::try_fine_train(const it begin, const it end) {
    const int nfft = ofdm.format.nfft;
    const int Nsc_used = ofdm.format.Nsc_used, Nsc = ofdm.format.Nsc;
    const int ts_reps = ofdm.format.ts_reps;
    const int ncp = ofdm.format.ncp;
    const int N_lts_period = nfft;
    const int N_lts_reps = ts_reps;
    const int lts_cp = ncp * N_lts_reps;

    if (end - begin < lts_cp + 8 + N_lts_period*N_lts_reps)
        return 0;

    // Next, obtain a fine frequency offset estimate from the long training sequences, and estimate
    // how uncertain this estimate is.

    std::vector<complex> lts(begin, begin + N_lts_period*N_lts_reps);
    for (int j=0; j<lts.size(); j++)
        lts[j] *= expj(freq_off_estimate*(j+nextSampleIndex));

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
    complex acc = 0;
    p = &lts[0];
    for (int i=0; i<N_lts_reps-1; i++) {
        for (int j=0; j<N_lts_period; j++)
            acc += p[j]*conj(p[N_lts_period+j]);
        p += N_lts_period;
    }
    freq_off_estimate += arg(acc)/N_lts_period;

    v input(begin, begin + lts_cp + 8 + N_lts_period*N_lts_reps);

    for (int j=0; j<input.size(); j++)
        input[j] *= expj(freq_off_estimate*(j+nextSampleIndex));

    // if each subcarrier has SNR=snr, then var(input) = ((snr+1) num_used_sc + num_unused_sc) var(n_i)
    // var(n) = var(input) / (snr num_used_sc/num_sc + 1)
    // var(x_i) = (var(input) - var(n)) / num_used_sc
    std::vector<v> Gs;
    std::vector<float> snrs, lsnr_estimates;
    std::vector<int> offsets;
    int offset_index = -1;
    lsnr_estimate = -std::numeric_limits<float>::infinity();


    for (int i=0; i<16; i++) {
        int off = lts_cp + i-8;
        assert(off >= 0 && off+N_lts_period*N_lts_reps < input.size());

        offsets.push_back(off);
        Gs.push_back(v());
        snrs.push_back(0);
        lsnr_estimates.push_back(-std::numeric_limits<float>::infinity());

        wienerFilter(input.begin()+off, input.begin()+off+N_lts_period*N_lts_reps, Gs.back(), snrs.back(), lsnr_estimates.back());

        if (Gs.back().size() && lsnr_estimates.back() > lsnr_estimate) {
            offset_index = Gs.size()-1;
            lsnr_estimate = lsnr_estimates.back();
        }
    }

    if (offset_index != -1) {
        // pick the offset that gives the highest SNR
        float snr = snrs[offset_index];
        G = Gs[offset_index];
        float var_input = var(input);
        float var_n = var_input / (float(snr * Nsc_used) / Nsc + 1);
        float var_x = var_input - var_n;
        float var_y = 2*var_n*var_x + var_n*var_n;
        float uncertainty = atan(sqrt(var_y) / var_x) * (nfft+ncp) / N_lts_period / sqrt(nfft);

        kalman = KalmanPilotTracker(uncertainty, var_x/Nsc_used/snr);

        return offsets[offset_index] + N_lts_period*N_lts_reps;
    }

    status = error;

    return 0;
}

void OFDMFrame::transform_and_equalize_symbol(v & sym, int absolutePosition, v & data) {
    const int nfft = ofdm.format.nfft;
    const int ncp = ofdm.format.ncp;

    for (int j=0; j<nfft; j++)
        sym[j] *= expj(freq_off_estimate*(j+absolutePosition));

    fft(&sym[0], nfft);

    for (int j=0; j<nfft; j++)
        sym[j] *= G[j];

    for (int j=0; j<data.size(); j++)
        data[j] = sym[ofdm.format.dataSubcarriers[j]];

    complex pilot_sum = 0;
    complex polarity = pilotPolarity.next();
    for (int j=0; j<ofdm.format.pilotSubcarriers.size(); j++)
        pilot_sum += sym[ofdm.format.pilotSubcarriers[j]] * polarity * ofdm.format.pilotTemplate[j];

    complex kalman_u;
    kalman.update(pilot_sum, kalman_u);
    for (int j=0; j<data.size(); j++)
        data[j] = data[j] * kalman_u;
}

size_t OFDMFrame::try_consume_header(const it begin, const it end) {
    const int nfft = ofdm.format.nfft;
    const int ncp = ofdm.format.ncp;

    if (end - begin < ncp+nfft)
        return 0;

    // get data bits
    v sym(begin+ncp, begin+ncp+nfft), data(ofdm.format.Nsc);
    transform_and_equalize_symbol(sym, nextSampleIndex+ncp, data);
    std::vector<int> signal_bits(data.size());
    for (int j=0; j<data.size(); j++)
        signal_bits[j] = (real(data[j]) > 0) ? 1 : -1;

    // decode them
    std::vector<int> signal_bits_deinterleaved;
    interleave(signal_bits, ofdm.format.Nsc, 1, true, signal_bits_deinterleaved);
    bitvector scrambled_plcp_estimate, plcp_estimate;
    code.decode(signal_bits_deinterleaved, 18, scrambled_plcp_estimate);
    Scrambler::scramble(scrambled_plcp_estimate, ofdm.format.Nsc/2, plcp_estimate, 0);

    // parity valid?
    int parity = 0;
    for (int j=0; j<plcp_estimate.size(); j++)
        parity ^= plcp_estimate[j] & 1;
    parity = (parity == 0);
    if (!parity) {
        status = error;
        return 0;
    }

    // interpret
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
        status = error;
        return 0;
    }
    r_est = &rates[rate_estimate];
    int Ncbps = r_est->Ncbps;
    int length_octets = (plcp_estimate_value >> 5) & 0xfff;
    length_bits = length_octets * 8;
    int length_coded_bits = (length_bits+16+6)*2;
    length_symbols = (length_coded_bits+Ncbps-1) / Ncbps;
    bitvector signal_bits_encoded, signal_bits_interleaved;
    code.encode(scrambled_plcp_estimate, signal_bits_encoded);
    interleave(signal_bits_encoded, ofdm.format.Nsc, 1, false, signal_bits_interleaved);
    std::vector<complex> residual;
    rates[0].constellation.map(signal_bits_interleaved, residual);
    for (int j=0; j<residual.size(); j++)
        residual[j] = residual[j] - data[j];
    dispersion = var(residual);

    symbols_processed = 0;

    return nfft+ncp;
}

size_t OFDMFrame::try_consume_symbol(const it begin, const it end) {
    const int nfft = ofdm.format.nfft;
    const int ncp = ofdm.format.ncp;

    if (end - begin < ncp+nfft)
        return 0;

    // get data bits
    v sym(begin+ncp, begin+ncp+nfft), data(ofdm.format.Nsc);
    transform_and_equalize_symbol(sym, nextSampleIndex+ncp, data);

    std::vector<int> ll;
    r_est->constellation.demap(data, dispersion, ll);
    demapped_bits.insert(demapped_bits.end(), ll.begin(), ll.end());

    symbols_processed++;

    return nfft+ncp;
}

bool OFDMFrame::try_finish_decode() {
    if (demapped_bits.size() == 0)
        return false;

    std::vector<int> punctured_bits_estimate;
    interleave(demapped_bits, r_est->Ncbps, r_est->Nbpsc, true, punctured_bits_estimate);

    std::vector<int> llr;
    r_est->puncturingMask.depuncture(punctured_bits_estimate, llr);

    int length_coded_bits = (length_bits+16+6)*2;

    if (llr.size() < length_coded_bits)
        return false;

    llr.resize(length_coded_bits);

    bitvector scrambled_bits;
    code.decode(llr, length_bits+16, scrambled_bits);
    scrambled_bits.resize(length_bits+16);

    bitvector output_bits;
    Scrambler::scramble(scrambled_bits, 0, output_bits, 0x5d);

    std::vector<int> payload_vector;

    shiftin(output_bits, 8, payload_vector);

    frame_payload = std::string(payload_vector.begin(), payload_vector.end());

    return true;
}

size_t OFDMFrame::try_consume(const it begin, const it end) {
    size_t used = 0, usedTotal = 0;
    while (1) {
        switch (status) {
            initialized:
                if (!(used = try_coarse_train(begin+used, end)))
                    return usedTotal;
                status = coarse_training_complete;
                break;

            coarse_training_complete:
                if (!(used = try_fine_train(begin+used, end)))
                    return usedTotal;
                status = fine_training_complete;
                break;

            fine_training_complete:
                if (!(used = try_consume_header(begin+used, end)))
                    return usedTotal;
                status = header_complete;
                break;

            header_complete:
                if (!(used = try_consume_symbol(begin+used, end)))
                    return usedTotal;
                status = (symbols_processed == length_symbols) ? payload_complete : header_complete;
                break;

            payload_complete:
                if (!try_finish_decode())
                    status = error;
                else
                    status = done;
                return usedTotal;

            done:
            error:
                return usedTotal;
        }
        usedTotal += used;
        nextSampleIndex += used;
    }
}

OFDMFrame::OFDMFrame(const WiFi80211 & wifi)
    : ofdm(wifi.ofdm), code(wifi.code), rates(wifi.rates) {
}
