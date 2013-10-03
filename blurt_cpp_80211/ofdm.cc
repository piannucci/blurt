#include "ofdm.h"
#include "fft.h"
#include <cmath>

void trainingSequenceFromFreq(const std::vector<complex> &ts_freq, int reps, int ncp, std::vector<complex> &ts_time) {
    int nfft = ts_freq.size();
    int tilesNeeded = (ncp*reps + nfft-1) / nfft + reps + 1; // +1 for cross-fade
    std::vector<complex> ts(ts_freq);
    ifft(&ts[0], nfft);
    int start = (-(ncp*reps) % nfft + nfft) % nfft; // mod with round towards negative infinity
    ts.reserve(nfft*tilesNeeded); // make sure no reallocations while tiling
    for (int i=1; i<tilesNeeded; i++)
        ts.insert(ts.end(), ts.begin(), ts.begin() + nfft); // tile out training sequence
    ts_time.assign(ts.begin() + start, ts.end()-(nfft-1));
}

OFDMFormat audioLTFormat() {
    OFDMFormat f;
    f.nfft = 64;
    f.ncp = 64;
    f.ts_reps = 6;
    f.sts_freq.resize(f.nfft, 0);
    complex sts_unit = float(sqrt(13./6.)) * complex(1,1);
    int sts_nonzero_idx[] = {4, 8, 12, 16, 20, 24, -24, -20, -16, -12, -8, -4};
    complex sts_nonzero_phase[] = {-1, -1, 1, 1, 1, 1, 1, -1, 1, -1, -1, 1};
    for (int i=0; i<sizeof(sts_nonzero_idx)/sizeof(int); i++)
        f.sts_freq[(sts_nonzero_idx[i]+f.nfft)%f.nfft] = sts_unit * sts_nonzero_phase[i];
    trainingSequenceFromFreq(f.sts_freq, f.ts_reps, f.ncp, f.sts_time);
    complex lts_freq_vals[] = {
        0, 1,-1,-1, 1, 1,-1, 1,-1, 1,-1,-1,-1,-1,-1, 1,
        1,-1,-1, 1,-1, 1,-1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1,
        1, 1, 1, 1, 1,-1,-1, 1, 1,-1, 1,-1, 1, 1, 1, 1
    };
    f.lts_freq.assign(lts_freq_vals, lts_freq_vals+f.nfft);
    trainingSequenceFromFreq(f.lts_freq, f.ts_reps, f.ncp, f.lts_time);
    int dataSubcarriers_vals[] = {
        38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
        55, 56, 58, 59, 60, 61, 62, 63,  1,  2,  3,  4,  5,  6,  8,  9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26
    };
    f.dataSubcarriers.assign(dataSubcarriers_vals, dataSubcarriers_vals+sizeof(dataSubcarriers_vals)/sizeof(int));
    int pilotSubcarriers_vals[] = {43, 57,  7, 21};
    f.pilotSubcarriers.assign(pilotSubcarriers_vals, pilotSubcarriers_vals+sizeof(pilotSubcarriers_vals)/sizeof(int));
    complex pilotTemplate_vals[] = {1,1,1,-1};
    f.pilotTemplate.assign(pilotTemplate_vals, pilotTemplate_vals+sizeof(pilotTemplate_vals)/sizeof(complex));
    f.Nsc = f.dataSubcarriers.size();
    f.Nsc_used = f.dataSubcarriers.size() + f.pilotSubcarriers.size();
    f.preambleLength = f.sts_time.size() + f.lts_time.size() - 2;
    return f;
}

PilotPolarity::PilotPolarity() : sc(0x7f) {
}

complex PilotPolarity::next() {
    return 1 - 2*sc.next();
}

void stitch(const std::vector<std::vector<complex> > &input, std::vector<complex> &output) {
    size_t count = 1;
    for (int i=0; i<input.size(); i++)
        count += input[i].size() - 1;
    output.assign(count, 0);
    int j = 0;
    std::vector<complex> y;
    for (int i=0; i<input.size(); i++) {
        y.assign(input[i].begin(), input[i].end());
        y[0] *= .5;
        y[y.size()-1] *= .5;
        for (int k=0; k<y.size(); k++)
            output[j+k] += y[k];
        j += y.size()-1;
    }
}

OFDM::OFDM(const OFDMFormat &format) : format(format) {
}

void OFDM::encode(const std::vector<complex> &input, std::vector<complex> &output) const {
    PilotPolarity pilotPolarity;
    std::vector<std::vector<complex> > output_chunks;
    int tilesNeeded = (format.ncp+format.nfft-1) / format.nfft + 2; // +1 for symbol, +1 for cross-fade
    int start = (-format.ncp % format.nfft + format.nfft) % format.nfft;
    output_chunks.push_back(format.sts_time);
    output_chunks.push_back(format.lts_time);
    for (int i=0; i<input.size(); i+=format.Nsc) {
        std::vector<complex> symbol(format.nfft, 0);
        for (int j=0; j<format.dataSubcarriers.size(); j++)
            symbol[format.dataSubcarriers[j]] = input[i+j];
        complex polarity = pilotPolarity.next();
        for (int j=0; j<format.pilotSubcarriers.size(); j++)
            symbol[format.pilotSubcarriers[j]] = format.pilotTemplate[j] * polarity;
        ifft(&symbol[0], format.nfft);
        symbol.reserve(format.nfft*tilesNeeded);
        for (int i=1; i<tilesNeeded; i++)
            symbol.insert(symbol.end(), symbol.begin(), symbol.begin() + format.nfft);
        output_chunks.push_back(std::vector<complex>(symbol.begin()+start, symbol.end()-(format.nfft-1)));
    }
    stitch(output_chunks, output);
}
