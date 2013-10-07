#ifndef WIFI80211_H
#define WIFI80211_H
#include "blurt.h"
#include "constellation.h"
#include "rate.h"
#include "ofdm.h"
#include "cc.h"
#include "crc.h"
#include "kalman.h"

class DecodeResult {
public:
    std::vector<int> payload;
    size_t startIndex, endIndex;
    float lsnr;
};

class WiFi80211 {
private:
    void plcp_bits(const Rate &rate, size_t octets, bitvector &output) const;
    void subcarriersFromBits(const bitvector &bits, const Rate &rate, uint32_t scramblerState, std::vector<complex> &output) const;
    void autocorrelate(const std::vector<complex> &input, std::vector<float> &output) const;
    void synchronize(const std::vector<complex> &input, std::vector<size_t> &startIndices) const;
    void wienerFilter(const std::vector<complex> &lts, std::vector<complex> &G, float &snr, float &lsnr_estimate) const;
    void train(std::vector<complex> &input, std::vector<complex> &G, float &uncertainty, float &var_ni, size_t &offset, float &lsnr_estimate) const;
    void demodulate(const std::vector<complex> &input, const std::vector<complex> &G, float uncertainty, float var_n,
            std::vector<int> &coded_bits, size_t &length_bits, size_t &offset) const;
    void decodeFromLLR(const std::vector<int> &input, size_t length_bits, bitvector &output) const;
public:
    OFDM ofdm;
    ConvolutionalCode code;
    std::vector<Rate> rates;
    CRC crc;
    WiFi80211();
    void encode(const std::vector<uint8_t> &input, size_t rate_index, std::vector<complex> &output) const;
    void decode(const std::vector<complex> &input, std::vector<DecodeResult> &output) const;
};

float var(const std::vector<complex> &input);

class OFDMFrame {
    using v = std::vector<complex>;
    using it = v::iterator;
    size_t nextSampleIndex = 0;
    // valid after coarse training complete
    float freq_off_estimate = 0.;
    // valid after fine training complete
    float lsnr_estimate = 0.;
    v G;
    KalmanPilotTracker kalman = {0,0};
    PilotPolarity pilotPolarity;
    // valid after header complete
    int length_symbols = -1, symbols_processed = -1;
    int length_bits = -1;
    float dispersion = 0.;
    const Rate *r_est = nullptr;
    // valid after payload complete
    std::vector<int> demapped_bits;
    // valid after done
    std::string frame_payload;
    enum {
        initialized=0,
        coarse_training_complete,
        fine_training_complete,
        header_complete,
        payload_complete,
        done,
        error
    } status;

    const OFDM & ofdm;
    const ConvolutionalCode & code;
    const std::vector<Rate> & rates;

    void wienerFilter(const it & begin, const it & end, v &G, float &snr, float &lsnr_estimate) const;
    void transform_and_equalize_symbol(v & sym, int absolutePosition, v & data);

    size_t try_coarse_train(const it begin, const it end);
    size_t try_fine_train(const it begin, const it end);
    size_t try_consume_header(const it begin, const it end);
    size_t try_consume_symbol(const it begin, const it end);
    bool try_finish_decode();
public:
    OFDMFrame(const WiFi80211 & wifi);

    size_t try_consume(const it begin, const it end);

    bool iserror() const { return status == error; }
    bool isdone() const { return status == done; }
    float lsnr() const { return lsnr_estimate; }
    std::string payload() const { return frame_payload; }
};

#endif
