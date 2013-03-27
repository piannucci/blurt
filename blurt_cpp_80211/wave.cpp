#include "wave.h"
#include "chunk.h"
#include <fstream>
#include <string.h>

typedef struct {
    uint16_t wFormatTag;
    uint16_t wChannels;
    uint32_t dwFrameRate;
    uint32_t dwAvgBytesPerSec;
    uint16_t wBlockAlign;
    uint16_t wBitsPerSample;
} WaveFormatEx;

const static int WAVE_FORMAT_PCM = 0x0001;

class Wave_read {
private:
    int framesize;
    Chunk file_chunk, *data_chunk;
    std::ifstream file;

    void read_fmt_chunk(Chunk &chunk) {
        WaveFormatEx fmt;
        chunk.read(&fmt, 14);
        nchannels = fmt.wChannels;
        framerate = fmt.dwFrameRate;
        if (fmt.wFormatTag == WAVE_FORMAT_PCM) {
            sampwidth = 0;
            chunk.read(&sampwidth, 2);
            sampwidth = (sampwidth + 7) / 8;
        } else {
            throw "unknown format";
        }
        framesize = nchannels * sampwidth;
        comptype = "NONE";
        compname = "not compressed";
    };

public:
    std::string comptype, compname;
    int nframes, nchannels, framerate, sampwidth;

    Wave_read(const std::string &filename) :
        file(filename.c_str(), std::fstream::in | std::fstream::binary),
        file_chunk(file)
    {
        try {
            bool fmt_chunk_read = false, data_chunk_read = false;
            if (file_chunk.id != 'RIFF')
                throw "file does not start with RIFF id";
            uint32_t format;
            file_chunk.read(&format, 4);
            if (format != 'WAVE')
                throw "not a WAVE file";
            file_chunk.parseSubchunks();
            for (int i=0; i<file_chunk.subchunks.size(); i++) {
                Chunk &chunk = file_chunk.subchunks[i];
                if (chunk.id == 'fmt ') {
                    read_fmt_chunk(chunk);
                    fmt_chunk_read = true;
                } else if (chunk.id == 'data') {
                    if (!fmt_chunk_read)
                        throw "data chunk before fmt chunk";
                    data_chunk = &chunk;
                    data_chunk_read = true;
                    nframes = chunk.size / framesize;
                    break;
                }
            }
            if (!fmt_chunk_read || !data_chunk_read)
                throw "fmt chunk and/or data chunk missing";
        } catch (...) {
            file.close();
            throw;
        }
    };

    ~Wave_read() {
        close();
    };

    void close() {
        file.close();
    };

    void readframes(void *buffer, size_t nframes) {
        data_chunk->read(buffer, nframes * framesize);
    };
};

class Wave_write {
private:
    bool headerwritten;

    Chunk file_chunk, *data_chunk;
    std::ofstream file;

public:
    int nframes, nchannels, framerate, sampwidth;
    std::string comptype, compname;

    Wave_write(const std::string &filename) :
        file(filename.c_str(), std::fstream::out | std::fstream::binary),
        file_chunk(file, 'RIFF'), nframes(0), nchannels(0), sampwidth(0), framerate(0),
        comptype("NONE"), compname("not compressed")
    {
        headerwritten = false;
        file_chunk.write((char *)"WAVE", 4);
    };

    ~Wave_write() {
        close();
    };

    void setnchannels(int nchannels) {
        if (headerwritten) throw "cannot change parameters after starting to write";
        if (nchannels < 1) throw "bad # of channels";
        this->nchannels = nchannels;
    };

    void setsampwidth(int sampwidth) {
        if (headerwritten) throw "cannot change parameters after starting to write";
        if (sampwidth < 1 || sampwidth > 4) throw "bad sample width";
        this->sampwidth = sampwidth;
    };

    void setframerate(int framerate) {
        if (headerwritten) throw "cannot change parameters after starting to write";
        if (framerate <= 0) throw "bad frame rate";
        this->framerate = framerate;
    };

    void writeframes(void *data, size_t nframes) {
        size_t bytes = nframes * sampwidth * nchannels;
        _ensure_header_written();
        data_chunk->write(data, bytes);
    };

    void close() {
        try {
            _ensure_header_written();
            file_chunk.close();
            file.flush();
        } catch (...) {
            file.close();
        }
    };

    void _ensure_header_written() {
        if (!headerwritten) {
            if (!nchannels) throw "# channels not specified";
            if (!sampwidth) throw "sample width not specified";
            if (!framerate) throw "sampling rate not specified";

            Chunk &fmt_chunk = file_chunk.addSubchunk('fmt ');

            WaveFormatEx fmt;
            fmt.wFormatTag = WAVE_FORMAT_PCM;
            fmt.wChannels = nchannels;
            fmt.dwFrameRate = framerate;
            fmt.dwAvgBytesPerSec = nchannels * framerate * sampwidth;
            fmt.wBlockAlign = nchannels * sampwidth;
            fmt.wBitsPerSample = sampwidth * 8;
            fmt_chunk.write(&fmt, 16);

            data_chunk = &file_chunk.addSubchunk('data');
            headerwritten = true;
        }
    };
};

void readwave(const std::string &filename, std::vector<float> &output, float &Fs) {
    Wave_read f(filename);
    Fs = f.framerate;
    int nchannels = f.nchannels;
    int nframes = f.nframes;
    int sampwidth = f.sampwidth;
    output.resize(nframes);
    char readbuf[sampwidth*nchannels];
    float scale = 1.f/(1<<(8*sampwidth))/nchannels;
    for (int i=0; i<nframes; i++) {
        float acc = 0;
        f.readframes(readbuf, 1);
        for (int j=0; j<nchannels; j++) {
            int32_t sample = 0;
            memcpy(&sample, &readbuf[j*sampwidth], sampwidth);
            acc += sample * scale;
        }
        output[i] = acc;
    }
}

void writewave(const std::string &filename, const std::vector<float> &input, int Fs, int sampwidth, int nchannels) {
    int nframes = input.size()/nchannels;
    float scale = (1<<(8*sampwidth)) - 1;
    std::vector<char> quantized_input(input.size()*sampwidth);
    for (int i=0; i<input.size(); i++) {
        float clipped_input = input[i];
        clipped_input = (clipped_input > 1.f) ? 1.f : clipped_input;
        clipped_input = (clipped_input < -1.f) ? -1.f : clipped_input;
        int32_t sample = clipped_input * scale;
        memcpy(&quantized_input[i*sampwidth], &sample, sampwidth);
    }
    Wave_write f(filename);
    f.setnchannels(nchannels);
    f.setsampwidth(sampwidth);
    f.setframerate(Fs);
    f.writeframes(&*quantized_input.begin(), nframes);
    f.close();
}
