#ifndef CHUNK_H
#define CHUNK_H
#include "blurt.h"

class Chunk {
private:
    uint32_t offset, counter;
    Chunk *parent;
    Chunk(std::ofstream &file, std::string id, Chunk *parent);
    void writeHeader();
    void allocate(size_t count);
public:
    std::string id;
    uint32_t size;
    std::vector<Chunk> subchunks;
    std::ifstream *ifile;
    std::ofstream *ofile;
    Chunk(std::ifstream &file);
    Chunk(std::ofstream &file, std::string id);
    ~Chunk();
    void parseSubchunks();
    Chunk &addSubchunk(std::string id);
    void read(void *output, size_t count);
    void write(void *input, size_t count);
    void close();
};

#endif
