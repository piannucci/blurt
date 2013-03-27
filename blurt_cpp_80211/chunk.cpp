#include "chunk.h"
#include <fstream>

Chunk::Chunk(std::ifstream &file) :
    ifile(&file), ofile(0), offset(file.tellg()), counter(0)
{
    this->ifile->read((char *)id, 4);
    this->ifile->read((char *)size, 4);
}

void Chunk::writeHeader() {
    this->ofile->seekp(offset);
    this->ofile->write((char *)&id, 4);
    this->ofile->write((char *)&size, 4);
}

Chunk::Chunk(std::ofstream &file, uint32_t id, Chunk *parent) :
    ofile(&file), ifile(0), id(id), size(0), offset(file.tellp()), counter(0),
    parent(parent)
{
    if (parent)
        parent->allocate(8);
    writeHeader();
}

Chunk::Chunk(std::ofstream &file, uint32_t id) :
    ofile(&file), ifile(0), id(id), size(0), offset(file.tellp()), counter(0)
{
    writeHeader();
}

void Chunk::parseSubchunks() {
    while (counter < size) {
        ifile->seekg(offset+8+counter);
        subchunks.push_back(Chunk(*ifile));
        counter += 8+subchunks.back().size;
    }
}

void Chunk::read(void *output, size_t count) {
    ifile->seekg(offset+8+counter);
    ifile->read((char *)output, count);
    counter += count;
}

void Chunk::write(void *input, size_t count) {
    ofile->seekp(offset+8+counter);
    allocate(count);
    ofile->write((char *)input, count);
}

void Chunk::allocate(size_t count) {
    if (parent)
        parent->allocate(count);
    counter += count;
}

Chunk &Chunk::addSubchunk(uint32_t id) {
    subchunks.push_back(Chunk(*ofile, id, this));
    return subchunks.back();
}

void Chunk::close() {
    if (ofile)
        writeHeader();
    subchunks.clear();
}

Chunk::~Chunk() {
    close();
}
