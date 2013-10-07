#include "chunk.h"
#include <fstream>

Chunk::Chunk(std::ifstream &file) :
    offset(file.is_open() ? (uint32_t)file.tellg() : 0), counter(0), parent(0), ifile(&file), ofile(0)
{
    id.resize(4);
    this->ifile->read(&id[0], 4);
    this->ifile->read((char *)&size, 4);
}

void Chunk::writeHeader() {
    this->ofile->seekp(offset);
    this->ofile->write(&id[0], 4);
    this->ofile->write((char *)&size, 4);
}

Chunk::Chunk(std::ofstream &file, std::string id_, Chunk *parent_) :
    offset((uint32_t)file.tellp()), counter(0), parent(parent_), id(id_), size(0), ifile(0), ofile(&file)
{
    if (parent)
        parent->allocate(8);
    writeHeader();
}

Chunk::Chunk(std::ofstream &file, std::string id_) :
    offset((uint32_t)file.tellp()), counter(0), parent(0), id(id_), size(0), ifile(0), ofile(&file)
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
    ifile->read((char *)output, (std::streamsize)count);
    counter += count;
}

void Chunk::write(void *input, size_t count) {
    ofile->seekp(offset+8+counter);
    allocate(count);
    ofile->write((char *)input, (std::streamsize)count);
}

void Chunk::allocate(size_t count) {
    if (parent)
        parent->allocate(count);
    counter += count;
}

Chunk &Chunk::addSubchunk(std::string id_) {
    subchunks.push_back(Chunk(*ofile, id_, this));
    return subchunks.back();
}

void Chunk::close() {
    if (size < counter)
        size = counter;
    if (ofile)
        writeHeader();
    subchunks.clear();
}

Chunk::~Chunk() {
    close();
}
