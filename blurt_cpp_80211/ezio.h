#ifndef EZIO_H
#define EZIO_H

#include <string>

std::string readall( const int fd );
void writeall( const int fd, const std::string & buf );

#endif
