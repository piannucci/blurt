#ifndef tapdevice_h
#define tapdevice_h

#include <string>
#include <stdint.h>
#include "ezio.h"

class TapDevice
{
    private:
        int fd_;
    public:
        TapDevice( std::string name );
        ~TapDevice( );
        int fd() const { return fd_; }
        void write( const std::string & buf ) const { writeall(fd_, buf); }
        std::string read() const { std::string buf = readall( fd_ ); return buf;}

};

#endif
