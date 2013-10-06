#ifndef tapdevice_h
#define tapdevice_h

#include <string>
#include <stdint.h>
#include "ezio.h"

using namespace std;

class TapDevice
{
    private:
        int fd_;
    public:
        TapDevice( std::string name );
        ~TapDevice( );
        int fd() const { return fd_; }
        void write( const string & buf ) const { writeall(fd_, buf); }
        string read() const { string buf = readall( fd_ ); return buf;}

};

#endif
