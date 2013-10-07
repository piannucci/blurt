#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>

#if defined(__linux__)
#include <linux/if.h>
#include <linux/if_tun.h>
#elif defined(__APPLE__)
#include <net/if.h>
#endif

#include <string>
#include <cstdlib>
#include <sys/ioctl.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include "tapdevice.h"

using namespace std;

TapDevice::TapDevice( std::string name )
  : fd_()
{
    struct ifreq ifr;
    int err;

    // fills enough of memory area pointed to by ifr
    memset( &ifr, 0, sizeof( ifr ) );

    // uses specified name for tap device
    strncpy( ifr.ifr_name, name.c_str(), IFNAMSIZ );

#if defined(__linux__)
    if ( ( fd_ = open( "/dev/net/tun", O_RDWR ) ) < 0 ) {
        throw "open";
    }

    // specifies to create tap device
    ifr.ifr_flags = IFF_TAP;

    // create interface
    if ( ( err = ioctl( fd_, TUNSETIFF, ( void * ) &ifr ) ) < 0 ) {
        close( fd_ );
        throw "ioctl failed.  You probably forgot to sudo.";
    }
#elif defined(__APPLE__)
    for (int i=0; i<16; i++) {
        name = "/dev/tap" + std::to_string(i);
        if ( ( fd_ = open( name.c_str(), O_RDWR ) ) >= 0 )
            break;
    }

    if (fd_ < 0) {
        if (errno == EPERM || errno == EACCES)
            throw "open failed.  You probably forgot to sudo.";
        else if (errno == ENOENT)
            throw "open failed.  You probably need to kextload tuntaposx.";
        else
            throw strdup((std::string("open: ") + strerror(errno)).c_str());
    }

    strncpy( ifr.ifr_name, name.c_str() + 5, IFNAMSIZ );
#endif

    int sockfd = socket( AF_INET, SOCK_DGRAM, 0 );

    if ( ( err = ioctl( sockfd, SIOCGIFFLAGS, ( void * ) &ifr ) ) < 0 ) {
        close( sockfd );
        std::cerr << strerror(errno) << std::endl;
        throw "ioctl failed; could not get interface flags";
    }

    // add flag to bring interface up
    ifr.ifr_flags += IFF_UP;

    if ( ( err = ioctl( sockfd, SIOCSIFFLAGS, ( void * ) &ifr ) ) < 0 ) {
        close( sockfd );
        throw "ioctl failed; could not set interface flags";
    }

#if defined(__APPLE__)
    const char *addr = "10.0.0.0", *mask = "255.0.0.0";

    struct ifreq ridreq;
    bzero(&ridreq, sizeof(ridreq));
    strncpy(ridreq.ifr_name, name.c_str()+5, sizeof(ridreq.ifr_name));

    if (ioctl(sockfd, SIOCDIFADDR, &ridreq) < 0)
        if (errno != EADDRNOTAVAIL) // no previous address for interface
            throw "ioctl (SIOCDIFADDR)";

    struct ifaliasreq addreq;
    bzero(&addreq, sizeof(addreq));

	addreq.ifra_addr.sa_len = sizeof(addreq.ifra_addr.sa_len);
    addreq.ifra_addr.sa_family = AF_INET;
	inet_aton(addr, &((struct sockaddr_in *)&addreq.ifra_addr)->sin_addr);

	addreq.ifra_mask.sa_len = sizeof(addreq.ifra_mask);
	inet_aton(mask, &((struct sockaddr_in *)&addreq.ifra_mask)->sin_addr);
    strncpy(addreq.ifra_name, name.c_str()+5, sizeof(addreq.ifra_name));

    if (ioctl(sockfd, SIOCAIFADDR, &addreq) < 0) {
        std::cerr << strerror(errno) << std::endl;
        throw "ioctl (SIOCAIFADDR)";
    }
#endif

    close( sockfd );
}

TapDevice::~TapDevice()
{
    if ( close( fd_ ) < 0 ) {
	throw "close";
    }
}
