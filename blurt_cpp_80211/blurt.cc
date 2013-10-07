#include <poll.h>
#include <string>
#include <cstdlib>
#if __APPLE__
#include <sys/select.h>
#endif
#include <ctime>

#include "blurt.h"
#include "util.h"
#include "wave.h"
#include "upsample.h"
#include "wifi80211.h"
#include "audioLoopback.h"
#include "tapdevice.h"

#include "ofdm.h"
#include "crc.h"
#include "iir.h"

const double Fs = 48000.;
const double Fc_out = 18000.;
const double Fc_in = 10000.;
const size_t upsample_factor = 8;
const size_t rate = 0;
const double gain = 15.;

WiFi80211 wifi;

void handle_packets_thread(audioFIFO * fifo, TapDevice * tap_device) [[noreturn]] {
    // if we have string f = some frame, call tap_device.write(f);
    DecodeResult result;
    while ( 1 ) {
        fifo->pop_front(result);
        tap_device->write(std::string(result.payload.begin(), result.payload.end()));
    }
}

int main(int, char **, char **) {
    try{
        audioFIFO fifo(Fs, Fc_out, Fc_in, upsample_factor, wifi);
        TapDevice tap_device("blurt");

        std::thread(handle_packets_thread, &fifo, &tap_device).detach();

        int fd = tap_device.fd();
#if defined(__linux__)
        // set up poll for tap devices
        struct pollfd pollfds[ 1 ];
        pollfds[ 0 ].fd = fd;
        pollfds[ 0 ].events = POLLIN;
#endif

        // poll on the tap device
        while ( 1 ) {
#if defined(__linux__)
            if( poll( pollfds, 1, -1 ) == -1 ) {
                perror( "poll" );
                return EXIT_FAILURE;
            }
            if (!( pollfds[ 0 ].revents & POLLIN ))
                continue;
#elif defined(__APPLE__)
            fd_set fds;
            FD_ZERO(&fds);
            FD_SET(fd, &fds);
            int ret;
            if ( (ret = select( fd+1, &fds, NULL, NULL, NULL )) < 0 ) {
                perror( "select" );
                return EXIT_FAILURE;
            }
            if (ret == 0 || !FD_ISSET(fd, &fds))
                continue;
#endif
            fifo.push_back(frame{tap_device.read(), rate, gain});
        }
    } catch (const char *e) {
        std::cerr << "Exception: " << e << std::endl;
    }
    return 0;
}
