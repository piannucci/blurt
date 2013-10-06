#include <poll.h>
#include <string>
#include <cstdlib>
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

double Fs = 48000.;
double Fc = 19000.;
int upsample_factor = 16;

WiFi80211 wifi;

void handle_packets_thread(audioFIFO * fifo, TapDevice * tap_device) {
    // if we have string f = some frame, call tap_device.write(f);
    DecodeResult result;
    while ( 1 ) {
        fifo->pop_front(result);
        tap_device->write(std::string(result.payload.begin(), result.payload.end()));
    }
}

int main(int argc, char **argp, char **envp) {
    try{
        audioFIFO fifo(Fs, Fc, upsample_factor, wifi);
        TapDevice tap_device("blurt");

        std::thread(handle_packets_thread, &fifo, &tap_device).detach();

        // set up poll for tap devices
        struct pollfd pollfds[ 1 ];
        pollfds[ 0 ].fd = tap_device.fd();
        pollfds[ 0 ].events = POLLIN;

        // poll on the tap device
        while ( 1 ) {
           // set poll wait time for each queue to time before head packet must be sent
            if( poll( pollfds, 1, -1 ) == -1 ) {
                perror( "poll" );
                return EXIT_FAILURE;
            }
            tap_device.read();
            // read from ingress which triggered POLLIN
            if ( pollfds[ 0 ].revents & POLLIN )
                fifo.push_back(frame{tap_device.read(), 0, 15.});
        }
        return 0;
    } catch (const char *e) {
        std::cerr << e << std::endl;
    }
}
