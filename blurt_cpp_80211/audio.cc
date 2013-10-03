#include <cstdlib>
#include <cstdint>
#include <cmath>
#include "math.h"
#include "portaudio.h"

static const double Fs=48000.0;
static const double dt=1./Fs, omega=2*M_PI*440.;
static const double dp=omega*dt;
static const int inChannels = 2;
static const int outChannels = 2;

static int myCallback( const void *inputBuffer, void *outputBuffer, size_t
        framesPerBuffer, const PaStreamCallbackTimeInfo* timeInfo,
        PaStreamCallbackFlags statusFlags, void *userData ) {
    const float *in  = (const float *)inputBuffer;
	float *out = (float *)outputBuffer;    

    double phase = fmod(timeInfo->outputBufferDacTime * omega, 2*M_PI);

    for(size_t i=0; i<framesPerBuffer; i++) {
        //leftInput = *in++;
        //rightInput = *in++;
        *out++ = sin(phase); //leftInput * rightInput;
        *out++ = sin(phase); //0.5f * (leftInput + rightInput);
        phase += dp;
    }
    return 0;
}

int main(void) {
    PaStream *stream;
    Pa_Initialize();
    Pa_OpenDefaultStream(&stream, inChannels, outChannels, paFloat32, Fs, 0, myCallback, NULL);
    Pa_StartStream(stream);
    Pa_Sleep(10000);
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();
    return 0;
}
