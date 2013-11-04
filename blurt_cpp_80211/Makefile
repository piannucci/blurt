# compile to native
OPTFLAGS?=-O4 -ffast-math -funsafe-loop-optimizations -fipa-matrix-reorg -flto -fno-math-errno -funsafe-math-optimizations -fcx-fortran-rules -funroll-loops #-fprofile-use
CXXFLAGS?=$(OPTFLAGS) -std=c++11 -MMD -ggdb
LDFLAGS?=$(OPTFLAGS) -std=c++11 -ggdb -pthread
LDLIBS?=-lportaudio -lm

# compile to JavaScript
EMCC=em++
EMOPTFLAGS=-O2 -g0 --closure 1 --js-opts 1 --llvm-lto 3 #-s PRECISE_I64_MATH=0
EMCFLAGS=-v $(EMOPTFLAGS) -fno-exceptions

EMSDK=$(pushd /opt/local/emscripten/; ./emsdk active_path; popd)

SOURCES=wifi80211.cc constellation.cc qam.cc blurt.cc cc.cc \
	puncture.cc crc.cc util.cc interleave.cc scrambler.cc \
	fft.cc iir.cc upsample.cc ofdm.cc rate.cc kalman.cc \
	chunk.cc wave.cc mkfilter.cc audioLoopback.cc pa_ringbuffer.cc \
	tapdevice.cc ezio.cc #frame_decoder.cc

OBJECTS=$(patsubst %.c,%.o,$(SOURCES:.cc=.o))

all: blurt

blurt: $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(LDLIBS) -o $@

blurt.js: $(SOURCES)
	PATH=$(EMSDK):$(PATH) $(EMCC) $(EMCFLAGS) $^ -o $@

-include *.d

.PHONY: clean
clean:
	rm -rf $(OBJECTS) blurt blurt.js *.js *.js.map *.d *.plist
