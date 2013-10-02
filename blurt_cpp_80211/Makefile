# compile to native
CC=g++-mp-4.7
OPTFLAGS=-O4 -ffast-math -funsafe-loop-optimizations -fipa-matrix-reorg -flto -fno-math-errno -funsafe-math-optimizations -fcx-fortran-rules -funroll-loops -fprofile-use
CFLAGS=-c -I/opt/local/include $(OPTFLAGS)
LDFLAGS=-L/opt/local/lib $(OPTFLAGS) -fwhole-program

# compile to JavaScript
EMCC=em++
EMOPTFLAGS=-O2 -g0 --closure 1 --js-opts 1 --llvm-lto 3 #-s PRECISE_I64_MATH=0
EMCFLAGS=-v $(EMOPTFLAGS) -fno-exceptions

SOURCES=wifi80211.cpp constellation.cpp qam.cpp blurt.cpp cc.cpp \
		puncture.cpp crc.cpp util.cpp interleave.cpp scrambler.cpp \
		fft.cpp iir.cpp upsample.cpp ofdm.cpp rate.cpp kalman.cpp \
		chunk.cpp wave.cpp mkfilter.cpp
OBJECTS=$(SOURCES:.cpp=.o)

all: blurt

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

blurt: $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

blurt.js: $(SOURCES)
	$(EMCC) $(EMCFLAGS) $^ -o $@

clean:
	rm -rf $(OBJECTS) blurt blurt.js *.js *.js.map

.PHONY: clean
