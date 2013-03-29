CC=clang++
CFLAGS=-c -I/opt/local/include -O4
LDFLAGS=-L/opt/local/lib -O4
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

clean:
	rm $(OBJECTS) blurt

.PHONY: clean
