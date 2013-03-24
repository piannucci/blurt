CC=clang++
CFLAGS=-c -I/opt/local/include -O3 -g
LDFLAGS=-L/opt/local/lib -O3 -g
SOURCES=wifi80211.cpp constellation.cpp qam.cpp blurt.cpp cc.cpp puncture.cpp crc.cpp util.cpp
OBJECTS=$(SOURCES:.cpp=.o)

all: blurt

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

blurt: $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

clean:
	rm $(OBJECTS) blurt

.PHONY: clean
