CC=clang++
CFLAGS=-c -I/opt/local/include -g
LDFLAGS=-L/opt/local/lib -g
SOURCES=wifi80211.cpp constellation.cpp qam.cpp blurt.cpp cc.cpp
OBJECTS=$(SOURCES:.cpp=.o)

all: blurt

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

blurt: $(OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

clean:
	rm $(OBJECTS) blurt

.PHONY: clean
