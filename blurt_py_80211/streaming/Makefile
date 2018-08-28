PYTHON ?= python3.7
BLITZ_PREFIX := /opt/local
INCLUDES := -I$(BLITZ_PREFIX)/include/ -I$(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
CFLAGS := $(shell $(PYTHON)-config --cflags) $(INCLUDES) -Wall -Wstrict-prototypes -std=c++14
LDFLAGS := $(shell $(PYTHON)-config --ldflags) -fPIC -shared

all: kernels.so

kernels.so: kernels.cpp
	clang++ $^ $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f kernels.so

.PHONY: all clean
