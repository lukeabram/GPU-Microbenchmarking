# Builds CUDA code 
NVCC ?= nvcc
ARCH ?= native
CXXSTD ?= c++20
OPTFLAGS ?= -O3
LIBS      :=

.PHONY: all clean

all: analyze

analyze: analyze.cu utilities.h
	$(NVCC) $(OPTFLAGS) -std=$(CXXSTD) -arch=$(ARCH) -o $@ $< $(LIBS)

clean:
	rm -f analyze
