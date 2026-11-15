NVCC ?= nvcc
GENCODE ?= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75
NVFLAGS ?= -O3 -std=c++17 -lineinfo $(GENCODE) -Xcompiler -Wall
HEADERS := src/kernels.cuh src/common.cuh
# NVML (clock and power readings) links against the toolkit's stub; the driver supplies the library
CUDA_HOME ?= $(shell dirname $$(dirname $$(which $(NVCC))))

all: build/gemm_test build/gemm_bench

build/gemm_test: src/test.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas

build/gemm_bench: src/bench.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) $< -o $@ -lcublas -L$(CUDA_HOME)/lib64/stubs -lnvidia-ml

# registers and spills per kernel, from the assembler
build/ptxas.txt: src/bench.cu $(HEADERS) | build
	$(NVCC) $(NVFLAGS) -Xptxas -v -c $< -o build/ptxas.o 2> $@

build:
	mkdir -p build

clean:
	rm -rf build

.PHONY: all clean
