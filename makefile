CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall

# Detect OS and set appropriate OpenMP flags
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    # macOS - requires libomp from Homebrew
    OMPFLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
else
    # Linux - simple flag
    OMPFLAGS = -fopenmp
endif

NVCC = /usr/local/cuda/bin/nvcc
CUDAFLAGS = -O3 -arch=sm_60 -ccbin /usr/bin/gcc

# Define source and include directories
SRCDIR = src
INCDIR = include

# Add the include path to CXXFLAGS and CUDAFLAGS
CXXFLAGS += -I$(INCDIR)
CUDAFLAGS += -I$(INCDIR)


# Week 1 targets
serial: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) -o ray_serial $(SRCDIR)/main.cpp

openmp: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o ray_openmp $(SRCDIR)/main.cpp

# Week 2 target
cuda: $(SRCDIR)/main_gpu.cu
	$(NVCC) $(CUDAFLAGS) -lstdc++ -lm -o ray_cuda $(SRCDIR)/main_gpu.cu

# Week 3 target
hybrid: src/kernel.cu src/main_hybrid.cpp
	$(NVCC) -O3 -arch=sm_60 -Iinclude -c src/kernel.cu -o kernel.o
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c src/main_hybrid.cpp -o main_hybrid.o
	$(NVCC) -O3 -arch=sm_60 -Xcompiler -fopenmp kernel.o main_hybrid.o -o ray_hybrid -lgomp

# Build all targets
all: serial openmp cuda hybrid

clean:
	rm -f ray_serial ray_openmp ray_cuda ray_hybrid *.o *.ppm *.png

# Quick test for hybrid
test_hybrid: hybrid
	@echo "=== Testing Hybrid Ray Tracer ==="
	./ray_hybrid scenes/simple.txt -o output_hybrid_simple.ppm -t 64
	./ray_hybrid scenes/medium.txt -o output_hybrid_medium.ppm -t 64
	./ray_hybrid scenes/complex.txt -o output_hybrid_complex.ppm -t 64
	@echo "=== Hybrid tests complete ==="

test: serial
	./ray_serial
	@echo "Check output_serial.ppm"

# Full benchmark including hybrid
benchmark: serial openmp cuda hybrid
	@echo "=== Performance Comparison ==="
	@echo ""
	@echo "--- Simple Scene ---"
	@bash -c 'SERIAL_TIME=$$(./ray_serial scenes/simple.txt 2>/dev/null | grep "Serial time" | awk "{print \$$3}"); \
	OPENMP_TIME=$$(./ray_openmp scenes/simple.txt 2>/dev/null | grep "OpenMP time" | awk "{print \$$3}"); \
	CUDA_TIME=$$(./ray_cuda scenes/simple.txt 2>/dev/null | grep "GPU rendering time" | awk "{print \$$4}"); \
	HYBRID_TIME=$$(./ray_hybrid scenes/simple.txt -o output_hybrid.ppm 2>/dev/null | grep "Hybrid rendering time" | awk "{print \$$4}"); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	echo "CUDA:   $$CUDA_TIME seconds"; \
	echo "Hybrid: $$HYBRID_TIME seconds"; \
	OPENMP_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v o=$$OPENMP_TIME "BEGIN {printf \"%.2f\", s/o}"); \
	CUDA_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v c=$$CUDA_TIME "BEGIN {printf \"%.2f\", s/c}"); \
	HYBRID_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v h=$$HYBRID_TIME "BEGIN {printf \"%.2f\", s/h}"); \
	echo "OpenMP Speedup: $${OPENMP_SPEEDUP}x"; \
	echo "CUDA Speedup:   $${CUDA_SPEEDUP}x"; \
	echo "Hybrid Speedup: $${HYBRID_SPEEDUP}x"'
	@mv output_serial.ppm output_simple_serial.ppm 2>/dev/null || true
	@mv output_openmp.ppm output_simple_openmp.ppm 2>/dev/null || true
	@mv output_gpu.ppm output_simple_cuda.ppm 2>/dev/null || true
	@mv output_hybrid.ppm output_simple_hybrid.ppm 2>/dev/null || true
	@magick output_simple_serial.ppm output_simple_serial.png 2>/dev/null || true
	@magick output_simple_openmp.ppm output_simple_openmp.png 2>/dev/null || true
	@magick output_simple_cuda.ppm output_simple_cuda.png 2>/dev/null || true
	@magick output_simple_hybrid.ppm output_simple_hybrid.png 2>/dev/null || true
	@echo ""
	@echo "--- Medium Scene ---"
	@bash -c 'SERIAL_TIME=$$(./ray_serial scenes/medium.txt 2>/dev/null | grep "Serial time" | awk "{print \$$3}"); \
	OPENMP_TIME=$$(./ray_openmp scenes/medium.txt 2>/dev/null | grep "OpenMP time" | awk "{print \$$3}"); \
	CUDA_TIME=$$(./ray_cuda scenes/medium.txt 2>/dev/null | grep "GPU rendering time" | awk "{print \$$4}"); \
	HYBRID_TIME=$$(./ray_hybrid scenes/medium.txt -o output_hybrid.ppm 2>/dev/null | grep "Hybrid rendering time" | awk "{print \$$4}"); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	echo "CUDA:   $$CUDA_TIME seconds"; \
	echo "Hybrid: $$HYBRID_TIME seconds"; \
	OPENMP_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v o=$$OPENMP_TIME "BEGIN {printf \"%.2f\", s/o}"); \
	CUDA_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v c=$$CUDA_TIME "BEGIN {printf \"%.2f\", s/c}"); \
	HYBRID_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v h=$$HYBRID_TIME "BEGIN {printf \"%.2f\", s/h}"); \
	echo "OpenMP Speedup: $${OPENMP_SPEEDUP}x"; \
	echo "CUDA Speedup:   $${CUDA_SPEEDUP}x"; \
	echo "Hybrid Speedup: $${HYBRID_SPEEDUP}x"'
	@mv output_serial.ppm output_medium_serial.ppm 2>/dev/null || true
	@mv output_openmp.ppm output_medium_openmp.ppm 2>/dev/null || true
	@mv output_gpu.ppm output_medium_cuda.ppm 2>/dev/null || true
	@mv output_hybrid.ppm output_medium_hybrid.ppm 2>/dev/null || true
	@magick output_medium_serial.ppm output_medium_serial.png 2>/dev/null || true
	@magick output_medium_openmp.ppm output_medium_openmp.png 2>/dev/null || true
	@magick output_medium_cuda.ppm output_medium_cuda.png 2>/dev/null || true
	@magick output_medium_hybrid.ppm output_medium_hybrid.png 2>/dev/null || true
	@echo ""
	@echo "--- Complex Scene ---"
	@bash -c 'SERIAL_TIME=$$(./ray_serial scenes/complex.txt 2>/dev/null | grep "Serial time" | awk "{print \$$3}"); \
	OPENMP_TIME=$$(./ray_openmp scenes/complex.txt 2>/dev/null | grep "OpenMP time" | awk "{print \$$3}"); \
	CUDA_TIME=$$(./ray_cuda scenes/complex.txt 2>/dev/null | grep "GPU rendering time" | awk "{print \$$4}"); \
	HYBRID_TIME=$$(./ray_hybrid scenes/complex.txt -o output_hybrid.ppm 2>/dev/null | grep "Hybrid rendering time" | awk "{print \$$4}"); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	echo "CUDA:   $$CUDA_TIME seconds"; \
	echo "Hybrid: $$HYBRID_TIME seconds"; \
	OPENMP_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v o=$$OPENMP_TIME "BEGIN {printf \"%.2f\", s/o}"); \
	CUDA_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v c=$$CUDA_TIME "BEGIN {printf \"%.2f\", s/c}"); \
	HYBRID_SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v h=$$HYBRID_TIME "BEGIN {printf \"%.2f\", s/h}"); \
	echo "OpenMP Speedup: $${OPENMP_SPEEDUP}x"; \
	echo "CUDA Speedup:   $${CUDA_SPEEDUP}x"; \
	echo "Hybrid Speedup: $${HYBRID_SPEEDUP}x"'
	@mv output_serial.ppm output_complex_serial.ppm 2>/dev/null || true
	@mv output_openmp.ppm output_complex_openmp.ppm 2>/dev/null || true
	@mv output_gpu.ppm output_complex_cuda.ppm 2>/dev/null || true
	@mv output_hybrid.ppm output_complex_hybrid.ppm 2>/dev/null || true
	@magick output_complex_serial.ppm output_complex_serial.png 2>/dev/null || true
	@magick output_complex_openmp.ppm output_complex_openmp.png 2>/dev/null || true
	@magick output_complex_cuda.ppm output_complex_cuda.png 2>/dev/null || true
	@magick output_complex_hybrid.ppm output_complex_hybrid.png 2>/dev/null || true
	@echo ""
	@echo "=== Benchmark Complete ==="

# Tile size comparison for hybrid
benchmark_tiles: hybrid
	@echo "=== Hybrid Tile Size Comparison (Complex Scene) ==="
	@for tile in 32 64 128 256; do \
		echo ""; \
		echo "--- Tile size: $$tile ---"; \
		./ray_hybrid scenes/complex.txt -o output_tile_$$tile.ppm -t $$tile; \
	done
	@echo ""
	@echo "=== Tile Comparison Complete ==="

.PHONY: all clean test test_hybrid benchmark benchmark_tiles