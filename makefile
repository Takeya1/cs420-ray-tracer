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

NVCC = nvcc
CUDAFLAGS = -O3 -arch=sm_60

# Define source and include directories
SRCDIR = src
INCDIR = include

# Add the include path to CXXFLAGS and CUDAFLAGS
# The -I flag tells the compiler to look in $(INCDIR) for header files
CXXFLAGS += -I$(INCDIR)
CUDAFLAGS += -I$(INCDIR)


# Week 1 targets
serial: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) -o ray_serial $(SRCDIR)/main.cpp

openmp: $(SRCDIR)/main.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o ray_openmp $(SRCDIR)/main.cpp

# Week 2 target (placeholder)
cuda: $(SRCDIR)/main_gpu.cu
	$(NVCC) $(CUDAFLAGS) -o ray_cuda $(SRCDIR)/main_gpu.cu

# Week 3 target (placeholder)
hybrid: $(SRCDIR)/main_hybrid.cpp $(SRCDIR)/kernel.cu
	$(NVCC) $(CUDAFLAGS) -c $(SRCDIR)/kernel.cu
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -c $(SRCDIR)/main_hybrid.cpp
	$(NVCC) $(CUDAFLAGS) kernel.o main_hybrid.o -o ray_hybrid

clean:
	rm -f ray_serial ray_openmp ray_cuda ray_hybrid *.o *.ppm *.png

test: serial
	./ray_serial
	@echo "Check output_serial.ppm"

benchmark: serial openmp
	@echo "=== Performance Comparison ==="
	@echo ""
	@echo "--- Simple Scene ---"
	@bash -c 'SERIAL_TIME=$$(./ray_serial scenes/simple.txt 2>/dev/null | grep "Serial time" | awk "{print \$$3}"); \
	OPENMP_TIME=$$(./ray_openmp scenes/simple.txt 2>/dev/null | grep "OpenMP time" | awk "{print \$$3}"); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v o=$$OPENMP_TIME "BEGIN {printf \"%.2f\", s/o}"); \
	echo "Speedup: $${SPEEDUP}x"'
	@mv output_serial.ppm output_simple_serial.ppm 2>/dev/null || true
	@mv output_openmp.ppm output_simple_openmp.ppm 2>/dev/null || true
	@magick output_simple_serial.ppm output_simple_serial.png 2>/dev/null || true
	@magick output_simple_openmp.ppm output_simple_openmp.png 2>/dev/null || true
	@echo ""
	@echo "--- Medium Scene ---"
	@bash -c 'SERIAL_TIME=$$(./ray_serial scenes/medium.txt 2>/dev/null | grep "Serial time" | awk "{print \$$3}"); \
	OPENMP_TIME=$$(./ray_openmp scenes/medium.txt 2>/dev/null | grep "OpenMP time" | awk "{print \$$3}"); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v o=$$OPENMP_TIME "BEGIN {printf \"%.2f\", s/o}"); \
	echo "Speedup: $${SPEEDUP}x"'
	@mv output_serial.ppm output_medium_serial.ppm 2>/dev/null || true
	@mv output_openmp.ppm output_medium_openmp.ppm 2>/dev/null || true
	@magick output_medium_serial.ppm output_medium_serial.png 2>/dev/null || true
	@magick output_medium_openmp.ppm output_medium_openmp.png 2>/dev/null || true
	@echo ""
	@echo "--- Complex Scene ---"
	@bash -c 'SERIAL_TIME=$$(./ray_serial scenes/complex.txt 2>/dev/null | grep "Serial time" | awk "{print \$$3}"); \
	OPENMP_TIME=$$(./ray_openmp scenes/complex.txt 2>/dev/null | grep "OpenMP time" | awk "{print \$$3}"); \
	echo "Serial: $$SERIAL_TIME seconds"; \
	echo "OpenMP: $$OPENMP_TIME seconds"; \
	SPEEDUP=$$(awk -v s=$$SERIAL_TIME -v o=$$OPENMP_TIME "BEGIN {printf \"%.2f\", s/o}"); \
	echo "Speedup: $${SPEEDUP}x"'
	@mv output_serial.ppm output_complex_serial.ppm 2>/dev/null || true
	@mv output_openmp.ppm output_complex_openmp.ppm 2>/dev/null || true
	@magick output_complex_serial.ppm output_complex_serial.png 2>/dev/null || true
	@magick output_complex_openmp.ppm output_complex_openmp.png 2>/dev/null || true
	@echo ""
	@echo "=== Benchmark Complete ==="
