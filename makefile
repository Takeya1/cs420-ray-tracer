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
	@echo -n "Serial: "; ./ray_serial | grep "Serial time"
	@echo -n "OpenMP: "; ./ray_openmp | grep "OpenMP time"
	convert output_serial.ppm output_serial.png
	convert output_openmp.ppm output_openmp.png
