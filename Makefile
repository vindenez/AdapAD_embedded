CXX = g++
CXXFLAGS = -g -std=c++17 -Iinclude -Wall -O3

# Architecture-specific flags
ARCH := $(shell uname -m)
ifeq ($(ARCH),arm64)
    # For M1 Mac (ARM64)
    CXXFLAGS += -march=armv8-a+fp+simd+crypto+crc
else ifeq ($(ARCH),armv7l)
    # For ARM32 (like Cortex-A7)
    CXXFLAGS += -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
else
    # For other architectures, no special flags
endif

SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)

TARGET = adapad

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)
