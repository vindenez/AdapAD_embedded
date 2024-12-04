CXX = g++
CXXFLAGS = -g -std=c++11 -Wall -O2

# Include paths
INCLUDES = -Iinclude

# Architecture-specific flags
ARCH := $(shell uname -m)
ifeq ($(ARCH),arm64)
    # For M1 Mac (ARM64)
    CXXFLAGS += -march=armv8-a+fp+simd+crypto+crc
    GTEST_ROOT = /opt/homebrew
else ifeq ($(ARCH),armv7l)
    # For ARM32 (like Cortex-A7)
    CXXFLAGS += -march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard
    GTEST_ROOT = /usr/local
else
    # For Intel Mac
    GTEST_ROOT = /usr/local
endif

# Build directory for object files
BUILD_DIR = build/src
$(shell mkdir -p $(BUILD_DIR))

# Source files
SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC))

# Targets
TARGET = adapad

# Main program
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -rf build
	rm -f $(TARGET)

.PHONY: all clean
