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

# Google Test configuration
GTEST_INCLUDES = -I$(GTEST_ROOT)/include
GTEST_LIBS = -L$(GTEST_ROOT)/lib -lgtest -lgtest_main

# Source files
SRC = $(wildcard src/*.cpp)
OBJ = $(SRC:.cpp=.o)

# Test source files
TEST_SRC = $(wildcard tests/*.cpp)
TEST_OBJ = $(TEST_SRC:.cpp=.o)

# Targets
TARGET = adapad
TEST_TARGET = run_tests

# Main program
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ)

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

$(TEST_TARGET): $(filter-out src/main.o, $(OBJ)) $(TEST_OBJ)
	$(CXX) $(CXXFLAGS) -o $(TEST_TARGET) $^ $(GTEST_LIBS)

tests/%.o: tests/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(GTEST_INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(OBJ) $(TEST_OBJ) $(TARGET) $(TEST_TARGET)

.PHONY: all test clean
