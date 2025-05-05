CXX = g++
CXXFLAGS = -g -std=c++11 -Wall

# Include paths
INCLUDES = -Iinclude

# Build modes
ifeq ($(DEBUG),1)
    CXXFLAGS += -O0 -DDEBUG
else
    CXXFLAGS += -O2
endif

# Memory analysis options
ifeq ($(MEMCHECK),valgrind)
    # For Valgrind compatibility (Linux only)
    CXXFLAGS += -fno-inline -fno-omit-frame-pointer
else ifeq ($(MEMCHECK),asan)
    # For AddressSanitizer
    CXXFLAGS += -fsanitize=address -fno-omit-frame-pointer
else ifeq ($(MEMCHECK),lsan)
    # For LeakSanitizer
    CXXFLAGS += -fsanitize=leak -fno-omit-frame-pointer
endif

# Architecture and OS detection
ARCH := $(shell uname -m)
OS := $(shell uname -s)

# Architecture-specific flags
ifeq ($(OS),Darwin)
    # macOS settings
    ifeq ($(ARCH),arm64)
        # For M1/M2 Mac (ARM64)
        CXXFLAGS += -march=armv8-a+fp+simd+crypto+crc+fp16
        GTEST_ROOT = /opt/homebrew
    else
        # For Intel Mac
        CXXFLAGS += -march=native
        GTEST_ROOT = /usr/local
    endif
    # macOS doesn't support Valgrind
    VALGRIND_AVAILABLE = 0
else ifeq ($(OS),Linux)
    # Linux settings
    ifeq ($(ARCH),x86_64)
        # For 64-bit Linux
        CXXFLAGS += -march=native
    endif
    GTEST_ROOT = /usr/local
    VALGRIND_AVAILABLE = 1
endif

# Build directory for object files
BUILD_DIR = build/$(OS)_$(ARCH)/src
$(shell mkdir -p $(BUILD_DIR))

# Source files
SRC = $(wildcard src/*.cpp)
OBJ = $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC))

# Target executable
TARGET = adapad

# Main program
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Memory analysis targets
ifeq ($(VALGRIND_AVAILABLE),1)
# Linux-only targets
valgrind: $(TARGET)
	valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./$(TARGET)

massif: $(TARGET)
	valgrind --tool=massif ./$(TARGET)
	@echo "Run 'ms_print massif.out.*' to view the report"
else
# Alternative for macOS
valgrind:
	@echo "Valgrind is not available on macOS. Consider using AddressSanitizer instead:"
	@echo "make clean && make MEMCHECK=asan"

massif:
	@echo "Massif is not available on macOS. Consider using Instruments app instead."
endif

# Run with AddressSanitizer
run-asan: $(TARGET)
	ASAN_OPTIONS=detect_leaks=1 ./$(TARGET)

# Clean
clean:
	rm -rf build
	rm -f $(TARGET)

.PHONY: all clean valgrind massif run-asan