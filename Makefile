

.PHONY: all configure build test example sanitize benchmark benchmark-save clean

BUILD_DIR ?= build
CMAKE ?= cmake
CLANG ?= $(shell command -v clang-23 2>/dev/null || command -v clang-20 2>/dev/null || command -v clang-19 2>/dev/null || command -v clang-18 2>/dev/null || command -v clang)
BENCH_DIR ?= build-bench
BENCH_ARGS ?=
BENCH_OUT ?= $(BENCH_DIR)/results.csv

all: test

configure:
	$(CMAKE) --fresh -S . -B $(BUILD_DIR) -G Ninja -DCMAKE_C_COMPILER=$(CLANG) -DCMAKE_BUILD_TYPE=Debug

build: configure
	$(CMAKE) --build $(BUILD_DIR)

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

example: build
	./$(BUILD_DIR)/mantix_example

sanitize:
	$(CMAKE) --fresh -S . -B $(BUILD_DIR)-sanitize -G Ninja -DCMAKE_C_COMPILER=$(CLANG) -DCMAKE_BUILD_TYPE=Debug -DMANTIX_ENABLE_SANITIZERS=ON
	$(CMAKE) --build $(BUILD_DIR)-sanitize
	ctest --test-dir $(BUILD_DIR)-sanitize --output-on-failure

benchmark:
	$(CMAKE) --fresh -S . -B $(BENCH_DIR) -G Ninja -DCMAKE_C_COMPILER=$(CLANG) -DCMAKE_BUILD_TYPE=Release -DMANTIX_BUILD_BENCHMARKS=ON -DMANTIX_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
	$(CMAKE) --build $(BENCH_DIR) --target mantix_bench
	./$(BENCH_DIR)/mantix_bench $(BENCH_ARGS)

benchmark-save:
	$(CMAKE) --fresh -S . -B $(BENCH_DIR) -G Ninja -DCMAKE_C_COMPILER=$(CLANG) -DCMAKE_BUILD_TYPE=Release -DMANTIX_BUILD_BENCHMARKS=ON -DMANTIX_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
	$(CMAKE) --build $(BENCH_DIR) --target mantix_bench
	./$(BENCH_DIR)/mantix_bench --csv --output $(BENCH_OUT) $(BENCH_ARGS)

clean:
	$(CMAKE) -E remove_directory $(BUILD_DIR)
	$(CMAKE) -E remove_directory $(BUILD_DIR)-sanitize
	$(CMAKE) -E remove_directory $(BENCH_DIR)
