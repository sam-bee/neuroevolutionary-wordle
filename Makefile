.PHONY: configure build test test-cpu test-gpu test-gpu-sanitized smoke format clean rebuild agents-rebuild build-and-test run-ga

-include .env

export DOCKERCOMPOSE_UID
export DOCKERCOMPOSE_GID
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

AGENTS_CUDA_DEVICE_ORDER := $(if $(CUDA_DEVICE_ORDER),$(CUDA_DEVICE_ORDER),FASTEST_FIRST)
AGENTS_CUDA_VISIBLE_DEVICES := $(if $(CUDA_VISIBLE_DEVICES),$(CUDA_VISIBLE_DEVICES),0)

GA_GENERATIONS ?= 200
GA_GENOTYPE_VRAM_GB ?= 12
GA_INITIAL_WORD_COUNT ?= 20
GA_WORD_COUNT_STEP ?= 10
GA_WORD_COUNT_STEP_PERIOD ?= 5
GA_SHARD_RADIUS_GROWTH_PERIOD ?= 2
GA_SEED ?=
GA_SEED_ARG := $(if $(GA_SEED),--seed $(GA_SEED))

FORMAT_FILES := $(shell find src tests -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) | sort)

configure: .env
	cmake -S . -B build -G Ninja

build: .env
	cmake --build build

test: .env
	cd build && ctest --output-on-failure

test-cpu: .env
	cd build && ctest --output-on-failure -L cpu

test-gpu: .env
	cd build && ctest --output-on-failure -L gpu

test-gpu-sanitized: .env
	cmake --build build --target input_encoder_device_smoke_test
	compute-sanitizer --error-exitcode=1 ./build/input_encoder_device_smoke_test

smoke: test-gpu

run-ga: .env
	cmake --build build --target run_genetic_algorithm
	./build/run_genetic_algorithm \
		--generations $(GA_GENERATIONS) \
		--genotype-vram-gb $(GA_GENOTYPE_VRAM_GB) \
		--initial-word-count $(GA_INITIAL_WORD_COUNT) \
		--word-count-step $(GA_WORD_COUNT_STEP) \
		--word-count-step-period $(GA_WORD_COUNT_STEP_PERIOD) \
		--shard-radius-growth-period $(GA_SHARD_RADIUS_GROWTH_PERIOD) $(GA_SEED_ARG)

format:
	clang-format -i $(FORMAT_FILES)

clean:
	rm -rf build

rebuild: clean format configure build test

agents-rebuild: .env
	CUDA_DEVICE_ORDER=$(AGENTS_CUDA_DEVICE_ORDER) CUDA_VISIBLE_DEVICES=$(AGENTS_CUDA_VISIBLE_DEVICES) $(MAKE) rebuild

build-and-test: configure build test

.env:
	cp .env.example .env
	@echo "Created .env from .env.example"
