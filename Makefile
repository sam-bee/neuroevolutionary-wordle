.PHONY: configure build test test-cpu test-gpu test-gpu-sanitized smoke format clean rebuild agents-rebuild build-and-test run-ga-desktop run-ga-laptop run-ga-growth-smoke play-wordle

-include .env

export DOCKERCOMPOSE_UID
export DOCKERCOMPOSE_GID
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

AGENTS_CUDA_DEVICE_ORDER := $(if $(CUDA_DEVICE_ORDER),$(CUDA_DEVICE_ORDER),FASTEST_FIRST)
AGENTS_CUDA_VISIBLE_DEVICES := $(if $(CUDA_VISIBLE_DEVICES),$(CUDA_VISIBLE_DEVICES),0)

GA_GENERATIONS ?= 1000
GA_SEED ?=
GA_SEED_ARG := $(if $(GA_SEED),--seed $(GA_SEED))
PLAY_WORDLE_MODEL ?=
PLAY_WORDLE_METADATA ?= $(patsubst %.bin,%.json,$(PLAY_WORDLE_MODEL))

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

run-ga-desktop: GA_TARGET_GENOTYPE_VRAM_GB := 12
run-ga-laptop: GA_TARGET_GENOTYPE_VRAM_GB := 6

run-ga-desktop run-ga-laptop: .env
	cmake --build build --target run_genetic_algorithm
	./build/run_genetic_algorithm \
		--generations $(GA_GENERATIONS) \
		--genotype-vram-gb $(GA_TARGET_GENOTYPE_VRAM_GB) \
		--initial-word-count 20 \
		--word-count-step 20 \
		--word-count-step-period 25 \
		--shard-radius-growth-period 2 $(GA_SEED_ARG)

run-ga-growth-smoke: .env
	cmake --build build --target run_genetic_algorithm
	stdbuf -oL -eL ./build/run_genetic_algorithm \
		--generations 4 \
		--genotype-vram-gb 1 \
		--initial-word-count 20 \
		--word-count-step 20 \
		--word-count-step-period 2 \
		--shard-radius-growth-period 2

play-wordle: .env
	@if [ -z "$(PLAY_WORDLE_MODEL)" ]; then \
		echo "Set PLAY_WORDLE_MODEL=path/to/winner.bin"; \
		exit 1; \
	fi
	@if [ ! -f "$(PLAY_WORDLE_MODEL)" ]; then \
		echo "Model file not found: $(PLAY_WORDLE_MODEL)"; \
		exit 1; \
	fi
	@if [ ! -f "$(PLAY_WORDLE_METADATA)" ]; then \
		echo "Metadata file not found: $(PLAY_WORDLE_METADATA)"; \
		exit 1; \
	fi
	cmake --build build --target play_wordle
	./build/play_wordle "$(PLAY_WORDLE_MODEL)" "$(PLAY_WORDLE_METADATA)"

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
