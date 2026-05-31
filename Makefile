.PHONY: configure build test test-cpu test-gpu test-gpu-sanitized smoke format clean rebuild agents-rebuild build-and-test run-ga-desktop run-ga-laptop run-ga-growth-smoke run-ga-benchmark-growth run-ga-benchmark-two-gen play-wordle telemetry-up telemetry-stop

-include .env

export DOCKERCOMPOSE_UID
export DOCKERCOMPOSE_GID
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

AGENTS_CUDA_DEVICE_ORDER := $(if $(CUDA_DEVICE_ORDER),$(CUDA_DEVICE_ORDER),FASTEST_FIRST)
AGENTS_CUDA_VISIBLE_DEVICES := $(if $(CUDA_VISIBLE_DEVICES),$(CUDA_VISIBLE_DEVICES),0)

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

run-ga-desktop: .env
	cmake --build build --target run_genetic_algorithm
	./build/run_genetic_algorithm \
		--generations 1000 \
		--genotype-vram-gb 12 \
		--initial-word-count 20 \
		--word-count-step 20 \
		--shard-release-min-gap 10 \
		--first-new-shard-release-generation 10 \
		--shard-release-fitness-p99-threshold 0.05 \
		--shard-radius-growth-period 2 \
		--verbose

run-ga-laptop: .env
	cmake --build build --target run_genetic_algorithm
	./build/run_genetic_algorithm \
		--generations 1000 \
		--genotype-vram-gb 6 \
		--initial-word-count 20 \
		--word-count-step 20 \
		--shard-release-min-gap 10 \
		--first-new-shard-release-generation 10 \
		--shard-release-fitness-p99-threshold 0.05 \
		--shard-radius-growth-period 2 \
		--verbose

run-ga-prod: .env
	cmake --build build --target run_genetic_algorithm
	./build/run_genetic_algorithm \
		--generations 3000 \
		--genotype-vram-gb 12 \
		--generation-vram-gb 8 \
		--initial-word-count 50 \
		--word-count-step 50 \
		--shard-release-min-gap 10 \
		--first-new-shard-release-generation 10 \
		--shard-release-centroid-threshold 4 \
		--shard-release-fitness-p99-threshold 0.05 \
		--shard-radius-growth-period 2 \
		--checkpoint-path checkpoints/ga-runtime.bin \
		--checkpoint-every 10 \
		--breeding-radius 3 \
		--parent-selection-rank-exponent 0.5 \
		--crossover-temperature-level1 0.02 \
		--crossover-temperature-level2 0.01 \
		--crossover-temperature-level3 0.005 \
		--telemetry-dir telemetry/runs \
		--telemetry-genetic-convergence \
		--verbose

run-ga-growth-smoke: .env
	cmake --build build --target run_genetic_algorithm
	stdbuf -oL -eL ./build/run_genetic_algorithm \
		--generations 11 \
		--genotype-vram-gb 1 \
		--initial-word-count 20 \
		--word-count-step 20 \
		--shard-release-min-gap 10 \
		--first-new-shard-release-generation 10 \
		--shard-release-centroid-threshold 1000000 \
		--shard-release-fitness-p99-threshold 0.05 \
		--shard-radius-growth-period 2 \
		--verbose

run-ga-benchmark-two-gen: run-ga-benchmark-growth

run-ga-benchmark-growth: .env
	cmake --build build --target run_genetic_algorithm
	stdbuf -oL -eL ./build/run_genetic_algorithm \
		--generations 11 \
		--population-size 1024 \
		--genotype-vram-gb 1 \
		--generation-vram-gb 0.5 \
		--initial-word-count 20 \
		--word-count-step 1980 \
		--shard-release-min-gap 10 \
		--first-new-shard-release-generation 10 \
		--shard-release-centroid-threshold 1000000 \
		--shard-release-fitness-p99-threshold 0.05 \
		--shard-initial-radius-infinite \
		--verbose

telemetry-up:
	docker compose up --build -d telemetry-web

telemetry-stop:
	docker compose stop telemetry-web

play-wordle: .env
	@if [ ! -f "models/winner-2026-05-21_01-56-52-g0-seed7.bin" ]; then \
		echo "Model file not found: models/winner-2026-05-21_01-56-52-g0-seed7.bin"; \
		exit 1; \
	fi
	@if [ ! -f "models/winner-2026-05-21_01-56-52-g0-seed7.json" ]; then \
		echo "Metadata file not found: models/winner-2026-05-21_01-56-52-g0-seed7.json"; \
		exit 1; \
	fi
	cmake --build build --target play_wordle
	./build/play_wordle "models/winner-2026-05-21_01-56-52-g0-seed7.bin" "models/winner-2026-05-21_01-56-52-g0-seed7.json"

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
