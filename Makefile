.PHONY: configure build test test-cpu test-gpu test-gpu-sanitized smoke format clean rebuild sandbox-rebuild build-and-test

-include .env

export DOCKERCOMPOSE_UID
export DOCKERCOMPOSE_GID
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

SANDBOX_CUDA_DEVICE_ORDER := $(if $(CUDA_DEVICE_ORDER),$(CUDA_DEVICE_ORDER),FASTEST_FIRST)
SANDBOX_CUDA_VISIBLE_DEVICES := $(if $(CUDA_VISIBLE_DEVICES),$(CUDA_VISIBLE_DEVICES),0)

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

format:
	clang-format -i $(FORMAT_FILES)

clean:
	rm -rf build

rebuild: clean format configure build test

sandbox-rebuild: .env
	CUDA_DEVICE_ORDER=$(SANDBOX_CUDA_DEVICE_ORDER) CUDA_VISIBLE_DEVICES=$(SANDBOX_CUDA_VISIBLE_DEVICES) $(MAKE) rebuild

build-and-test: configure build test

.env:
	cp .env.example .env
	@echo "Created .env from .env.example"
