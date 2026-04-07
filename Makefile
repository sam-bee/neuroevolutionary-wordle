.PHONY: configure build test smoke format clean rebuild build-and-test

-include .env

export DOCKERCOMPOSE_UID
export DOCKERCOMPOSE_GID
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

FORMAT_FILES := $(shell find src tests -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) | sort)

configure: .env
	cmake -S . -B build -G Ninja

build: .env
	cmake --build build

test: .env
	cd build && ctest --output-on-failure

smoke: .env
	cmake --build build --target input_encoder_device_smoke_test
	./build/input_encoder_device_smoke_test

format:
	clang-format -i $(FORMAT_FILES)

clean:
	rm -rf build

rebuild: clean format configure build test smoke

build-and-test: configure build test

.env:
	cp .env.example .env
	@echo "Created .env from .env.example"
