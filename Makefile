.PHONY: configure build test smoke format clean build-and-test

-include .env

FORMAT_FILES := $(shell find src tests -type f \( -name '*.hpp' -o -name '*.cpp' -o -name '*.cu' \) | sort)

configure:
	cmake -S . -B build -G Ninja

build: --env
	cmake --build build

test:
	cd build && ctest --output-on-failure

smoke:
	cmake --build build --target input_encoder_device_smoke_test
	./build/input_encoder_device_smoke_test

format:
	clang-format -i $(FORMAT_FILES)

clean:
	rm -rf build

rebuild: --env clean format configure build test smoke

--env:
	@test -f .env || (echo ".env is missing. Please copy .env.example" >&2; exit 1)
