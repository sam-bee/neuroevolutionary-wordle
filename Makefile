.PHONY: configure build test smoke clean build-and-test

-include .env

configure:
	cmake -S . -B build -G Ninja

build: --env
	cmake --build build

test:
	cd build && ctest --output-on-failure

smoke:
	cmake --build build --target input_encoder_device_smoke_test
	./build/input_encoder_device_smoke_test

clean:
	rm -rf build

build-and-test: --env clean configure build test smoke

--env:
	@test -f .env || (echo ".env is missing. Please copy .env.example" >&2; exit 1)
