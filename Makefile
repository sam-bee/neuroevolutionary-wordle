.PHONY: configure build test smoke clean

CUDA_DEVICE_ORDER ?= PCI_BUS_ID
CUDA_VISIBLE_DEVICES ?= 1
CUDA_ENV := CUDA_DEVICE_ORDER=$(CUDA_DEVICE_ORDER) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)

configure:
	cmake -S . -B build -G Ninja

build:
	cmake --build build

test:
	ctest --test-dir build --output-on-failure

smoke:
	cmake --build build --target input_encoder_device_smoke_test
	$(CUDA_ENV) ./build/input_encoder_device_smoke_test

clean:
	rm -rf build
