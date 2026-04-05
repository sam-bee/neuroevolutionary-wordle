.PHONY: configure build test smoke clean build-and-test

ifeq ($(wildcard .env),)
ifneq ($(wildcard .env.example),)
$(shell cp .env.example .env)
endif
endif

-include .env
export CUDA_DEVICE_ORDER
export CUDA_VISIBLE_DEVICES

CUDA_DEVICE_ORDER ?= PCI_BUS_ID
CUDA_VISIBLE_DEVICES ?= 0
CUDA_ENV := CUDA_DEVICE_ORDER=$(CUDA_DEVICE_ORDER) CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES)

configure:
	cmake -S . -B build -G Ninja

build:
	cmake --build build

test:
	cd build && ctest --output-on-failure

smoke:
	cmake --build build --target input_encoder_device_smoke_test
	$(CUDA_ENV) ./build/input_encoder_device_smoke_test

clean:
	rm -rf build

build-and-test: clean configure build test smoke
