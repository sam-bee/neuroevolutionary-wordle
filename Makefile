.PHONY: configure build test clean

configure:
	cmake -S . -B build -G Ninja

build:
	cmake --build build

test:
	ctest --test-dir build --output-on-failure

clean:
	rm -rf build
