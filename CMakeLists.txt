cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES 120)

project(neuroevolution_wordle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(input_encoder STATIC
    src/model/input_encoder/shared_encoder.cu
    src/model/input_encoder/turn_features.cpp
)

target_include_directories(input_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(input_encoder PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
