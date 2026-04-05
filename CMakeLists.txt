cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES 120)

project(neuroevolution_wordle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

add_library(input_encoder STATIC
    src/model/input_encoder/shared_encoder.cu
    src/model/input_encoder/turn_features.cpp
)

set_property(TARGET input_encoder PROPERTY CUDA_ARCHITECTURES 120)

target_include_directories(input_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(input_encoder PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)

add_executable(turn_features_test
    tests/model/input_encoder/turn_features_test.cpp
    src/model/input_encoder/turn_features.cpp
)

target_include_directories(turn_features_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(turn_features_test PRIVATE
    -Wall -Wextra -Wpedantic
)

add_test(NAME turn_features_test COMMAND turn_features_test)

add_executable(shared_encoder_test
    tests/model/input_encoder/shared_encoder_test.cpp
)

target_include_directories(shared_encoder_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(shared_encoder_test PRIVATE
    input_encoder
)

target_compile_options(shared_encoder_test PRIVATE
    -Wall -Wextra -Wpedantic
)

add_test(NAME shared_encoder_test COMMAND shared_encoder_test)

add_executable(input_encoder_device_smoke_test EXCLUDE_FROM_ALL
    tests/model/input_encoder/input_encoder_device_smoke_test.cu
)

set_property(TARGET input_encoder_device_smoke_test PROPERTY CUDA_ARCHITECTURES 120)

target_include_directories(input_encoder_device_smoke_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(input_encoder_device_smoke_test PRIVATE
    input_encoder
)

target_compile_options(input_encoder_device_smoke_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
)
