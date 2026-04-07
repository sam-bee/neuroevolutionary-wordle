cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_ARCHITECTURES 120)

project(neuroevolution_wordle LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_testing()

function(enable_project_warnings target_name)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>
    )
endfunction()

add_library(input_encoder STATIC
    src/model/input_encoder/shared_encoder.cu
    src/model/input_encoder/turn_features.cpp
)

target_include_directories(input_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

enable_project_warnings(input_encoder)

add_library(model_input STATIC
    src/model/model_input/wordle_grid_state.cpp
)

target_include_directories(model_input PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(model_input PUBLIC
    input_encoder
)

enable_project_warnings(model_input)

add_executable(turn_features_test
    tests/model/input_encoder/turn_features_test.cpp
)

target_link_libraries(turn_features_test PRIVATE
    input_encoder
)

enable_project_warnings(turn_features_test)

add_test(NAME turn_features_test COMMAND turn_features_test)
set_tests_properties(turn_features_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(shared_encoder_test
    tests/model/input_encoder/shared_encoder_test.cpp
)

target_link_libraries(shared_encoder_test PRIVATE
    input_encoder
)

enable_project_warnings(shared_encoder_test)

add_test(NAME shared_encoder_test COMMAND shared_encoder_test)
set_tests_properties(shared_encoder_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(wordle_grid_state_test
    tests/model/input_encoder/wordle_grid_state_test.cpp
)

target_link_libraries(wordle_grid_state_test PRIVATE
    model_input
)

enable_project_warnings(wordle_grid_state_test)

add_test(NAME wordle_grid_state_test COMMAND wordle_grid_state_test)
set_tests_properties(wordle_grid_state_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(wordle_grid_test
    tests/wordle/wordle_grid_test.cpp
)

target_include_directories(wordle_grid_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

enable_project_warnings(wordle_grid_test)

add_test(NAME wordle_grid_test COMMAND wordle_grid_test)
set_tests_properties(wordle_grid_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(input_encoder_device_smoke_test
    tests/model/input_encoder/input_encoder_device_smoke_test.cu
)

target_link_libraries(input_encoder_device_smoke_test PRIVATE
    model_input
)

enable_project_warnings(input_encoder_device_smoke_test)

add_test(NAME input_encoder_device_smoke_test COMMAND input_encoder_device_smoke_test)
set_tests_properties(input_encoder_device_smoke_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_executable(wordle_grid_device_e2e_test
    tests/model/input_encoder/wordle_grid_device_e2e_test.cu
)

target_link_libraries(wordle_grid_device_e2e_test PRIVATE
    model_input
)

enable_project_warnings(wordle_grid_device_e2e_test)

add_test(NAME wordle_grid_device_e2e_test COMMAND wordle_grid_device_e2e_test)
set_tests_properties(wordle_grid_device_e2e_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)
