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

add_library(dense_trunk STATIC
    src/model/dense_trunk/dense_trunk.cpp
)

target_include_directories(dense_trunk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(dense_trunk PUBLIC
    model_input
)

enable_project_warnings(dense_trunk)

add_library(policy_model STATIC
    src/model/policy_model/policy_model.cpp
)

target_include_directories(policy_model PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(policy_model PUBLIC
    dense_trunk
)

enable_project_warnings(policy_model)

add_library(output_embedding STATIC
    src/model/output_embedding/output_embedding.cpp
)

target_include_directories(output_embedding PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(output_embedding PUBLIC
    dense_trunk
)

enable_project_warnings(output_embedding)

add_library(model_initialization STATIC
    src/model/initialization/parameter_initialization.cpp
)

target_include_directories(model_initialization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(model_initialization PUBLIC
    output_embedding
    policy_model
)

enable_project_warnings(model_initialization)

add_library(training_folder STATIC
    src/training_folder/training_data.cpp
    src/training_folder/training_data.cu
)

target_include_directories(training_folder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_compile_definitions(training_folder PUBLIC
    NEUROEVOLUTION_PROJECT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

enable_project_warnings(training_folder)

set_target_properties(training_folder PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(genetic_algorithm INTERFACE)

target_include_directories(genetic_algorithm INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(genetic_algorithm INTERFACE
    model_initialization
    output_embedding
    policy_model
)

add_library(genotype_slab STATIC
    src/genetic_algorithm/genotype_slab/assembly.cpp
    src/genetic_algorithm/genotype_slab/slab_allocator.cpp
)

target_include_directories(genotype_slab PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(genotype_slab PUBLIC
    genetic_algorithm
)

enable_project_warnings(genotype_slab)

add_library(genotype_slab_device STATIC
    src/genetic_algorithm/genotype_slab/device_runtime.cu
)

target_include_directories(genotype_slab_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(genotype_slab_device PUBLIC
    genotype_slab
    training_folder
)

enable_project_warnings(genotype_slab_device)

set_target_properties(genotype_slab_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_library(genetic_algorithm_slab_device STATIC
    src/genetic_algorithm/device/slab_runtime.cu
)

target_include_directories(genetic_algorithm_slab_device PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(genetic_algorithm_slab_device PUBLIC
    genotype_slab_device
    training_folder
)

enable_project_warnings(genetic_algorithm_slab_device)

set_target_properties(genetic_algorithm_slab_device PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(run_genetic_algorithm
    src/cli/run_genetic_algorithm.cu
)

target_link_libraries(run_genetic_algorithm PRIVATE
    genetic_algorithm_slab_device
)

enable_project_warnings(run_genetic_algorithm)

set_target_properties(run_genetic_algorithm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

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

add_executable(hint_grid_test
    tests/wordle/hint_grid_test.cpp
)

target_link_libraries(hint_grid_test PRIVATE
    model_input
)

enable_project_warnings(hint_grid_test)

add_test(NAME hint_grid_test COMMAND hint_grid_test)
set_tests_properties(hint_grid_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(dense_trunk_test
    tests/model/dense_trunk/dense_trunk_test.cpp
)

target_link_libraries(dense_trunk_test PRIVATE
    dense_trunk
)

enable_project_warnings(dense_trunk_test)

add_test(NAME dense_trunk_test COMMAND dense_trunk_test)
set_tests_properties(dense_trunk_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(policy_model_test
    tests/model/policy_model/policy_model_test.cpp
)

target_link_libraries(policy_model_test PRIVATE
    policy_model
)

enable_project_warnings(policy_model_test)

add_test(NAME policy_model_test COMMAND policy_model_test)
set_tests_properties(policy_model_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(output_embedding_test
    tests/model/output_embedding/output_embedding_test.cpp
)

target_link_libraries(output_embedding_test PRIVATE
    output_embedding
)

enable_project_warnings(output_embedding_test)

add_test(NAME output_embedding_test COMMAND output_embedding_test)
set_tests_properties(output_embedding_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(parameter_initialization_test
    tests/model/initialization/parameter_initialization_test.cpp
)

target_link_libraries(parameter_initialization_test PRIVATE
    model_initialization
)

enable_project_warnings(parameter_initialization_test)

add_test(NAME parameter_initialization_test COMMAND parameter_initialization_test)
set_tests_properties(parameter_initialization_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(genetic_algorithm_test
    tests/genetic_algorithm/genetic_algorithm_test.cpp
)

target_link_libraries(genetic_algorithm_test PRIVATE
    genetic_algorithm
)

enable_project_warnings(genetic_algorithm_test)

add_test(NAME genetic_algorithm_test COMMAND genetic_algorithm_test)
set_tests_properties(genetic_algorithm_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(output_embedding_genome_runtime_test
    tests/genetic_algorithm/output_embedding_genome_runtime_test.cpp
)

target_link_libraries(output_embedding_genome_runtime_test PRIVATE
    genetic_algorithm
)

enable_project_warnings(output_embedding_genome_runtime_test)

add_test(NAME output_embedding_genome_runtime_test COMMAND output_embedding_genome_runtime_test)
set_tests_properties(output_embedding_genome_runtime_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(genotype_slab_test
    tests/genetic_algorithm/genotype_slab_test.cpp
)

target_link_libraries(genotype_slab_test PRIVATE
    genotype_slab
)

enable_project_warnings(genotype_slab_test)

add_test(NAME genotype_slab_test COMMAND genotype_slab_test)
set_tests_properties(genotype_slab_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(spatial_grid_test
    tests/genetic_algorithm/spatial_grid_test.cpp
)

target_link_libraries(spatial_grid_test PRIVATE
    genetic_algorithm
)

enable_project_warnings(spatial_grid_test)

add_test(NAME spatial_grid_test COMMAND spatial_grid_test)
set_tests_properties(spatial_grid_test PROPERTIES
    LABELS "cpu;unit"
)

add_executable(training_data_test
    tests/training_folder/training_data_test.cpp
)

target_link_libraries(training_data_test PRIVATE
    training_folder
)

enable_project_warnings(training_data_test)

add_test(NAME training_data_test COMMAND training_data_test)
set_tests_properties(training_data_test PROPERTIES
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
    policy_model
)

enable_project_warnings(wordle_grid_device_e2e_test)

add_test(NAME wordle_grid_device_e2e_test COMMAND wordle_grid_device_e2e_test)
set_tests_properties(wordle_grid_device_e2e_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_executable(output_embedding_device_e2e_test
    tests/model/output_embedding/output_embedding_device_e2e_test.cu
)

target_link_libraries(output_embedding_device_e2e_test PRIVATE
    output_embedding
    policy_model
)

enable_project_warnings(output_embedding_device_e2e_test)

add_test(NAME output_embedding_device_e2e_test COMMAND output_embedding_device_e2e_test)
set_tests_properties(output_embedding_device_e2e_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_executable(training_data_device_test
    tests/training_folder/training_data_device_test.cu
)

target_link_libraries(training_data_device_test PRIVATE
    training_folder
)

enable_project_warnings(training_data_device_test)

set_target_properties(training_data_device_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME training_data_device_test COMMAND training_data_device_test)
set_tests_properties(training_data_device_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_executable(genotype_slab_device_runtime_test
    tests/genetic_algorithm/device/genotype_slab_device_runtime_test.cu
)

target_link_libraries(genotype_slab_device_runtime_test PRIVATE
    genotype_slab_device
)

enable_project_warnings(genotype_slab_device_runtime_test)

set_target_properties(genotype_slab_device_runtime_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME genotype_slab_device_runtime_test COMMAND genotype_slab_device_runtime_test)
set_tests_properties(genotype_slab_device_runtime_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_executable(slab_runtime_test
    tests/genetic_algorithm/device/slab_runtime_test.cu
)

target_link_libraries(slab_runtime_test PRIVATE
    genetic_algorithm_slab_device
    training_folder
)

enable_project_warnings(slab_runtime_test)

set_target_properties(slab_runtime_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME slab_runtime_test COMMAND slab_runtime_test)
set_tests_properties(slab_runtime_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_executable(output_embedding_injection_test
    tests/genetic_algorithm/output_embedding_injection_test.cu
)

target_link_libraries(output_embedding_injection_test PRIVATE
    genetic_algorithm
)

enable_project_warnings(output_embedding_injection_test)

set_target_properties(output_embedding_injection_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_test(NAME output_embedding_injection_test COMMAND output_embedding_injection_test)
set_tests_properties(output_embedding_injection_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)

add_test(NAME run_genetic_algorithm_smoke_test
    COMMAND run_genetic_algorithm --seed 7 --generations 2 --population-size 6 --genotype-vram-gb 0.01
)
set_tests_properties(run_genetic_algorithm_smoke_test PROPERTIES
    LABELS "gpu;integration"
    RESOURCE_LOCK cuda_device
)
