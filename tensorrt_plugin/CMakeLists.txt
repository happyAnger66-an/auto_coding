cmake_minimum_required(VERSION 3.18)
project(fmha_plugin LANGUAGES CXX CUDA)

# Find required packages
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA properties
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD 14)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TENSORRT_INCLUDE_DIR})

# Link directories
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${TENSORRT_LIBRARY_DIR})

# Define the plugin library
add_library(fmha_plugin SHARED
    fmha_plugin.cpp
    fmha_kernel.cu
)

# Set CUDA architectures (adjust based on your target GPU)
set_target_properties(fmha_plugin PROPERTIES
    CUDA_ARCHITECTURES "75;80;86;89"
)

# Link libraries
target_link_libraries(fmha_plugin
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBRARIES}
    cudart
    cublas
    cudnn
)

# Compiler flags
target_compile_options(fmha_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -O3 -arch=sm_75>
)

# Set output directory
set_target_properties(fmha_plugin PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install rules
install(TARGETS fmha_plugin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)