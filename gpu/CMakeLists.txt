cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
enable_language(CUDA)
project(visibilityLatticesCuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -G -g")

# Disabling specific warnings that are not relevant to the project
# 68: integer conversion resulted in a change of sign
# 191: type qualifier is meaningless on cast type
# 368: class template "CLASS" defines no constructor to initialize the following
# 549: variable "VARIABLE" is used before its value is set
# 550: variable "VARIABLE" was set but never used
# 2361: invalid narrowing conversion from NUMBER to NUMBER
# 20012: __host__ annotation is ignored on a function(FUNCTION) that is explicitly defaulted on its first declaration
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 20012 --diag-suppress 2361 --diag-suppress 368 --diag-suppress 550 --diag-suppress 68 --diag-suppress 549 --diag-suppress 191")

# Add CUDA support
include(${CMAKE_SOURCE_DIR}/cmake/polyscope.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/dgtal.cmake)

# Define executable
add_executable(visibilityLatticesCuda main.cpp main_gpu.cu)
set_target_properties(visibilityLatticesCuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")

# Link libraries
target_link_libraries(visibilityLatticesCuda
    polyscope
    DGtal
)
