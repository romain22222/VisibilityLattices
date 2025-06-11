cmake_minimum_required(VERSION 3.18)
project(visibilityLatticesCuda LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add CUDA support
enable_language(CUDA)

include(${CMAKE_SOURCE_DIR}/cmake/polyscope.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/dgtal.cmake)

# Define executable
add_executable(visibilityLatticesCuda main_gpu.cu)

# Link libraries
target_link_libraries(visibilityLatticesCuda
    polyscope
    DGtal
)
