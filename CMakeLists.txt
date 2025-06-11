cmake_minimum_required(VERSION 3.12)
project(VisibilityLattices)

# For OMP support use
# cmake -DCMAKE_PREFIX_PATH=$(brew --prefix libomp) -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" ..

include_directories("/opt/homebrew/opt/llvm/include" "/opt/homebrew/opt/libomp/include")
link_directories("/opt/homebrew/opt/llvm/lib" "/opt/homebrew/opt/libomp/lib")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(polyscope)
#include(eigen)

#set(BUILD_EXAMPLES OFF CACHE STRING "Build Examples")
# set(WITH_EIGEN ON CACHE STRING "DGtal - Eigen")
include(dgtal)
# include(geometry-central)
# include(spectra)

# add_subdirectory( CDT )

# If we are building for CPU, do the belows

# JOL: do not understand why I have to manually set thiis so that it finds Eigen
# include_directories( /Users/lachaud/GITHUB/DigitalShapeTools/src )
include_directories( /opt/homebrew/include/eigen3/)

set(CPU_BUILD ON CACHE BOOL "Build for CPU")

if (CPU_BUILD==ON)
    # Check if you have OMP support
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        add_executable(visibilityLattices main.cpp)
        target_link_libraries(visibilityLattices PUBLIC polyscope DGtal OpenMP::OpenMP_CXX)
    else ()
        # Crash the build
        message(FATAL_ERROR "OpenMP not found")
    endif()
else ()
    # Alert the user that they should use the GPU build in the gpu directory
    message(FATAL_ERROR "Building for GPU is not supported in this directory. Please use the gpu directory for GPU builds.")
endif()
