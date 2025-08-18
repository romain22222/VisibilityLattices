cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)

option(USE_CUDA_VISIBILITY "Enable CUDA visibility kernel" OFF)

set(SOURCES
        main.cpp
        gpu/Vec3i.cu
)

# If you want to use CUDA, uncomment the following lines
if (USE_CUDA_VISIBILITY)
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.8/bin/nvcc")
    enable_language(CUDA)
    project(VisibilityLattices LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -G -g")
    list(APPEND SOURCES ./gpu/main_gpu.cu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 20012 --diag-suppress 2361 --diag-suppress 368 --diag-suppress 550 --diag-suppress 68 --diag-suppress 549 --diag-suppress 191")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    add_definitions(-DUSE_CUDA_VISIBILITY)
else()
    project(VisibilityLattices)
endif ()

# For OMP support use
# cmake -DCMAKE_PREFIX_PATH=$(brew --prefix libomp) -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" ..

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

#set(BUILD_EXAMPLES OFF CACHE STRING "Build Examples")
# set(WITH_EIGEN ON CACHE STRING "DGtal - Eigen")
# include(dgtal)

FIND_PACKAGE(DGtal REQUIRED)
INCLUDE_DIRECTORIES(${DGTAL_INCLUDE_DIRS})
LINK_DIRECTORIES(${DGTAL_LIBRARY_DIRS})

# include(geometry-central)
# include(spectra)

# add_subdirectory( CDT )

# If we are building for CPU, do the belows

# JOL: do not understand why I have to manually set thiis so that it finds Eigen
# include_directories( /Users/lachaud/GITHUB/DigitalShapeTools/src )
#include_directories(/opt/homebrew/include/eigen3/)

# Check if you have OMP support
find_package(OpenMP)

if (OpenMP_CXX_FOUND)
    add_executable(visibilityLattices ${SOURCES})
    target_link_libraries(visibilityLattices PUBLIC polyscope ${DGTAL_LIBRARIES} OpenMP::OpenMP_CXX)
    if (CUSE_CUDA_VISIBILITY)
    endif ()
else ()
    # Crash the build
    message(FATAL_ERROR "OpenMP not found")
endif ()
