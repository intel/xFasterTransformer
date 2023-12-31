# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cmake_minimum_required(VERSION 3.15.1)
project(xfastertransformer LANGUAGES C CXX)

# Get gcc version
execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpfullversion
                OUTPUT_VARIABLE GCC_VERSION
                OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Notice: GCC version: ${GCC_VERSION}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fopenmp -mavx512f -mavx512bw -mavx512vl -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

if(GCC_VERSION VERSION_GREATER_EQUAL "12.3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512bf16")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Notice: Using Debug mode.")
    set(CMAKE_C_FLAGS "-O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
    message(STATUS "Notice: Using Release mode.")
    set(CMAKE_C_FLAGS "-O2")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
endif()

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR
                "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles."
    )
endif()

include("cmake/mklml.cmake")
include("cmake/onednn.cmake")
include("cmake/xdnn.cmake")
include("cmake/mkl.cmake")
include(GNUInstallDirs)

set(DEPEND_LIST "onednn" "xdnn_lib")

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onednn/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onednn/build/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/xdnn)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mkl/include)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/kernels)
include_directories(${CMAKE_SOURCE_DIR}/src/layers)
include_directories(${CMAKE_SOURCE_DIR}/src/searchers)
include_directories(${CMAKE_SOURCE_DIR}/src/utils)
include_directories(${CMAKE_SOURCE_DIR}/src/models)
include_directories(${CMAKE_SOURCE_DIR}/src/common)

link_directories(${CMAKE_SOURCE_DIR}/src/kernels)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/onednn/build/src)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/xdnn)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/mkl/lib)

find_package(oneCCL REQUIRED)

if(MPI_FOUND AND oneCCL_FOUND)
    set(MPI_LIBS MPI::MPI_CXX)

    message(STATUS "oneCCL & MPI found. Using found oneCCL instead of building from source.")
else()
    message(STATUS "oneCCL or MPI not found. Build oneCCL from source.")

    include("cmake/oneccl.cmake")
    list(APPEND DEPEND_LIST "oneccl")
    set(MPI_LIBS "mpicxx" "mpi" "rt" "dl")

    include_directories(${CMAKE_SOURCE_DIR}/3rdparty/oneccl/build/_install/include)
    link_directories(${CMAKE_SOURCE_DIR}/3rdparty/oneccl/build/_install/lib)
    link_directories(${CMAKE_SOURCE_DIR}/3rdparty/oneccl/build/_install/lib/prov)
endif()

set(3RDPART_LIB_LIST "rt" "dl" "dnnl" "numa")

# Enable MKL library
list(APPEND 3RDPART_LIB_LIST "mkl_intel_ilp64" "mkl_gnu_thread" "mkl_core" "mkl_gnu_thread")

option(BUILD_WITH_SHARED_LIBS "Build with shared libraries" OFF)
if(BUILD_WITH_SHARED_LIBS)
    message(STATUS "Notice: Building with shared libraries.")
    list(APPEND 3RDPART_LIB_LIST "xdnn")
else()
    message(STATUS "Notice: Building with static libraries.")
    list(APPEND 3RDPART_LIB_LIST "xdnn_static")
endif()

# Enable AVX512_FP16 optimization
# add_definitions(-DAVX512_FP32_WEIGHT_ONLY_FP16=true)
add_definitions(-DAVX512_FP16_WEIGHT_ONLY_FP16=true)
add_definitions(-DAVX512_BF16_WEIGHT_ONLY_BF16=true)
# add_definitions(-DAVX512_FP32_WEIGHT_ONLY_INT8=true)
add_definitions(-DAVX512_FP16_WEIGHT_ONLY_INT8=true)
# add_definitions(-DAVX512_FP32_WEIGHT_ONLY_INT4=true)
add_definitions(-DAVX512_FP16_WEIGHT_ONLY_INT4=true)
add_definitions(-DAVX512_FP32_WEIGHT_ONLY_NF4=true)
# add_definitions(-DAVX512_FP16_WEIGHT_ONLY_NF4=true)

# add_definitions(-DDEBUG=true)
# add_definitions(-DSTEP_BY_STEP_ATTN=true)
add_definitions(-DUSE_SHM=true)
option(XFT_BUILD_TESTS "Build xfastertransformer unit tests" OFF)

# timeline event
option(WITH_TIMELINE "Build with timeline event support" OFF)
if(WITH_TIMELINE)
    message(STATUS "Notice: Building with timeline event support.")
    add_definitions(-DTIMELINE=true)
    include("cmake/jsoncpp.cmake")
    include_directories(${CMAKE_SOURCE_DIR}/3rdparty/jsoncpp/include)
    link_directories(${CMAKE_SOURCE_DIR}/3rdparty/jsoncpp/${CMAKE_INSTALL_LIBDIR})
    list(APPEND 3RDPART_LIB_LIST "jsoncpp")
    list(APPEND DEPEND_LIST "jsoncpp_lib")
endif()

# .so file
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# exe file
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(src)
add_subdirectory(examples/cpp)

add_subdirectory(tests)

option(XFT_BUILD_EVALUATION "Build xfastertransformer evalution tests" OFF)
if(XFT_BUILD_EVALUATION)
add_subdirectory(evaluation)
endif()
