# Copyright (C) 2023-2024 Intel Corporation
cmake_minimum_required(VERSION 3.15.1)
project(xfastertransformer LANGUAGES C CXX)

find_package(MPI REQUIRED)
find_package(oneCCL REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fopenmp -mavx512f -mavx512bw -mavx512vl -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message("Notice: Using Debug mode.")
    set(CMAKE_C_FLAGS "-O0 -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
else()
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

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/mklml/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onednn/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/onednn/build/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

link_directories(${CMAKE_SOURCE_DIR}/src/kernels)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/mklml/lib)
link_directories(${CMAKE_SOURCE_DIR}/3rdparty/onednn/build/src)

option(BUILD_WITH_SHARED_LIBS "Build with shared libraries" OFF)
if(BUILD_WITH_SHARED_LIBS)
    message("Building with shared libraries.")
    file(GLOB GEMM_KERNEL_FILES ${CMAKE_SOURCE_DIR}/src/kernels/*.so)
else()
    message("Building with static libraries.")
    file(GLOB GEMM_KERNEL_FILES ${CMAKE_SOURCE_DIR}/src/kernels/*.a)
endif()

set(GEMM_KERNEL_LIST)
foreach(GEMM_KERNEL_FILE ${GEMM_KERNEL_FILES})
    get_filename_component(FILE_NAME ${GEMM_KERNEL_FILE} NAME)
    message("Found gemm kernel file: ${FILE_NAME}")
    list(APPEND GEMM_KERNEL_LIST ${FILE_NAME})
endforeach()

set(3RDPART_LIB_LIST "MPI::MPI_CXX" "ccl" "dnnl")
set(DEPEND_LIST "onednn" "mklml")

# Enable AVX512_FP16 optimization
# add_definitions(-DAVX512_FP32_WEIGHT_ONLY_FP16=true)
add_definitions(-DAVX512_FP16_WEIGHT_ONLY_FP16=true)
add_definitions(-DAVX512_BF16_WEIGHT_ONLY_BF16=true)
# add_definitions(-DAVX512_FP32_WEIGHT_ONLY_INT8=true)
add_definitions(-DAVX512_FP16_WEIGHT_ONLY_INT8=true)
# add_definitions(-DDEBUG=true)
# add_definitions(-DSTEP_BY_STEP_ATTN=true)
# add_definitions(-DUSE_MKLML=true)
# add_definitions(-DTIMELINE=true)
add_definitions(-DUSE_SHM=true)

get_directory_property(MYDEFS COMPILE_DEFINITIONS)

if(MYDEFS MATCHES "^TIMELINE=" OR MYDEFS MATCHES ";TIMELINE=")
    include("cmake/jsoncpp.cmake")
    include_directories(${CMAKE_SOURCE_DIR}/3rdparty/jsoncpp/include)
    link_directories(${CMAKE_SOURCE_DIR}/3rdparty/jsoncpp/lib64)
    list(APPEND 3RDPART_LIB_LIST "jsoncpp")
    list(APPEND DEPEND_LIST "jsoncpp_lib")
endif()

# .so file
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
# exe file
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_subdirectory(src)
add_subdirectory(examples/cpp)
