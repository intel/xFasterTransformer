cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} KERNELS_SRCS)

add_library(kernels OBJECT ${KERNELS_SRCS})
target_include_directories(kernels PUBLIC ../common ../utils)
