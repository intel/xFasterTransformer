cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} MODEL_SRCS)

add_library(models OBJECT ${MODEL_SRCS})
target_include_directories(models PUBLIC ../common ../utils ../kernels ../layers ../searchers)
