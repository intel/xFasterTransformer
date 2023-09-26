cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} MODEL_SRCS)

add_library(models OBJECT ${MODEL_SRCS})
add_dependencies(models utils)
