cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} LAYERS_SRCS)

add_library(layers OBJECT ${LAYERS_SRCS})
add_dependencies(layers utils)
