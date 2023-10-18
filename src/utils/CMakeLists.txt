cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} UTILS_SRCS)

add_library(utils OBJECT ${UTILS_SRCS})

add_dependencies(utils ${DEPEND_LIST})

add_subdirectory(gptq)