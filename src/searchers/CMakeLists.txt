cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SEARCHERS_SRCS)

add_library(searchers OBJECT ${SEARCHERS_SRCS})
target_include_directories(searchers PUBLIC ../common ../utils ../models)
