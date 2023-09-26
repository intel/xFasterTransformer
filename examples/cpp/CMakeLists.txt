# Copyright (C) 2023-2024 Intel Corporation
cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} EXAMPLE_SCR)

include(${CMAKE_SOURCE_DIR}/cmake/cmdline.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/sentencepiece.cmake)

add_executable(example ${EXAMPLE_SCR})

target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/cmdline)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece/include)

target_link_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece/${CMAKE_INSTALL_LIBDIR})

if(BUILD_WITH_SHARED_LIBS)
    target_link_libraries(example PRIVATE xfastertransformer)
else()
    target_link_libraries(example PRIVATE xfastertransformer_static)
endif()
target_link_libraries(example PRIVATE sentencepiece -lstdc++fs)

add_dependencies(example cmdline sentencepiece_lib)
