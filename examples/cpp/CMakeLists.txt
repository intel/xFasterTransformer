# Copyright (C) 2023-2024 Intel Corporation
cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} EXAMPLE_SCR)

add_executable(example ${EXAMPLE_SCR})
if(BUILD_WITH_SHARED_LIBS)
    target_link_libraries(example PRIVATE xfastertransformer)
else()
    target_link_libraries(example PRIVATE xfastertransformer_static)
endif()
