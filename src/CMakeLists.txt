# Copyright (C) 2023-2024 Intel Corporation
cmake_minimum_required(VERSION 3.15.1)

add_subdirectory(utils)
add_subdirectory(utils/gptq)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(searchers)

add_library(xfastertransformer_static STATIC)

set(SRC_LIB_LIST "utils" "layers" "kernels" "models" "searchers")

target_link_libraries(xfastertransformer_static
                      ${SRC_LIB_LIST}
                      ${3RDPART_LIB_LIST}
                      libiomp5.so
                      libmklml_intel.so)

add_library(xfastertransformer SHARED)

target_link_libraries(xfastertransformer
                      ${SRC_LIB_LIST}
                      ${3RDPART_LIB_LIST}
                      libiomp5.so
                      libmklml_intel.so)

add_dependencies(xfastertransformer_static ${DEPEND_LIST})
add_dependencies(xfastertransformer ${DEPEND_LIST})

execute_process(COMMAND python -m pip show torch
                RESULT_VARIABLE EXIT_CODE
                OUTPUT_QUIET)
if(${EXIT_CODE} EQUAL 0)
    message(STATUS "PyTorch found. Compiling torch extension...")
    add_subdirectory(pytorch)
else()
    message(STATUS "PyTorch not found. Skipping torch extension compliling...")
endif()
