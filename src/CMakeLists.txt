# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
cmake_minimum_required(VERSION 3.15.1)

add_subdirectory(utils)
add_subdirectory(kernels)
add_subdirectory(layers)
add_subdirectory(models)
add_subdirectory(searchers)

add_subdirectory(comm_helper)

add_library(xfastertransformer_static STATIC)

set(SRC_LIB_LIST "utils" "layers" "kernels" "models" "searchers" "stdc++fs")

target_link_libraries(xfastertransformer_static
                      ${SRC_LIB_LIST}
                      ${3RDPART_LIB_LIST})

add_library(xfastertransformer SHARED)

target_link_libraries(xfastertransformer
                      ${SRC_LIB_LIST}
                      ${3RDPART_LIB_LIST})

add_dependencies(xfastertransformer_static ${DEPEND_LIST})
add_dependencies(xfastertransformer ${DEPEND_LIST})

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Using python: ${Python_EXECUTABLE}")

execute_process(COMMAND ${Python_EXECUTABLE} -m pip show torch
                RESULT_VARIABLE EXIT_CODE
                OUTPUT_QUIET)
if(${EXIT_CODE} EQUAL 0)
    message(STATUS "PyTorch found. Compiling torch extension...")
    add_subdirectory(pytorch)
else()
    message(STATUS "PyTorch not found. Skipping torch extension compliling...")
endif()

# Custom target to copy specific libraries and include folder to xft-devel
set(XFT_DEVEL_DIR "${CMAKE_BINARY_DIR}/xfastertransformer-devel")
file(MAKE_DIRECTORY "${XFT_DEVEL_DIR}/lib")
file(MAKE_DIRECTORY "${XFT_DEVEL_DIR}/include")

add_custom_target(copy_to_xft_devel ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libxfastertransformer.so ${XFT_DEVEL_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/src/libxfastertransformer_static.a ${XFT_DEVEL_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/libxft_comm_helper.so ${XFT_DEVEL_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${XFT_DEVEL_DIR}/include
    COMMENT "Copying specific libraries and include folder to xfastertransformer-devel"
)

add_dependencies(copy_to_xft_devel xfastertransformer xfastertransformer_static)
