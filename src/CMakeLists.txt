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
