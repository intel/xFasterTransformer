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

execute_process(COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE)

string(REPLACE "share/cmake" "include" PyTorch_INCLUDE_DIR "${TORCH_CMAKE_PREFIX_PATH}")
string(REGEX REPLACE "\n$" "" PyTorch_INCLUDE_DIR "${PyTorch_INCLUDE_DIR}")

string(REPLACE "share/cmake" "lib" PyTorch_LIBRARIES_DIR "${TORCH_CMAKE_PREFIX_PATH}")
string(REGEX REPLACE "\n$" "" PyTorch_LIBRARIES_DIR "${PyTorch_LIBRARIES_DIR}")

find_library(TORCH_LIB torch HINTS ${PyTorch_LIBRARIES_DIR})
find_library(TORCH_GLOBAL_DEPS_LIB torch_global_deps HINTS ${PyTorch_LIBRARIES_DIR})
find_library(TORCH_CPU_LIB torch_cpu HINTS ${PyTorch_LIBRARIES_DIR})
find_library(TORCH_PYTHON_LIB torch_python HINTS ${PyTorch_LIBRARIES_DIR})
find_library(SHM_CPU_LIB shm HINTS ${PyTorch_LIBRARIES_DIR})
find_library(C10_CPU_LIB c10 HINTS ${PyTorch_LIBRARIES_DIR})
set(TORCH_LIBS ${TORCH_GLOBAL_DEPS_LIB} ${TORCH_CPU_LIB} ${TORCH_PYTHON_LIB} ${SHM_CPU_LIB}
               ${C10_CPU_LIB})

# Set the CMAKE_PREFIX_PATH
set(CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src/xfastertransformer)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} TORCH_SRCS)
add_library(xfastertransformer_pt SHARED ${TORCH_SRCS})

target_include_directories(xfastertransformer_pt PUBLIC ${PyTorch_INCLUDE_DIR})

# Link against LibTorch and others
target_link_libraries(xfastertransformer_pt
                      PRIVATE "${TORCH_LIBS}"
                              xfastertransformer_static
                              stdc++fs)

set(COMM_HELPER_SO_FILE "${PROJECT_BINARY_DIR}/libxft_comm_helper.so")

add_custom_command(TARGET xfastertransformer_pt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    ${COMM_HELPER_SO_FILE}
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copying libxft_comm_helper.so to src/xfastertransformer directory"
)

add_custom_command(TARGET xfastertransformer_pt POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_SOURCE_DIR}/3rdparty/mkl/lib/libiomp5.so"
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
    COMMENT "Copying libiomp5.so to src/xfastertransformer directory"
)