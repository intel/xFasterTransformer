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

include(${CMAKE_SOURCE_DIR}/cmake/cmdline.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/sentencepiece.cmake)

set(EXAMPLE_SRCS "example.cpp" "vocab_opt.cpp" "vocab_qwen.cpp")
set(CB_CHECK_SRCS "cb_check.cpp" "vocab_opt.cpp" "vocab_qwen.cpp")

add_executable(example ${EXAMPLE_SRCS})
add_executable(cb_check ${CB_CHECK_SRCS})

target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/cmdline)
target_include_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece/include)
target_include_directories(cb_check PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/cmdline)
target_include_directories(cb_check PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece/include)

target_link_directories(example PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece/${CMAKE_INSTALL_LIBDIR})
target_link_directories(cb_check PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece/${CMAKE_INSTALL_LIBDIR})

if(BUILD_WITH_SHARED_LIBS)
    target_link_libraries(example PRIVATE xfastertransformer)
    target_link_libraries(cb_check PRIVATE xfastertransformer)
else()
    target_link_libraries(example PRIVATE xfastertransformer_static)
    target_link_libraries(cb_check PRIVATE xfastertransformer_static)
endif()
target_link_libraries(example PRIVATE sentencepiece -lstdc++fs)
target_link_libraries(cb_check PRIVATE sentencepiece -lstdc++fs)
if(WITH_GPU)
    target_link_libraries(example PRIVATE -fsycl -fsycl-device-code-split=per_kernel -lOpenCL)
    target_link_libraries(cb_check PRIVATE -fsycl -fsycl-device-code-split=per_kernel -lOpenCL)
endif()

add_dependencies(example cmdline sentencepiece_lib)
add_dependencies(cb_check cmdline sentencepiece_lib)
