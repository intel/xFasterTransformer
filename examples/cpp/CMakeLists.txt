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
