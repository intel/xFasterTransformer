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

cmake_minimum_required(VERSION 3.18)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

project(dependency NONE)

include(ExternalProject)

# cmake-format: off
ExternalProject_Add(oneccl
  GIT_REPOSITORY    https://github.com/oneapi-src/oneCCL.git
  GIT_TAG           2021.10
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/oneccl
  BINARY_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/oneccl
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory "build" && ${CMAKE_COMMAND} -E chdir "build" ${CMAKE_COMMAND} -DBUILD_EXAMPLES=OFF -DBUILD_FT=OFF -DBUILD_REG_TESTS=OFF -DENABLE_MPI_TESTS=OFF ..
  BUILD_COMMAND     ${CMAKE_COMMAND} -E chdir "build" make -j install
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
# cmake-format: on
