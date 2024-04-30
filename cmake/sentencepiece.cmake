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

set(SP_BUILD_OPTIONS -DSPM_ENABLE_SHARED=OFF)
if(WITH_GPU)
    set(SP_BUILD_OPTIONS "${SP_BUILD_OPTIONS};-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1")
else()
    set(SP_BUILD_OPTIONS "${SP_BUILD_OPTIONS};-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0")
endif()

set(SP_3rdparty_DIR "${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece")

# cmake-format: off
ExternalProject_Add(sentencepiece_lib
  URL               https://github.com/google/sentencepiece/releases/download/v0.1.99/sentencepiece-0.1.99.tar.gz
  URL_HASH          MD5=6af04027121d138eb12c458a53df937e
  TIMEOUT           60
  SOURCE_DIR        ./sentencepiece-prefix
  BINARY_DIR        ./sentencepiece-prefix
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory "build" && ${CMAKE_COMMAND} -E chdir "build" ${CMAKE_COMMAND} ${SP_BUILD_OPTIONS} -DCMAKE_INSTALL_PREFIX=${SP_3rdparty_DIR} ../sentencepiece
  BUILD_COMMAND     ${CMAKE_COMMAND} -E chdir "build" make -j 
  INSTALL_COMMAND   ${CMAKE_COMMAND} -E chdir "build" make install
  TEST_COMMAND      ""
)
# cmake-format: on
