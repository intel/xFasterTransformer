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
ExternalProject_Add(xdnn_lib
  URL               https://github.com/intel/xFasterTransformer/releases/download/IntrinsicGemm/xdnn_v1.5.9.tar.gz
  URL_HASH          MD5=3aa9cd15df3eb2a7a1c178f3edcf9d37
  TIMEOUT           120
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/xdnn
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
# cmake-format: on
