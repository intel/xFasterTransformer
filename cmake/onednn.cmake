# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the Apache V2.0 License.

cmake_minimum_required(VERSION 3.18)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

project(dependency NONE)

include(ExternalProject)

# cmake-format: off
ExternalProject_Add(onednn
  GIT_REPOSITORY    https://github.com/oneapi-src/oneDNN.git
  GIT_TAG           v3.2
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/onednn
  BINARY_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/onednn
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory "build" && ${CMAKE_COMMAND} -E chdir "build" ${CMAKE_COMMAND} -DONEDNN_LIBRARY_TYPE=STATIC -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF ..
  BUILD_COMMAND     ${CMAKE_COMMAND} -E chdir "build" make -j all
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
# cmake-format: on
