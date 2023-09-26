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
ExternalProject_Add(mklml
  URL               https://github.com/oneapi-src/oneDNN/releases/download/v0.21/mklml_lnx_2019.0.5.20190502.tgz
  URL_HASH          MD5=dfcea335652dbf3518e1d02cab2cea97
  TIMEOUT           60
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/mklml
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
# cmake-format: on
