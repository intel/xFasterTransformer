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
ExternalProject_Add(cmdline
  URL               https://github.com/tanakh/cmdline/archive/refs/heads/master.zip
  URL_HASH          MD5=69f98dc95edcae8c423a62ceccf81644
  TIMEOUT           60
  SOURCE_DIR        ${CMAKE_SOURCE_DIR}/3rdparty/cmdline
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
# cmake-format: on
