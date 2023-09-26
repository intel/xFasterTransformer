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
ExternalProject_Add(sentencepiece_lib
  URL               https://github.com/google/sentencepiece/releases/download/v0.1.99/sentencepiece-0.1.99.tar.gz
  URL_HASH          MD5=6af04027121d138eb12c458a53df937e
  TIMEOUT           60
  SOURCE_DIR        ./sentencepiece-prefix
  BINARY_DIR        ./sentencepiece-prefix
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory "build" && ${CMAKE_COMMAND} -E chdir "build" ${CMAKE_COMMAND} -DSPM_ENABLE_SHARED=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/3rdparty/sentencepiece ../sentencepiece
  BUILD_COMMAND     ${CMAKE_COMMAND} -E chdir "build" make -j 
  INSTALL_COMMAND   ${CMAKE_COMMAND} -E chdir "build" make install
  TEST_COMMAND      ""
)
# cmake-format: on
