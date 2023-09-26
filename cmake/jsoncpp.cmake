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
ExternalProject_Add(jsoncpp_lib
  URL               https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/1.8.4.tar.gz
  URL_HASH          MD5=fa47a3ab6b381869b6a5f20811198662
  TIMEOUT           60
  SOURCE_DIR        ./jsoncpp-prefix
  BINARY_DIR        ./jsoncpp-prefix
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory "build" && ${CMAKE_COMMAND} -E chdir "build" ${CMAKE_COMMAND} -DCMAKE_CXX_FLAGS=-fPIC -DCMAKE_BUILD_TYPE=release -DBUILD_STATIC_LIBS=ON -DBUILD_SHARED_LIBS=ON -DJSONCPP_WITH_TESTS=OFF -DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_SOURCE_DIR}/3rdparty/jsoncpp ..
  BUILD_COMMAND     ${CMAKE_COMMAND} -E chdir "build" make
  INSTALL_COMMAND   ${CMAKE_COMMAND} -E chdir "build" make install
  TEST_COMMAND      ""
)
# cmake-format: on
