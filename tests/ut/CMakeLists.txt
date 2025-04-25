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
# Unit tests
cmake_minimum_required(VERSION 3.15.1)

project(LLM_DISTRIBUTED_INFER)

find_package(GTest REQUIRED)
find_package(oneCCL REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ut)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} sources)

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)

    if(${executable} STREQUAL "messenger_test")
        add_executable(messenger_test ${src} ${SRC_DIR}/utils/shm_ccl.cpp)
    elseif(${executable} STREQUAL "shm_test")
        add_executable(shm_test ${src} ${SRC_DIR}/utils/shm_ccl.cpp)
    elseif(${executable} STREQUAL "token_embedding_test")
        add_executable(token_embedding_test
                       ${src}
                       ${SRC_DIR}/layers/layer_norm.cpp
                       ${SRC_DIR}/models/opt_decoder.cpp
                       ${SRC_DIR}/models/kvcache_manager.cpp
                       ${SRC_DIR}/utils/numa_allocator.cpp
                       ${SRC_DIR}/utils/shm_ccl.cpp
                       ${SRC_DIR}/kernels/token_embedding_kernels.cpp)
    elseif(${executable} STREQUAL "kv_reorder_test")
        add_executable(kv_reorder_test
                       ${src}
                       ${SRC_DIR}/layers/layer_norm.cpp
                       ${SRC_DIR}/models/opt_decoder.cpp
                       ${SRC_DIR}/models/kvcache_manager.cpp
                       ${SRC_DIR}/utils/numa_allocator.cpp
                       ${SRC_DIR}/utils/shm_ccl.cpp
                       ${SRC_DIR}/kernels/gemm_kernel_ext.cpp)
    elseif(${executable} STREQUAL "alibi_embedding_test")
        add_executable(alibi_embedding_test ${src} ${SRC_DIR}/layers/alibi_embedding.cpp)
    elseif(${executable} STREQUAL "rotary_embedding_test")
        add_executable(rotary_embedding_test ${src} ${SRC_DIR}/layers/rotary_embedding.cpp)
    elseif(${executable} STREQUAL "gemm_kernel_ext_test")
        add_executable(gemm_kernel_ext_test ${src} ${SRC_DIR}/kernels/gemm_kernel_ext.cpp)
    elseif(${executable} STREQUAL "timeline_test")
        if(NOT WITH_TIMELINE)
            continue()
        endif()
        add_executable(timeline_test ${src})
    elseif(${executable} STREQUAL "repetition_penalty_test")
        add_executable(repetition_penalty_test ${src} ${SRC_DIR}/searchers/search_utils.cpp)
    else()
        add_executable(${executable} ${src})
    endif()

    target_link_libraries(${executable} PRIVATE ${GTEST_LIBRARIES})
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PUBLIC gtest)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    target_link_libraries(${executable} PUBLIC pthread)
    target_link_libraries(${executable} PUBLIC stdc++)
    target_link_libraries(${executable} PUBLIC mpi)
    target_link_libraries(${executable} PUBLIC numa)
    target_link_libraries(${executable} PUBLIC xfastertransformer)

    # List of executable names and their corresponding libraries
    set(executables_need_gemm kv_reorder_test small_gemm_test)

    # Gemm libraries needed for all executables
    foreach(name ${executables_need_gemm})
        if(${executable} STREQUAL ${name})
            target_link_libraries(${executable} PRIVATE ${GEMM_KERNEL_FILES} dnnl)
        endif()
        add_dependencies(${executable} onednn)
    endforeach()

    # gtest_discover_tests(${executable})
endforeach()

# enable_testing()
