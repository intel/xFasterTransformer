# Unit tests
cmake_minimum_required(VERSION 3.15.1)

project(LLM_DISTRIBUTED_INFER)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -fpermissive")

find_package(GTest REQUIRED)
find_package(oneCCL REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
include_directories("../../src")
include_directories("../../3rdparty")
include_directories("../../src/kernels")
include_directories("../../src/layers")
include_directories("../../src/searchers")
include_directories("../../src/utils")
include_directories("../../src/models")
include_directories("../../src/common")

file(GLOB sources "*.cpp")

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)

    if(${executable} STREQUAL "messenger_test")
        add_executable(messenger_test ${src} ../../src/utils/shm_reduction.cpp
                                      ../../src/utils/messenger.cpp)
    elseif(${executable} STREQUAL "kv_reorder_test")
        add_executable(kv_reorder_test ${src} ../../src/models/opt_decoder.cpp
                                       ../../src/kernels/gemm_kernel_ext.cpp)
    elseif(${executable} STREQUAL "beam_search_test")
        add_executable(beam_search_test ${src} ../../src/models/opt_decoder.cpp
                                        ../../src/kernels/gemm_kernel_ext.cpp)
    elseif(${executable} STREQUAL "rotary_embedding_test")
        add_executable(rotary_embedding_test ${src} ../../src/layers/rotary_embedding.cpp)
    elseif(${executable} STREQUAL "gemm_kernel_ext_test")
        add_executable(gemm_kernel_ext_test ${src} ../../src/kernels/gemm_kernel_ext.cpp)
    else()
        add_executable(${executable} ${src})
    endif()

    target_link_libraries(${executable} PRIVATE ${GTEST_LIBRARIES})
    target_link_libraries(${executable} PRIVATE ccl)
    target_link_libraries(${executable} PUBLIC rt)
    target_link_libraries(${executable} PUBLIC m)
    target_link_libraries(${executable} PUBLIC dl)
    target_link_libraries(${executable} PUBLIC pthread)
    target_link_libraries(${executable} PUBLIC stdc++)
    target_link_libraries(${executable} PUBLIC mpi)

    if(${executable} STREQUAL "kv_reorder_test")
        target_link_libraries(${executable}
                              PRIVATE
                                  ${CMAKE_CURRENT_LIST_DIR}/../../src/kernels/libsgemm.a
                                  ${CMAKE_CURRENT_LIST_DIR}/../../src/kernels/libsgemm_f32f16f32.a
                                  ${CMAKE_CURRENT_LIST_DIR}/../../src/kernels/libhgemm_f32f16f32.a)
    endif()

    if(${executable} STREQUAL "beam_search_test")
        target_link_libraries(${executable}
                              PRIVATE
                                  ${CMAKE_CURRENT_LIST_DIR}/../../src/kernels/libsgemm.a
                                  ${CMAKE_CURRENT_LIST_DIR}/../../src/kernels/libsgemm_f32f16f32.a
                                  ${CMAKE_CURRENT_LIST_DIR}/../../src/kernels/libhgemm_f32f16f32.a)
    endif()
endforeach()
