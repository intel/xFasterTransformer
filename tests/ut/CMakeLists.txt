# Unit tests
cmake_minimum_required(VERSION 3.15.1)

project(LLM_DISTRIBUTED_INFER)

# GTEST uses ABI=1 and Pytorch defaultly uses ABI=0
string(FIND "${CMAKE_CXX_FLAGS}" "-D_GLIBCXX_USE_CXX11_ABI=0" FLAG_INDEX)

if(FLAG_INDEX GREATER -1)
    message(STATUS
                "Found -D_GLIBCXX_USE_CXX11_ABI=0 in CMAKE_CXX_FLAGS. Replacing with -D_GLIBCXX_USE_CXX11_ABI=1."
    )
    string(REPLACE "-D_GLIBCXX_USE_CXX11_ABI=0" "-D_GLIBCXX_USE_CXX11_ABI=1" CMAKE_CXX_FLAGS
                   "${CMAKE_CXX_FLAGS}")
else()
    message(STATUS "Did not find -D_GLIBCXX_USE_CXX11_ABI=0 in CMAKE_CXX_FLAGS.")
endif()

find_package(GTest REQUIRED)
find_package(oneCCL REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/ut)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src/)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} sources)

foreach(src ${sources})
    get_filename_component(executable ${src} NAME_WE)

    if(${executable} STREQUAL "messenger_test")
        add_executable(messenger_test ${src} ${SRC_DIR}/utils/shm_reduction.cpp)
    elseif(${executable} STREQUAL "shm_test")
        add_executable(shm_test ${src} ${SRC_DIR}/utils/shm_reduction.cpp)
    elseif(${executable} STREQUAL "kv_reorder_test")
        add_executable(kv_reorder_test
                       ${src}
                       ${SRC_DIR}/models/opt_decoder.cpp
                       ${SRC_DIR}/models/kvcache_manager.cpp
                       ${SRC_DIR}/utils/numa_allocator.cpp
                       ${SRC_DIR}/utils/shm_reduction.cpp
                       ${SRC_DIR}/kernels/gemm_kernel_ext.cpp
                       ${SRC_DIR}/kernels/gemm_kernel_ext_fp16.cpp)
    elseif(${executable} STREQUAL "beam_search_test")
        add_executable(beam_search_test
                       ${src}
                       ${SRC_DIR}/models/opt_decoder.cpp
                       ${SRC_DIR}/models/kvcache_manager.cpp
                       ${SRC_DIR}/searchers/beam_search.cpp
                       ${SRC_DIR}/utils/numa_allocator.cpp
                       ${SRC_DIR}/utils/shm_reduction.cpp
                       ${SRC_DIR}/kernels/gemm_kernel_ext.cpp
                       ${SRC_DIR}/kernels/gemm_kernel_ext_fp16.cpp)
    elseif(${executable} STREQUAL "rotary_embedding_test")
        add_executable(rotary_embedding_test ${src} ${SRC_DIR}/layers/rotary_embedding.cpp)
    elseif(${executable} STREQUAL "gemm_kernel_ext_test")
        add_executable(gemm_kernel_ext_test ${src} ${SRC_DIR}/kernels/gemm_kernel_ext.cpp)
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
    target_link_libraries(${executable} PUBLIC numa)

    # List of executable names and their corresponding libraries
    set(executables_need_gemm kv_reorder_test beam_search_test small_gemm_test)

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
