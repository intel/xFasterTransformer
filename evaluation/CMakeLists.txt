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
cmake_minimum_required(VERSION 3.15.1)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} EVAL_SRC)

set(CLONE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lm-evaluation-harness")

find_package (Python COMPONENTS Interpreter Development)

if(NOT EXISTS ${CLONE_DIR})

    execute_process(COMMAND git clone https://github.com/EleutherAI/lm-evaluation-harness.git ${CLONE_DIR}
                    RESULT_VARIABLE GIT_CLONE_RESULT)

    if(NOT GIT_CLONE_RESULT EQUAL "0")
        message(FATAL_ERROR "Failed to clone Git repository.")
    endif()

    execute_process(COMMAND git -c advice.detachedHead=false checkout v0.4.0
                    WORKING_DIRECTORY ${CLONE_DIR}
                    RESULT_VARIABLE GIT_CHECKOUT_RESULT
                    OUTPUT_QUIET)

    if(NOT GIT_CHECKOUT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to checkout commit.")
    endif()

    execute_process(COMMAND ${Python_EXECUTABLE} -m pip install -e .
                            #COMMAND ${Python_EXECUTABLE} -m pip install -e ".[multilingual]"
                    WORKING_DIRECTORY ${CLONE_DIR}
                    RESULT_VARIABLE EXIT_CODE
                    OUTPUT_QUIET)

    if(${EXIT_CODE} EQUAL 0)
        message(STATUS "lm-eval installed.")
    else()
        message(FATAL_ERROR "Failed to install.")
    endif()

endif()


execute_process(
    COMMAND python -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "share/cmake" "include" PyTorch_INCLUDE_DIR "${TORCH_CMAKE_PREFIX_PATH}")
string(REGEX REPLACE "\n$" "" PyTorch_INCLUDE_DIR "${PyTorch_INCLUDE_DIR}")

string(REPLACE "share/cmake" "lib" PyTorch_LIBRARIES_DIR "${TORCH_CMAKE_PREFIX_PATH}")
string(REGEX REPLACE "\n$" "" PyTorch_LIBRARIES_DIR "${PyTorch_LIBRARIES_DIR}")

find_library(TORCH_LIB torch HINTS ${PyTorch_LIBRARIES_DIR})
find_library(TORCH_GLOBAL_DEPS_LIB torch_global_deps HINTS ${PyTorch_LIBRARIES_DIR})
find_library(TORCH_CPU_LIB torch_cpu HINTS ${PyTorch_LIBRARIES_DIR})
find_library(TORCH_PYTHON_LIB torch_python HINTS ${PyTorch_LIBRARIES_DIR})
set(TORCH_LIBS ${TORCH_GLOBAL_DEPS_LIB} ${TORCH_CPU_LIB} ${TORCH_PYTHON_LIB})


add_library(evaluation SHARED ${EVAL_SRC})

target_include_directories(evaluation PUBLIC
			   ${PyTorch_INCLUDE_DIR})

if(BUILD_WITH_SHARED_LIBS)
    target_link_libraries(evaluation PRIVATE xfastertransformer ${TORCH_LIBS})
else()
    target_link_libraries(evaluation PRIVATE xfastertransformer_static ${TORCH_LIBS})
endif()
