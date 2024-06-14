// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once
#include <chrono>
#include <cinttypes>
#include <iostream>
#include <mutex>

#include "dtype.h"
#include "environment.h"

class FunTimer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
    FunTimer() : start_time(std::chrono::high_resolution_clock::now()) {}

    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed_seconds = end_time - start_time;
        return elapsed_seconds.count();
    }
};

class Printer {
public:
    static void gemm(const char *api_func, int M, int N, int K, double ms) {
        printf("xft_verbose,exec,cpu,api,%s,m%dn%dk%d,%.6lf\n", api_func, M, N, K, ms);
        fflush(stdout);
    }

    static void matrix(int rows, int cols, int stride, size_t totalmem) {
        printf("xft_verbose,matrix:rows%d_cols%d_stride%d,use:%zu bytes of memory\n", rows, cols, stride, totalmem);
        fflush(stdout);
    }

    template <typename T>
    static void print(std::string buf_name, T *buf, int rows, int cols, int stride, bool printAll = false,
            void *device = nullptr) {
        std::cout << buf_name.c_str() << ":" << std::endl;
#ifdef XFT_GPU
        if (device != nullptr) {
            sycl::queue *gpu_queue = static_cast<sycl::queue *>(device);
            gpu_queue
                    ->submit([&](sycl::handler &cgh) {
                        auto out = sycl::stream(10240, 7680, cgh);
                        cgh.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1> item) {
                            int idx_col = item.get_global_id(0);
                            if (idx_col == 0) {
                                if (printAll == false) {
                                    for (int row = 0; row < 6; ++row) {
                                        for (int col = 0; col < 6; ++col) {
                                            out << (float)buf[row * stride + col] << ", ";
                                        }
                                        out << " ... ";
                                        for (int col = cols - 6; col < cols; ++col) {
                                            out << (float)buf[row * stride + col] << ", ";
                                        }
                                        out << sycl::endl;
                                    }
                                    out << "..." << sycl::endl;
                                    for (int row = rows - 6; row < rows; ++row) {
                                        for (int col = 0; col < 6; ++col) {
                                            out << (float)buf[row * stride + col] << ", ";
                                        }
                                        out << " ... ";
                                        for (int col = cols - 6; col < cols; ++col) {
                                            out << (float)buf[row * stride + col] << ", ";
                                        }
                                        out << sycl::endl;
                                    }
                                    out << sycl::endl;
                                } else {
                                    for (int row = 0; row < rows; ++row) {
                                        for (int col = 0; col < 6; ++col) {
                                            out << (float)buf[row * stride + col] << ", ";
                                        }
                                        out << " ... ";
                                        for (int col = cols - 6; col < cols; ++col) {
                                            out << (float)buf[row * stride + col] << ", ";
                                        }
                                        out << sycl::endl;
                                    }
                                }
                            }
                        });
                    })
                    .wait();
        } else {
            for (int row = 0; row < 6; ++row) {
                for (int col = 0; col < 6; ++col) {
                    std::cout << (float)buf[row * stride + col] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << "..." << std::endl;
            for (int row = rows - 6; row < rows; ++row) {
                for (int col = cols - 6; col < cols; ++col) {
                    std::cout << (float)buf[row * stride + col] << ", ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
#endif
    }
};

#define GEMMVERBOSE(api_func, compute_func)                \
    if (Env::getInstance().getVerbose() >= 1) {            \
        TimeLine t(api_func);                              \
        FunTimer timer;                                    \
        compute_func;                                      \
        Printer::gemm(api_func, M, N, K, timer.elapsed()); \
    } else {                                               \
        TimeLine t(api_func);                              \
        compute_func;                                      \
    }
