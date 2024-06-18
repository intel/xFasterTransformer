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
