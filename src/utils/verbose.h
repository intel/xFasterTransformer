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

class Env {
private:
    static int &verboseValue() {
        static int value = 0;
        return value;
    }

public:
    static void initVerbose() {
        char *xft_verbose_value = getenv("XFT_VERBOSE");
        if (xft_verbose_value != NULL) {
            int value = atoi(xft_verbose_value);
            verboseValue() = value;
        } else {
            verboseValue() = 0;
        }

        // TODO: Move XFT_FAKE_MODEL here.
        if (getenv("XFT_FAKE_MODEL") ? atoi(getenv("XFT_FAKE_MODEL")) : 0) {
            printf("[INFO] XFT_FAKE_MODEL is enabled. Using `export XFT_FAKE_LOAD_INFO=1` for more details.\n");
        }
    }

    static int getVerbose() { return verboseValue(); }

// Pipeline Parallel
private:
    static int &pipelineStageValue() {
        static int value = 1;
        return value;
    }

public:
    static void initPipelineStage() {
        char *xft_pipeline_value = getenv("XFT_PIPELINE_STAGES");
        if (xft_pipeline_value != NULL) {
#ifdef PIPELINE_PARALLEL
            int value = atoi(xft_pipeline_value);
            if (value >= 1)
                pipelineStageValue() = value;
#else
            printf("[WARNING] XFT_PIPELINE_STAGES need to build with WITH_PIPELINE_PARALLEL=ON.\n");
#endif
        } else {
            pipelineStageValue() = 1;
        }
    }

    static int getPipelineStage() { return pipelineStageValue(); }
};

#define GEMMVERBOSE(api_func, compute_func)                \
    if (Env::getVerbose() >= 1) {                          \
        TimeLine t(api_func);                              \
        FunTimer timer;                                    \
        compute_func;                                      \
        Printer::gemm(api_func, M, N, K, timer.elapsed()); \
    } else {                                               \
        TimeLine t(api_func);                              \
        compute_func;                                      \
    }
