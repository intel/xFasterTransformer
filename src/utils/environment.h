// Copyright (c) 2023-2024 Intel Corporation
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
#include <iostream>
#include "dtype.h"
#include <sstream>

class Env {

public:
    static void initEnvValue() {
        // init Verbose
        initVerbose();

        // init Pipeline Parallel
        initPipelineStage();

        // init Engine Kind and Index
        initEngineKindIndex();

        // init AMX Threshold M
        initAMXThresholdM();

        // TODO: Move XFT_FAKE_MODEL here.
        if (getenv("XFT_FAKE_MODEL") ? atoi(getenv("XFT_FAKE_MODEL")) : 0) {
            printf("[INFO] XFT_FAKE_MODEL is enabled. Using `export XFT_FAKE_LOAD_INFO=1` for more details.\n");
        }
    }

    // get Verbose
    static int getVerbose() { return verboseValue(); }

    // get Pipeline Parallel
    static xft::DeviceKind getEngineKind() { return engineKindValue(); }
    static int getEngineIndex() { return engineIndexValue(); }

    // get Engine Kind and Index
    static int getPipelineStage() { return pipelineStageValue(); }

    // get AMX Threshold M
    static int getAMXThresholdM() { return AMXThresholdMValue(); }

private:
    // Verbose
    static int &verboseValue() {
        static int value = 0;
        return value;
    }

    static void initVerbose() {
        char *xft_verbose_value = getenv("XFT_VERBOSE");
        if (xft_verbose_value != NULL) {
            int value = atoi(xft_verbose_value);
            if (value >= 0)
                verboseValue() = value;
            else
                printf("[ERROR] XFT_VERBOSE value need to be greater than or equal to 0.\n");
        } else {
            verboseValue() = 0;
        }
    }

    // Engine Kind and Index
    static xft::DeviceKind &engineKindValue() {
        static xft::DeviceKind value = xft::DeviceKind::iCPU;
        return value;
    }

    static int &engineIndexValue() {
        static int value = 0;
        return value;
    }

    static void initEngineKindIndex() {
        char *xft_engine_env = getenv("XFT_ENGINE");
        if (xft_engine_env != NULL) {
            std::string xft_engine_str(xft_engine_env);
            std::stringstream ss(xft_engine_str);
            std::string token;

            if (std::getline(ss, token, ':')) {
                if (token == "CPU") {
                    engineKindValue() = xft::DeviceKind::iCPU;
                    engineIndexValue() = 0;
                    return;
                } else if (token == "GPU")
                    engineKindValue() = xft::DeviceKind::iGPU;
                else
                    printf("[ERROR] Undefined device kind in XFT_ENGINE.\n");
            } else {
                printf("[ERROR] Wrong value: XFT_ENGINE.\n");
            }

            if (std::getline(ss, token, ':')) {
                int value = std::stoi(token);
                if (value >= 0)
                    engineIndexValue() = value;
                else
                    printf("[ERROR] Undefined device index in XFT_ENGINE.\n");
            } else {
                engineIndexValue() = -1;
            }
        } else {
            engineKindValue() = xft::DeviceKind::iCPU;
            engineIndexValue() = 0;
        }
    }

    // Pipeline Parallel
    static int &pipelineStageValue() {
        static int value = 1;
        return value;
    }

    static void initPipelineStage() {
        char *xft_pipeline_value = getenv("XFT_PIPELINE_STAGE");
        if (xft_pipeline_value != NULL) {
#ifdef PIPELINE_PARALLEL
            int value = atoi(xft_pipeline_value);
            if (value >= 1)
                pipelineStageValue() = value;
            else
                printf("[ERROR] XFT_PIPELINE_STAGE value need to be greater than 0.\n");
#else
            printf("[WARNING] XFT_PIPELINE_STAGE need to build with WITH_PIPELINE_PARALLEL=ON.\n");
#endif
        } else {
            pipelineStageValue() = 1;
        }
    }

    // AMX Threshold M
    static int &AMXThresholdMValue() {
        static int value = 1;
        return value;
    }

    static void initAMXThresholdM() {
        char *xFTAMXThresholdMValue = getenv("XFT_USE_AMX_M");
        if (xFTAMXThresholdMValue != NULL) {
            int value = atoi(xFTAMXThresholdMValue);
            if (value >= 0)
                AMXThresholdMValue() = value;
            else
                printf("[ERROR] XFT_USE_AMX_M value need to be greater than or equal to 0.\n");
        } else {
            AMXThresholdMValue() = 1;
        }
    }

};