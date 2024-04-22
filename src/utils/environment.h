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
#include <sstream>
#include <string>
#include "dtype.h"

class Env {

public:
    // Meyers' Singleton
    static Env &getInstance() {
        static Env instance;
        return instance;
    }

    // Don't allow copying & assignment
    Env(Env const &) = delete;
    void operator=(Env const &) = delete;

    // get Verbose
    int getVerbose() { return verboseValue; }

    // get Pipeline Parallel
    xft::DeviceKind getEngineKind() { return engineKindValue; }
    int getEngineIndex() { return engineIndexValue; }

    // get Engine Kind and Index
    int getPipelineStage() { return pipelineStageValue; }

    // get Engine Kind and Index
    int getMaxRequestNum() { return maxRequestNumValue; }

    // get AMX Threshold M
    int getAMXThresholdM() { return AMXThresholdMValue; }

    // get THP_enabled
    bool getTHPEnabled() { return thpEnabled; }

    // get fake Model Enabled
    bool getFakeModelEnabled() { return fakeModelEnabled; }

    // get fake Load Info Enabled
    bool getFakeLoadInfoEnabled() { return fakeLoadInfoEnabled; }

    // get Debug Dir
    std::string getDebugDir() { return debugDir; }

    // get Timeline Whitelist
    std::string getTimelineWhitelist() { return timelineWhitelist; }

    // get Single Instance
    bool getSingleInstance() { return singleInstance; }

    // get OneCCL Enabled
    bool getOneCCLEnabled() { return oneCCLEnabled; }

private:
    Env() {
        // init Verbose
        initVerbose();

        // init Pipeline Parallel
        initPipelineStage();

        // init Max request number
        initMaxRequestNum();

        // init Engine Kind and Index
        initEngineKindIndex();

        // init AMX Threshold M
        initAMXThresholdM();

        // init THPEnabled
        initTHPEnabled();

        // init fake Model Enabled
        initFakeModelEnabled();

        // init fake Load Info Enabled
        initFakeLoadInfoEnabled();

        // init Debug Dir
        initDebugDir();

        // init Timeline Whitelist
        initTimelineWhitelist();

        // init Single Instance
        initSingleInstance();

        // init OneCCL Enabled
        initOneCCLEnabled();
    }

    // Verbose
    int verboseValue = 0;
    void initVerbose() {
        char *xft_verbose_value = getenv("XFT_VERBOSE");
        if (xft_verbose_value != NULL) {
            int value = atoi(xft_verbose_value);
            if (value >= 0)
                verboseValue = value;
            else
                printf("[ERROR] XFT_VERBOSE value need to be greater than or equal to 0.\n");
        } else {
            verboseValue = 0;
        }
    }

    // Engine Kind and Index
    xft::DeviceKind engineKindValue = xft::DeviceKind::iCPU;
    int engineIndexValue = 0;
    void initEngineKindIndex() {
        char *xft_engine_env = getenv("XFT_ENGINE");
        if (xft_engine_env != NULL) {
            std::string xft_engine_str(xft_engine_env);
            std::stringstream ss(xft_engine_str);
            std::string token;

            if (std::getline(ss, token, ':')) {
                if (token == "CPU") {
                    engineKindValue = xft::DeviceKind::iCPU;
                    engineIndexValue = 0;
                    return;
                } else if (token == "GPU")
                    engineKindValue = xft::DeviceKind::iGPU;
                else
                    printf("[ERROR] Undefined device kind in XFT_ENGINE.\n");
            } else {
                printf("[ERROR] Wrong value: XFT_ENGINE.\n");
            }

            if (std::getline(ss, token, ':')) {
                int value = std::stoi(token);
                if (value >= 0)
                    engineIndexValue = value;
                else
                    printf("[ERROR] Undefined device index in XFT_ENGINE.\n");
            } else {
                engineIndexValue = -1;
            }
        } else {
            engineKindValue = xft::DeviceKind::iCPU;
            engineIndexValue = 0;
        }
    }

    // Pipeline Parallel
    int pipelineStageValue = 1;
    void initPipelineStage() {
        char *xft_pipeline_value = getenv("XFT_PIPELINE_STAGE");
        if (xft_pipeline_value != NULL) {
#ifdef PIPELINE_PARALLEL
            int value = atoi(xft_pipeline_value);
            if (value >= 1)
                pipelineStageValue = value;
            else
                printf("[ERROR] XFT_PIPELINE_STAGE value need to be greater than 0.\n");
#else
            printf("[WARNING] XFT_PIPELINE_STAGE need to build with WITH_PIPELINE_PARALLEL=ON.\n");
#endif
        } else {
            pipelineStageValue = 1;
        }
    }

    // Max request number
    int maxRequestNumValue = 1;
    void initMaxRequestNum() {
        char *xft_max_request_num_value = getenv("XFT_MAX_REQUEST_NUM");
        if (xft_max_request_num_value != NULL) {
            int value = atoi(xft_max_request_num_value);
            if (value >= 1)
                maxRequestNumValue = value;
            else
                printf("[ERROR] XFT_MAX_REQUEST_NUM value need to be greater than 0.\n");
        } else {
            maxRequestNumValue = 1;
        }
    }

    // AMX Threshold M
    int AMXThresholdMValue = 1;
    void initAMXThresholdM() {
        char *xFTAMXThresholdMValue = getenv("XFT_USE_AMX_M");
        if (xFTAMXThresholdMValue != NULL) {
            int value = atoi(xFTAMXThresholdMValue);
            if (value >= 0)
                AMXThresholdMValue = value;
            else
                printf("[ERROR] XFT_USE_AMX_M value need to be greater than or equal to 0.\n");
        } else {
            AMXThresholdMValue = 1;
        }
    }

    // ENABLE_THP
    bool thpEnabled = false;
    void initTHPEnabled() {
        char *xftThpValue = getenv("ENABLE_THP");
        thpEnabled = xftThpValue != nullptr ? std::atoi(xftThpValue) : false;
    }

    // XFT_FAKE_MODEL
    bool fakeModelEnabled = false;
    void initFakeModelEnabled() {
        char *xftFakeModelValue = getenv("XFT_FAKE_MODEL");
        fakeModelEnabled = xftFakeModelValue != nullptr ? std::atoi(xftFakeModelValue) : false;
        if (fakeModelEnabled) {
            printf("[INFO] XFT_FAKE_MODEL is enabled. Using `export XFT_FAKE_LOAD_INFO=1` for more details.\n");
        }
    }

    // XFT_FAKE_LOAD_INFO
    bool fakeLoadInfoEnabled = false;
    void initFakeLoadInfoEnabled() {
        char *xftFakeLoadInfoValue = getenv("XFT_FAKE_LOAD_INFO");
        fakeLoadInfoEnabled = xftFakeLoadInfoValue != nullptr ? std::atoi(xftFakeLoadInfoValue) : false;
    }

    // XFT_DEBUG_DIR
    std::string debugDir = "";
    void initDebugDir() {
        char *xftDebugDirValue = getenv("XFT_DEBUG_DIR");
        debugDir = xftDebugDirValue != nullptr ? xftDebugDirValue : "";
    }

    // XFT_TIMELINE_WHITELIST
    std::string timelineWhitelist = "";
    void initTimelineWhitelist() {
        char *xftTimelineWhitelistValue = getenv("XFT_TIMELINE_WHITELIST");
        timelineWhitelist = xftTimelineWhitelistValue != nullptr ? xftTimelineWhitelistValue : "";
    }

    // SINGLE_INSTANCE
    bool singleInstance = false;
    void initSingleInstance() {
        char *xftSingleInstanceValue = getenv("SINGLE_INSTANCE");
        singleInstance = xftSingleInstanceValue != nullptr ? std::atoi(xftSingleInstanceValue) : false;
    }

    // XFT_ONECCL
    bool oneCCLEnabled = false;
    void initOneCCLEnabled() {
        char *xftOneCCLValue = getenv("XFT_ONECCL");
        oneCCLEnabled = xftOneCCLValue != nullptr ? std::atoi(xftOneCCLValue) : false;
    }
};