// Copyright (c) 2024 Intel Corporation
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
#include <cstdlib>
#include <iostream>
#include <stdlib.h>

bool enableCATMLP() {
    // combine gate&up and calculate together, default enabled
    static int catMlp = -1;
    if (catMlp == -1) catMlp = (getenv("ENABLE_CAT_MLP") ? atoi(getenv("ENABLE_CAT_MLP")) : 1);
    return catMlp == 1;
}

bool tunedComm() {
    // Tuning between shm and ccl reduceAdd methods to find the faster way, default enabled
    static int tunedComm = -1;
    if (tunedComm == -1) {
        tunedComm = (getenv("ENABLE_TUNED_COMM") ? atoi(getenv("ENABLE_TUNED_COMM")) : 1);
        if (tunedComm == 1) printf("ENABLE_TUNED_COMM is enabled for faster reduceAdd.\n");
    }
    return tunedComm == 1;
}

int getFlashThresh() {
    // > threshold to enable flash attention, default 1024
    static int envFlashThresh = -1;
    if (envFlashThresh == -1)
        envFlashThresh = (getenv("FLASH_ATTN_THRESHOLD") ? atoi(getenv("FLASH_ATTN_THRESHOLD")) : 1024);
    return envFlashThresh;
}

bool kvTrans() {
    // Transpose KV Tensor to [batchSize, headNum, seqLen, headSize] for better perf of long sequence, default disabled
    // TODO: add support for reorder and expand when beam_search>1
    static int kvTrans = -1;
    if (kvTrans == -1) { kvTrans = (getenv("ENABLE_KV_TRANS") ? atoi(getenv("ENABLE_KV_TRANS")) : 0); }
    return kvTrans == 1;
}
