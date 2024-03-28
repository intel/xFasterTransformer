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
    static int catMlp = -1;
    if (catMlp == -1) catMlp = (getenv("ENABLE_CAT_MLP") ? atoi(getenv("ENABLE_CAT_MLP")) : 1);
    return catMlp == 1;
}

bool tunedComm() {
    static int tunedComm = -1;
    if (tunedComm == -1) {
        tunedComm = (getenv("ENABLE_TUNED_COMM") ? atoi(getenv("ENABLE_TUNED_COMM")) : 1);
        if (tunedComm == 1) printf("ENABLE_TUNED_COMM is enabled for faster reduceAdd.\n");
    }
    return tunedComm == 1;
}

int getFlashThresh() {
    static int envFlashThresh = -1;
    if (envFlashThresh == -1)
        envFlashThresh = (getenv("FLASH_ATTN_THRESHOLD") ? atoi(getenv("FLASH_ATTN_THRESHOLD")) : 1024);
    return envFlashThresh;
}

bool enableSkipMsk() {
    static int skipMsk = -1;
    if (skipMsk == -1) {
        skipMsk = (getenv("ENABLE_SKIP_MASK") ? atoi(getenv("ENABLE_SKIP_MASK")) : 0);
        if (skipMsk == 1) printf("ENABLE_SKIP_MASK is enabled for ignoring mask Q*K.\n");
    }
    return skipMsk == 1;
}

bool kvTrans() {
    static int kvTrans = -1;
    if (kvTrans == -1) {
        kvTrans = (getenv("ENABLE_KV_TRANS") ? atoi(getenv("ENABLE_KV_TRANS")) : 0);
        // if (kvTrans == 1)
        //     printf("ENABLE_KV_TRANS is enabled for kv cache optimization.\n");
    }
    return kvTrans == 1;
}
