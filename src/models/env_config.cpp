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
    if (catMlp == -1)
        catMlp = (getenv("ENABLE_CAT_MLP") ? atoi(getenv("ENABLE_CAT_MLP")) : 1);
    return catMlp == 1;
}

int getFlashThresh() {
    static int envFlashThresh = -1;
    if (envFlashThresh == -1)
        envFlashThresh = (getenv("FLASH_ATTN_THRESHOLD") ? atoi(getenv("FLASH_ATTN_THRESHOLD")) : 1024);
    return envFlashThresh;
}
