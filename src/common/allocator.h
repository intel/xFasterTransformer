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
#pragma once
#include <cstdio>
#include <cstdlib>
#include <sys/mman.h>
#include "environment.h"

namespace xft {

constexpr size_t g_thp_threshold = (size_t)2 * 1024 * 1024;

static inline bool is_thp_alloc(size_t nbytes) {
    return (Env::getInstance().getTHPEnabled() && (nbytes >= g_thp_threshold));
}

static inline void *alloc(size_t nbytes, size_t alignment = 64) {
    if (nbytes == 0) { return nullptr; }

    void *data;

    int err = posix_memalign(&data, alignment, nbytes);
    if (err != 0) {
        printf("Unable to allocate buffer with size of %zu, err=%d\n", nbytes, err);
        exit(-1);
    }

    if (is_thp_alloc(nbytes)) {
        // Advise to use huge page
        int ret = madvise(data, nbytes, MADV_HUGEPAGE);
        if (ret != 0) {
            // TODO: show warning
        }
    }

    return data;
}
} // namespace xft