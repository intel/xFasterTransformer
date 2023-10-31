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
#include "numa_allocator.h"
#include <cstdio>
#include <numa.h>

static int preferredNode = -1;

void *xft_numa_alloc(size_t size) {
    return xft_numa_alloc_onnode(size, preferredNode);
}

void *xft_numa_alloc_onnode(size_t size, int node) {
    void *memory = nullptr;

    if (node >= 0) {
        memory = numa_alloc_onnode(size, node);
    } else {
        memory = numa_alloc(size);
    }

    if (memory == nullptr) {
        printf("Failed to allocate memory (size=%zu, node=%d)\n", size, node);
        exit(-1);
    }

    return memory;
}

void xft_numa_free(void *start, size_t size) {
    numa_free(start, size);
}

void xft_set_preferred_node(int node) {
    preferredNode = node;
}