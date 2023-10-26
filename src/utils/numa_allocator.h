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
#include <cstdlib>

extern "C" {
// Allocate memory on preferred node, if the preferred node has been set
// If the preferred node is not set, allocate on local
void *xft_numa_alloc(size_t size);

// Allocate memory on a specified node
void *xft_numa_alloc_onnode(size_t size, int node);

// Free the memory allocated by us
void xft_numa_free(void *start, size_t size);

// Set preferred node to allocate memory
void xft_set_preferred_node(int node);
}