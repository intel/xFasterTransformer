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
#include <typeinfo>
#include "intrinsics_util.h"

namespace xft {

template <typename T1, typename T2>
inline void copy(T1 *dst, T2 *src, int size) {
    if constexpr (std::is_same_v<T1, T2>) {
        memcpy(dst, src, size * sizeof(T1));
    } else {
        constexpr int kStep = 16;
        int blockSize = size / kStep;
        int remainder = size % kStep;

        for (int i = 0; i < blockSize; ++i) {
            __m512 v = load_avx512(0xffff, src + i * kStep);
            store_avx512(dst + i * kStep, 0xffff, v);
        }

        if (remainder != 0) {
            __mmask16 mask = 0xFFFF >> (kStep - remainder);
            __m512 v = load_avx512(mask, src + size - remainder);
            store_avx512(dst + size - remainder, mask, v);
        }
    }
}
} // namespace xft