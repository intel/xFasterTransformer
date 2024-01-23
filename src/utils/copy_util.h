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