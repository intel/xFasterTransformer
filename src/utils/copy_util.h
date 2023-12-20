#pragma once

#include <cstdio>
#include <cstdlib>
#include <typeinfo>
#include "bfloat16.h"
#include "float16.h"

namespace xft {

template <typename T1, typename T2>
void copy(T1 *dst, T2 *src, int size) {
    if constexpr (std::is_same_v<T1, T2>) {
        memcpy(dst, src, size * sizeof(T1));
    } else {
        printf("Not supported yet: copy(%s, %s)\n", typeid(T1).name(), typeid(T2).name());
        exit(-1);
    }
}

// Specialization for T1=float, T2=float16_t
template <>
void copy(float *dst, float16_t *src, int size) {
    float16_t::cvt_float16_to_float(src, dst, size);
}

// Specialization for T1=float16_t, T2=float
template <>
void copy(float16_t *dst, float *src, int size) {
    float16_t::cvt_float_to_float16(src, dst, size);
}

// Specialization for T1=float, T2=bloat16_t
template <>
void copy(float *dst, bfloat16_t *src, int size) {
    bfloat16_t::cvt_bfloat16_to_float(src, dst, size);
}

// Specialization for T1=boat16_t, T2=float
template <>
void copy(bfloat16_t *dst, float *src, int size) {
    bfloat16_t::cvt_float_to_bfloat16(src, dst, size);
}

} // namespace xft