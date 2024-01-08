#pragma once
#include "bfloat16.h"

// Selected data types according to weight data type
template <typename WeiT>
struct TypeSelector {
    using InType = float;
    using ImType = float; // intermediate data type, default in float
    using OutType = float;
};

// Specialization for bfloat16_t
template <>
struct TypeSelector<bfloat16_t> {
    using InType = float;
    // TODO: change it to bfloat16_t, make it work and verify the accuray
    using ImType = float;
    using OutType = float;
};