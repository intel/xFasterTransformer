#pragma once

#include <cstdint>
#include <type_traits>

template <typename T1, typename T2>
inline T1 bit_cast(const T2 &u) {
    static_assert(sizeof(T1) == sizeof(T2), "Bit-casting must preserve size.");
    static_assert(std::is_trivial<T1>::value, "T1 must be trivially copyable.");
    static_assert(std::is_trivial<T2>::value, "T2 must be trivially copyable.");

    T1 t;
    uint8_t *t_ptr = reinterpret_cast<uint8_t *>(&t);
    const uint8_t *u_ptr = reinterpret_cast<const uint8_t *>(&u);
    for (size_t i = 0; i < sizeof(T2); i++)
        t_ptr[i] = u_ptr[i];
    return t;
}
