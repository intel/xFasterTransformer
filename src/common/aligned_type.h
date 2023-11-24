#pragma once

#include <cstddef>
#include <type_traits>

template <typename T, std::size_t Alignment>
struct AlignedType {
    alignas(Alignment) T data;

    // Default constructor
    AlignedType() = default;

    // Constructor to initialize with a value of type T
    explicit AlignedType(const T &value) : data(value) {}

    // Conversion operator to convert AlignedType to T
    operator T() const { return data; }

    // Overload the assignment operator to assign a value of type T
    AlignedType &operator=(const T &value) {
        data = value;
        return *this;
    }
};