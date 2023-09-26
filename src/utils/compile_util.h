#pragma once
#include <type_traits>

#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)

#define REQUIRES(assertion, message)           \
    do {                                       \
        if (unlikely(!(assertion))) {          \
            std::cout << message << std::endl; \
            exit(-1);                          \
        }                                      \
    } while (0)

// A class for forced loop unrolling at compile time
template <int i>
struct compile_time_for {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda &function, Args... args) {
        compile_time_for<i - 1>::op(function, args...);
        function(std::integral_constant<int, i - 1> {}, args...);
    }
};
template <>
struct compile_time_for<1> {
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda &function, Args... args) {
        function(std::integral_constant<int, 0> {}, args...);
    }
};
template <>
struct compile_time_for<0> {
    // 0 loops, do nothing
    template <typename Lambda, typename... Args>
    inline static void op(const Lambda &function, Args... args) {}
};