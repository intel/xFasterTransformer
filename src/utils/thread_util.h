#pragma once
#include <omp.h>

template <typename Lambda>
void parallel_for(int tasks, const Lambda &fn) {
#pragma omp parallel for
    for (int i = 0; i < tasks; i++) {
        fn(i);
    }
}

template <typename Lambda>
void parallel_for_dschedule(int tasks, const Lambda &fn) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < tasks; i++) {
        fn(i);
    }
}