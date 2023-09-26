#pragma once
#include <cassert>
#include <cstddef>

#include "float16.h"

typedef void (*BARRIER_CALLBACK)();
extern void shm_reduction(float *src, float *dst, int numel, int world_rank, int world_size);

class ShmReduction {
public:
    ShmReduction(BARRIER_CALLBACK callback, int world_rank, int world_size);

    ~ShmReduction() { shmClose(); }

    int getSHMSize();
    void shmClose();

    template <typename T>
    void reduceAdd(T *sendBuf, T *recvBuf, size_t count, int world_rank, int world_size) {
        if constexpr (std::is_same_v<T, float>) {
            shm_reduction(sendBuf, recvBuf, count, world_rank, world_size);
        } else if constexpr (std::is_same_v<T, float16_t>) {
            // TODO:
            printf("float16_t not supported!\n", typeid(T).name());
            exit(-1);
        } else {
            printf("Type %s not supported!\n", typeid(T).name());
            exit(-1);
        }
    }

    int m_rank;
};
