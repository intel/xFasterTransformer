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
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include "float16.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>

namespace xft {

#define SHM_NAME                "xft_shm_buffer"
#define MAX_RANK_NUMBER         16
#define MAX_SHM_SIZE            (1024 * 5120 * 8 * 4)

struct ShmContext {
    // current rank info
    char name[64];                  // shared memory file name
    int rank_idx;                   // index of current rank
    int pid_fd[2];                  // 0 - pid; 1 - shared memory fd

    // global shm info
    int nrank;                      // number of total ranks
    void *shm_ptr[MAX_RANK_NUMBER]; // shared memory addr per rank

    // per rank data in shared memory
    int     *all_pids;              // 2 per rank, handles: 0 - pid; 1 - shared memory fd; valid in rank 0
    int     *state;                 // 1, state per rank, 
                                    //          0-idle, 1-op start, 2-data prepared, 
                                    //          3-segment reduce-add finished, 4-rank update finished
    void    *address;               // MAX_SHM_SIZE, raw data addr
};

static inline int memfd_create(const char *name, unsigned int flags) {
    return syscall(__NR_memfd_create, name, flags);
}

inline void wait_state_until(const ShmContext *ctx, const int index, int state) {
    volatile int *state_ptr = (int*)(ctx->shm_ptr[index]) + MAX_RANK_NUMBER * 2;
    while (*state_ptr != state)
        ; //sched_yield();
}

inline void close_shm(ShmContext *ctx) {
    const int total_size = sizeof(int) * (MAX_RANK_NUMBER * 2  + 1) + MAX_SHM_SIZE;
    if (ctx->pid_fd[1] != -1) {
        munmap(ctx->shm_ptr[ctx->rank_idx], total_size);
        shm_unlink(ctx->name);
    }
}

} // namespace xft

class ShmReduction {
public:
    ShmReduction(int rank, int size, std::function<void(int *, size_t)> callback);

    ~ShmReduction() { xft::close_shm(&shmCtx_); }

    int getSHMSize();

    template <typename T>
    void reduceAdd(T *sendBuf, T *recvBuf, size_t count, int rank, int rankSize);

    int rank_;
    int rank_size_;

private:
    xft::ShmContext shmCtx_;
};
