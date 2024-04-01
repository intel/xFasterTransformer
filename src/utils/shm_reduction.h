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

#define SHM_NAME "xft_shm_buffer"
#define MAX_SHM_SIZE (8 * 1024 * 5120 * 4)
#define SHM_BLOCK_SIZE (16 * 5120)
#define MAX_SHM_BLOCK_COUNT 2048

struct ShmContext {
    const char *name;
    int fp;
    int pid_fd[2];
    int *state;
    uint8_t *blockState;
    void *address;
    size_t nstates;
    size_t nblocks;
    size_t nbytes;
};

static inline int memfd_create(const char *name, unsigned int flags) {
    return syscall(__NR_memfd_create, name, flags);
}

inline void wait_state_until(const ShmContext *ctx, const int index, int state) {
    volatile int *state_ptr = ctx->state + index;
    while (*state_ptr != state)
        ;
}

inline void wait_block_until(const ShmContext *ctx, const int index, uint8_t state) {
    volatile uint8_t *state_ptr = ctx->blockState + index;
    while (*state_ptr != state)
        ;
}

inline void connect_shm(ShmContext *ctx) {
    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/%d/fd/%d", ctx->pid_fd[0], ctx->pid_fd[1]);
    ctx->fp = open(fd_path, O_RDWR);
    if (ctx->fp == -1) {
        perror("Bad file descriptor.");
        exit(-1);
    }

    const size_t total_size = ctx->nstates * sizeof(int) + ctx->nbytes + ctx->nblocks * ctx->nstates;

    // Map the shared memory into the address space of the process
    void *shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fp, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("shm mmap failed.");
        exit(-1);
    }
    ctx->state = (int *)shm_ptr;
    ctx->blockState = (uint8_t *)((int *)shm_ptr + ctx->nstates);
    ctx->address = (void *)((uint8_t *)ctx->blockState + ctx->nblocks * ctx->nstates);
}

inline void create_shm(ShmContext *ctx) {
    ctx->fp = memfd_create(ctx->name, MFD_CLOEXEC);

    if (ctx->fp == -1) {
        perror("shm open failed.");
        exit(-1);
    }
    const size_t total_size = ctx->nstates * sizeof(int) + ctx->nbytes + ctx->nblocks * ctx->nstates;
    // Truncate the shared memory to the desired size
    if (ftruncate(ctx->fp, total_size) == -1) {
        perror("shm ftruncate failed.");
        exit(-1);
    }

    // Map the shared memory into the address space of the process
    void *shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fp, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("shm mmap failed.");
        exit(-1);
    }
    ctx->pid_fd[0] = getpid();
    ctx->pid_fd[1] = ctx->fp;
    ctx->state = (int *)shm_ptr;
    ctx->blockState = (uint8_t *)((int *)shm_ptr + ctx->nstates);
    ctx->address = (void *)((uint8_t *)ctx->blockState + ctx->nblocks * ctx->nstates);
}

inline void close_shm(ShmContext *ctx) {
    const size_t total_size = ctx->nstates * sizeof(int) + ctx->nbytes;
    if (ctx->fp != -1) {
        munmap(ctx->address, total_size);
        shm_unlink(ctx->name);
    }
}

} // namespace xft

class ShmReduction {
public:
    ShmReduction(int rank, size_t size, std::function<void(int *, size_t)> callback);

    ~ShmReduction() { xft::close_shm(&shmCtx_); }

    size_t getSHMSize();

    void ShmResize(int rank, size_t size);

    template <typename T>
    void reduceAdd(T *sendBuf, T *recvBuf, size_t count, int rank, int rankSize);

    int rank_;
    int rank_size_;

private:
    xft::ShmContext shmCtx_;
};
