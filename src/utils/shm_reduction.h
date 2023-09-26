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
#define SHM_BLOCK_SIZE (128 * 5120)

struct ShmContext {
    const char *name;
    int fp;
    int pid_fd[2];
    int *state;
    void *address;
    size_t nstates;
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

inline void connect_shm(ShmContext *ctx) {
    char fd_path[64];
    snprintf(fd_path, sizeof(fd_path), "/proc/%d/fd/%d", ctx->pid_fd[0], ctx->pid_fd[1]);
    ctx->fp = open(fd_path, O_RDWR);
    if (ctx->fp == -1) {
        perror("Bad file descriptor.");
        exit(-1);
    }

    const int total_size = ctx->nstates * sizeof(int) + ctx->nbytes;

    // Map the shared memory into the address space of the process
    void *shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fp, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("shm mmap failed.");
        exit(-1);
    }
    ctx->state = (int *)shm_ptr;
    ctx->address = (void *)((int *)shm_ptr + ctx->nstates);
}

inline void create_shm(ShmContext *ctx) {
    ctx->fp = memfd_create(ctx->name, MFD_CLOEXEC);

    if (ctx->fp == -1) {
        perror("shm open failed.");
        exit(-1);
    }
    const int total_size = ctx->nstates * sizeof(int) + ctx->nbytes;
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
    ctx->address = (void *)((int *)shm_ptr + ctx->nstates);
}

inline void close_shm(ShmContext *ctx) {
    const int total_size = ctx->nstates * sizeof(int) + ctx->nbytes;
    if (ctx->fp != -1) {
        munmap(ctx->address, total_size);
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
