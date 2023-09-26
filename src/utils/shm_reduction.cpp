#include <fcntl.h>
#include <immintrin.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "ccl_shm.h"
#include "shm_reduction.h"

ShmReduction::ShmReduction(BARRIER_CALLBACK callback, int rank, int size) : m_rank(rank) {
    if (rank == 0) {
        workspace = static_cast<struct allreduce_workspace *>(malloc(size * sizeof(struct allreduce_workspace)));
        if (workspace == nullptr) {
            printf("Error: workspace alloc failed\n");
            return;
        }

        shared_create(&allreduce_buffer, SHM_BUFFER_NAME, workspace, size * sizeof(struct allreduce_workspace));
        workspace = static_cast<struct allreduce_workspace *>(allreduce_buffer.bytes);
        memset(workspace, 0, size * sizeof(struct allreduce_workspace));
    }

    callback();

    if (rank != 0) { shared_open(&allreduce_buffer, SHM_BUFFER_NAME, size * sizeof(struct allreduce_workspace)); }

    workspace = static_cast<struct allreduce_workspace *>(allreduce_buffer.bytes);
}

void ShmReduction::shmClose() {
    shared_close(&allreduce_buffer);
    // TODO: fix the error when free the memory
    // if (m_rank == 0) {
    //     free(workspace);
    // }
}

int ShmReduction::getSHMSize() {
    return MAX_BUF_SIZE;
}

void shm_reduction(float *src, float *dst, int numel, int rank, int size) {
    int data_size = numel * sizeof(float);
    memcpy(workspace[rank].buffer, src, data_size);
    std::atomic_thread_fence(std::memory_order_release);
    workspace[rank].state = 1;

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            wait_buffer_state_until(i, 1);
        }
        reduce_all_buffers(workspace, numel, c10::ScalarType::Float, size);
        std::atomic_thread_fence(std::memory_order_release);
        workspace[rank].state = 2;
        memcpy(dst, workspace[0].buffer, data_size);

        for (int i = 1; i < size; i++) {
            wait_buffer_state_until(i, 2);
        }
        std::atomic_thread_fence(std::memory_order_release);
        workspace[rank].state = 0;
    }
    if (rank != 0) {
        wait_buffer_state_until(0, 2);
        memcpy(dst, workspace[0].buffer, data_size);
        std::atomic_thread_fence(std::memory_order_release);
        workspace[rank].state = 2;

        wait_buffer_state_until(0, 0);
        workspace[rank].state = 0;
    }
}
