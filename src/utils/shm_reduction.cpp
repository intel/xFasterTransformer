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
#include "shm_reduction.h"
#include "intrinsics_util.h"

static inline void multiThreadCopy(char *dst, char *src, size_t nbytes) {
    constexpr int sizePerSplit = 1024;
    int splits = (nbytes + sizePerSplit - 1) / sizePerSplit;

#pragma omp parallel for
    for (uint64_t i = 0; i < splits; ++i) {
        size_t size = (i == splits - 1) ? (nbytes - i * sizePerSplit) : sizePerSplit;
        memcpy(dst + i * sizePerSplit, src + i * sizePerSplit, size);
    }
}

ShmReduction::ShmReduction(int rank, size_t size, std::function<void(int *, size_t)> callback)
    : rank_(rank), rank_size_(size) {
    shmCtx_.name = SHM_NAME;
    shmCtx_.nstates = size;
    shmCtx_.nbytes = MAX_SHM_SIZE;
    shmCtx_.nblocks = MAX_SHM_BLOCK_COUNT;
    if (rank_ == 0) {
        xft::create_shm(&shmCtx_);
        memset(shmCtx_.state, 0, sizeof(int) * shmCtx_.nstates);
        memset((void *)shmCtx_.blockState, 0, shmCtx_.nstates * shmCtx_.nblocks);
    }

    callback(shmCtx_.pid_fd, 2);

    if (rank != 0) { xft::connect_shm(&shmCtx_); }
}

void ShmReduction::ShmResize(int rank, size_t size) {
    // remove old shm
    size_t total_size = sizeof(int) * shmCtx_.nstates + shmCtx_.nbytes + shmCtx_.nblocks * shmCtx_.nstates;
    munmap(shmCtx_.address, total_size);
    // shm_unlink(shmCtx_.name);

    // alloc and map new shm
    total_size = total_size - shmCtx_.nbytes + size;
    shmCtx_.nbytes = size;
    // Truncate the shared memory to the desired size
    if (rank == 0 && ftruncate(shmCtx_.fp, total_size) == -1) {
        perror("shm ftruncate failed.");
        exit(-1);
    }

    // Map the shared memory into the address space of the process
    void *shm_ptr = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, shmCtx_.fp, 0);
    if (shm_ptr == MAP_FAILED) {
        perror("shm mmap failed.");
        exit(-1);
    }
    shmCtx_.state = (int *)shm_ptr;
    shmCtx_.blockState = (uint8_t *)((int *)shm_ptr + shmCtx_.nstates);
    shmCtx_.address = (void *)((uint8_t *)shmCtx_.blockState + shmCtx_.nblocks * shmCtx_.nstates);

    if (rank == 0) {
        memset(shmCtx_.state, 0, sizeof(int) * shmCtx_.nstates);
        memset((void *)shmCtx_.blockState, 0, shmCtx_.nstates * shmCtx_.nblocks);
    }
}

size_t ShmReduction::getSHMSize() {
    return shmCtx_.nbytes;
}

template <typename T>
void ShmReduction::reduceAdd(T *sendBuf, T *recvBuf, size_t size, int rank, int rankSize) {
    size_t nbytes = sizeof(T) * size;
    size_t nBlockBytes = sizeof(T) * SHM_BLOCK_SIZE;
    int nblocks = (size + SHM_BLOCK_SIZE - 1) / SHM_BLOCK_SIZE;
    int nthreads = std::min(nblocks, omp_get_max_threads());

    T *address = (T *)shmCtx_.address;
    uint8_t *blocks = (uint8_t *)shmCtx_.blockState;
    int *states = shmCtx_.state;

    if (rank == 0) {
        for (int i = 1; i < rankSize; i++) {
            xft::wait_state_until(&shmCtx_, i, 0);
        }
        multiThreadCopy((char *)address, (char *)sendBuf, nbytes);
    } else {
        xft::wait_state_until(&shmCtx_, rank, 0);
        xft::wait_state_until(&shmCtx_, 0, 1);
    }
    shmCtx_.state[rank] = 1;

    if (rank != 0) {
#pragma omp parallel for num_threads(nthreads)
        for (int blockIndex = 0; blockIndex < nblocks; blockIndex++) {

            T *lSendBuf = sendBuf + SHM_BLOCK_SIZE * blockIndex;
            T *lAddrBuf = address + SHM_BLOCK_SIZE * blockIndex;
            int realBlockSize
                    = (blockIndex == (nblocks - 1) ? (size - SHM_BLOCK_SIZE * (nblocks - 1)) : SHM_BLOCK_SIZE);

            if (rank != 1) { xft::wait_block_until(&shmCtx_, blockIndex * rankSize + rank - 1, 1); }

            __m512 in1_val, inout_val;
            for (int index = 0; index < realBlockSize; index += 16) {
                int remain = realBlockSize - index;
                __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                in1_val = xft::load_avx512(mask, lSendBuf + index);
                inout_val = xft::load_avx512(mask, lAddrBuf + index);
                inout_val = _mm512_add_ps(inout_val, in1_val);
                xft::store_avx512(lAddrBuf + index, mask, inout_val);
            }
            shmCtx_.blockState[blockIndex * rankSize + rank - 1] = 0;
            shmCtx_.blockState[blockIndex * rankSize + rank] = 1;
        }
        shmCtx_.state[rank] = 2;
    }

    xft::wait_state_until(&shmCtx_, rankSize - 1, 2);

    multiThreadCopy((char *)recvBuf, (char *)address, nbytes);

    if (rank == rankSize - 1) {
        for (int i = 0; i < rankSize - 1; i++) {
            xft::wait_state_until(&shmCtx_, i, 3);
        }

        for (int i = 0; i < rankSize; i++) {
            shmCtx_.state[i] = 0;
        }
    } else {
        shmCtx_.state[rank] = 3;
    }
}

template void ShmReduction::reduceAdd<float>(float *sendBuf, float *recvBuf, size_t size, int rank, int rankSize);
template void ShmReduction::reduceAdd<bfloat16_t>(
        bfloat16_t *sendBuf, bfloat16_t *recvBuf, size_t size, int rank, int rankSize);
