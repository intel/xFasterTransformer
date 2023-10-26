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

static inline void multiThreadCopy(char *dst, char *src, int nbytes) {
    constexpr int sizePerSplit = 1024;
    int splits = (nbytes + sizePerSplit - 1) / sizePerSplit;

#pragma omp parallel for
    for (int i = 0; i < splits; ++i) {
        int size = (i == splits - 1) ? (nbytes - i * sizePerSplit) : sizePerSplit;
        memcpy(dst + i * sizePerSplit, src + i * sizePerSplit, size);
    }
}

ShmReduction::ShmReduction(int rank, int size, std::function<void(int *, size_t)> callback)
    : rank_(rank), rank_size_(size) {
    shmCtx_.name = SHM_NAME;
    shmCtx_.nstates = size;
    shmCtx_.nbytes = MAX_SHM_SIZE;
    if (rank_ == 0) {
        xft::create_shm(&shmCtx_);
        memset(shmCtx_.state, 0, shmCtx_.nstates * sizeof(int));
    }

    callback(shmCtx_.pid_fd, 2);

    if (rank != 0) { xft::connect_shm(&shmCtx_); }
}

int ShmReduction::getSHMSize() {
    return MAX_SHM_SIZE;
}

template <typename T>
void ShmReduction::reduceAdd(T *sendBuf, T *recvBuf, size_t count, int rank, int rankSize) {
    printf("Type %s not supported!\n", typeid(T).name());
    exit(-1);
}

template <>
void ShmReduction::reduceAdd(float *sendBuf, float *recvBuf, size_t size, int rank, int rankSize) {
    int nbytes = size * sizeof(float);
    int nblocks = size / SHM_BLOCK_SIZE > 0 ? size / SHM_BLOCK_SIZE : 1;
    int nthreads = std::min(nblocks, omp_get_max_threads());

    float *address = (float *)shmCtx_.address;
    if (rank == 0) {
        for (int i = 1; i < rankSize; i++) {
            xft::wait_state_until(&shmCtx_, i, 0);
        }
        multiThreadCopy((char *)address, (char *)sendBuf, nbytes);
    } else {
        xft::wait_state_until(&shmCtx_, rank, 1);

#pragma omp parallel for num_threads(nthreads)
        for (int index = 0; index < size; index += 16) {
            int remain = size - index;
            __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
            __m512 inout_val = _mm512_maskz_loadu_ps(mask, address + index);
            auto in1_val = _mm512_maskz_loadu_ps(mask, sendBuf + index);
            inout_val = _mm512_add_ps(inout_val, in1_val);
            _mm512_mask_storeu_ps(address + index, mask, inout_val);
        }
    }

    if (rank != rankSize - 1) { shmCtx_.state[rank + 1] = 1; }
    if (rank == rankSize - 1) {
        for (int i = 0; i < rankSize; i++) {
            shmCtx_.state[i] = 2;
        }
    }

    xft::wait_state_until(&shmCtx_, rank, 2);
    multiThreadCopy((char *)recvBuf, (char *)address, nbytes);
    shmCtx_.state[rank] = 0;
}
