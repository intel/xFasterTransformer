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
#include <time.h>

static inline void multiThreadCopy(char *dst, char *src, int nbytes, int unit) { 
#pragma omp parallel for
    for (int index = 0; index < nbytes; index += unit) {
        int size = (index + unit > nbytes) ? (nbytes - index) : unit;
        memcpy(dst + index, src + index, size);
    }
}

ShmReduction::ShmReduction(int rank, int size, std::function<void(int *, size_t)> callback)
    : rank_(rank), rank_size_(size) {

    snprintf(shmCtx_.name, sizeof(shmCtx_.name), "%s_rank_%d", SHM_NAME, rank);
    shmCtx_.rank_idx = rank;    
    shmCtx_.nrank = size;
    memset((void *)shmCtx_.shm_ptr, 0, MAX_RANK_NUMBER * sizeof(void *));

    // create local shared memory
    shmCtx_.pid_fd[0] = getpid();
    shmCtx_.pid_fd[1] = memfd_create(shmCtx_.name, MFD_CLOEXEC);
    if (shmCtx_.pid_fd[1] == -1) {
        perror("shm open failed.");
        exit(-1);
    }

    const int total_size = sizeof(int) * (MAX_RANK_NUMBER * 2  + 1) + MAX_SHM_SIZE;
    // Truncate the shared memory to the desired size
    if (ftruncate(shmCtx_.pid_fd[1], total_size) == -1) {
        perror("shm ftruncate failed.");
        exit(-1);
    }

    // Map the shared memory into the address space of the process
    shmCtx_.shm_ptr[shmCtx_.rank_idx] = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, shmCtx_.pid_fd[1], 0);
    if (shmCtx_.shm_ptr[shmCtx_.rank_idx] == MAP_FAILED) {
        perror("shm mmap failed.");
        exit(-1);
    }

    // update the offset addr in local shared memory
    shmCtx_.all_pids = (int *)shmCtx_.shm_ptr[shmCtx_.rank_idx];
    shmCtx_.state = shmCtx_.all_pids + 2 * MAX_RANK_NUMBER;
    shmCtx_.address = (void *)(shmCtx_.state + 1);

    // fill initial data
    shmCtx_.all_pids[2 * shmCtx_.rank_idx] = shmCtx_.pid_fd[0];
    shmCtx_.all_pids[2 * shmCtx_.rank_idx + 1] = shmCtx_.pid_fd[1];
    *(shmCtx_.state) = 0;

    // get rank 0 info
    callback(shmCtx_.all_pids, 2);

    if (rank != 0) {
        // connect to rank 0 shared memory
        char rank0_fd_path[64];
        snprintf(rank0_fd_path, sizeof(rank0_fd_path), "/proc/%d/fd/%d", shmCtx_.all_pids[0], shmCtx_.all_pids[1]);
        int rank0_fp = open(rank0_fd_path, O_RDWR);
        if (rank0_fp == -1) {
            perror("Bad rank 0 file descriptor.");
            exit(-1);
        }

        // Map the rank 0 shared memory into the address space of the process
        shmCtx_.shm_ptr[0] = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, rank0_fp, 0);
        if (shmCtx_.shm_ptr[0] == MAP_FAILED) {
            perror("rank 0 shm mmap failed.");
            exit(-1);
        }

        // update current process info into rank 0 shared memory
        *((int *)(shmCtx_.shm_ptr[0]) + 2 * shmCtx_.rank_idx) = shmCtx_.pid_fd[0];
        *((int *)(shmCtx_.shm_ptr[0]) + 2 * shmCtx_.rank_idx + 1) = shmCtx_.pid_fd[1];        
    }

    // connect to other rank share memory
    for (int i = 1; i < shmCtx_.nrank; i++) {
        if (i == rank) continue;                        // skip rank 0 and self rank
        if (NULL != shmCtx_.shm_ptr[i]) continue;       // already connected

        // connect to shared memory with info from rank0
        char rank_fd_path[64];
        snprintf(rank_fd_path, sizeof(rank_fd_path), 
                "/proc/%d/fd/%d", 
                *((int *)(shmCtx_.shm_ptr[0]) + 2 * i), 
                *((int *)(shmCtx_.shm_ptr[0]) + 2 * i + 1));

        // check share memory readiness
        while (access(rank_fd_path, F_OK) != 0) 
        {
            //sched_yield();
            snprintf(rank_fd_path, sizeof(rank_fd_path), 
                    "/proc/%d/fd/%d", 
                    *((int *)(shmCtx_.shm_ptr[0]) + 2 * i), 
                    *((int *)(shmCtx_.shm_ptr[0]) + 2 * i + 1));
        }
        
        int rank_fp = open(rank_fd_path, O_RDWR);
        if (rank_fp == -1) {
            perror("Bad rank file descriptor.");
            exit(-1);
        }

        // Map the other rank shared memory into the address space of this process
        const int total_size = sizeof(int) * (MAX_RANK_NUMBER * 2  + 1) + MAX_SHM_SIZE;
        shmCtx_.shm_ptr[i] = mmap(NULL, total_size, PROT_READ | PROT_WRITE, MAP_SHARED, rank_fp, 0);
        if (shmCtx_.shm_ptr[i] == MAP_FAILED) {
            perror("rank shm mmap failed.");
            exit(-1);
        }
    }
}

int ShmReduction::getSHMSize() {
    return MAX_SHM_SIZE;
}

template <typename T>
void ShmReduction::reduceAdd(T *sendBuf, T *recvBuf, size_t size, int rank, int rankSize) {
    // prepare the data in shared memory
    xft::wait_state_until(&shmCtx_, rank, 0);
    *(shmCtx_.state) = 1;
    
    // split the data buffer into segments according to rank number
    T *address = (T *)shmCtx_.address;
    T *rank_address[MAX_RANK_NUMBER] = {NULL};
    int *rank_state[MAX_RANK_NUMBER] = {NULL};
    for (int i = 0; i < shmCtx_.nrank; i++) {
        rank_state[i] = (int*)shmCtx_.shm_ptr[i] + 2 * MAX_RANK_NUMBER;
        rank_address[i] = (T *)((int*)shmCtx_.shm_ptr[i] + 2 * MAX_RANK_NUMBER + 1);
    }

    int rank_offset = size / shmCtx_.nrank * rank;
    int block_count = size / shmCtx_.nrank;
    if (shmCtx_.nrank - 1 == rank)  block_count = size - rank_offset;
    int block_size = (block_count + 16*omp_get_max_threads() - 1)/(omp_get_max_threads() * 16) * 16;

    multiThreadCopy((char *)address, (char *)sendBuf, size * sizeof(T), 1024);
    *(shmCtx_.state) = 2;
    
#pragma omp parallel for //num_threads(nthreads)
    for (int blockIndex = 0; blockIndex < block_count; blockIndex += block_size) {
        int rank_status[MAX_RANK_NUMBER] = {0};
        int rank_readiness = 1;
        rank_status[rank] = 1;

        int real_block_size
                = (blockIndex+block_size > block_count ? (block_count -  blockIndex) : (block_size));
        int block_offset = rank_offset + blockIndex;

        T *lAddrBuf = address + block_offset;
        __m512 in1_val, inout_val;
        for (int rank_index = 0; (rank_index < shmCtx_.nrank) && (rank_readiness < shmCtx_.nrank); rank_index = (rank_index+1) % shmCtx_.nrank) {
            if (rank_index == rank) continue;
            if (1 == rank_status[rank_index]) continue;

            if (2 <= *(rank_state[rank_index]))
            {
                T *rankAddrBuf = rank_address[rank_index] +  block_offset;
                for (int index = 0; index < real_block_size; index += 16) {
                    int remain = real_block_size - index;
                    __mmask16 mask = (remain >= 16 ? 0xffff : (1 << remain) - 1);
                    inout_val = xft::load_avx512(mask, lAddrBuf + index);
                    in1_val = xft::load_avx512(mask, rankAddrBuf + index);
                    inout_val = _mm512_add_ps(inout_val, in1_val);
                    xft::store_avx512(lAddrBuf + index, mask, inout_val);
                }
                rank_status[rank_index] = 1;
                rank_readiness++;
            }
        }
        for (int rank_index = 0; rank_index < shmCtx_.nrank; rank_index++) {
            if (rank_index == rank) continue;
            T *rankAddrBuf = rank_address[rank_index] +  block_offset;
            multiThreadCopy((char *)rankAddrBuf, (char *)lAddrBuf, real_block_size * sizeof(T), 1024);
        }
    }
   
    if (0 != rank) {
        *(shmCtx_.state) = 3;
        xft::wait_state_until(&shmCtx_, 0, 3);
    } else {
        // wait for all ranks to be ready
        for (int i = 1; i < shmCtx_.nrank; i++) xft::wait_state_until(&shmCtx_, i, 3);
        *(shmCtx_.state) = 3;
    }
    
    multiThreadCopy((char *)recvBuf, (char *)address, size * sizeof(T), 4096);
    
    if (0 != rank) {
        *(shmCtx_.state) = 4;
    } else {
        // wait for all ranks to be ready
        for (int i = 1; i < shmCtx_.nrank; i++) xft::wait_state_until(&shmCtx_, i, 4);
        for (int i = 0; i < shmCtx_.nrank; i++) *((int*)(shmCtx_.shm_ptr[i]) + MAX_RANK_NUMBER * 2) = 0;
    }
}

template void ShmReduction::reduceAdd<float>(float *sendBuf, float *recvBuf, size_t size, int rank, int rankSize);
template void ShmReduction::reduceAdd<bfloat16_t>(bfloat16_t *sendBuf, bfloat16_t *recvBuf, size_t size, int rank, int rankSize);