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

#include <vector>
#include "bfloat16.h"
#include "shm_reduction.h"

class Messenger {
public:
    static Messenger& getInstance();

    bool isMaster();
    int getRank();
    int getSize();
    int getColor();

    void reduceAdd(float* sendBuf, float* recvBuf, size_t count);
    void reduceAdd(bfloat16_t* sendBuf, bfloat16_t* recvBuf, size_t count);
    template <typename T>
    void broadcast(T *buf, size_t count) {
        if (check()) {
            // assume always broadcast from master (rank 0)
            (*helperBroadcast)(buf, count);
        }
    }
    void allgatherv(const float* send_buf, size_t count, float* recv_buf, const std::vector<long unsigned int>& recv_counts);
    void worldSendFP32(const float* buf, int count, int dest, int tag);
    void worldRecvFP32(float* buf, int count, int source, int tag);
    void worldSendINT32(const int32_t* buf, int count, int dest, int tag);
    void worldRecvINT32(int32_t* buf, int count, int source, int tag);

private:
    Messenger();
    ~Messenger();

    Messenger(const Messenger& messenger) = delete;
    Messenger& operator=(const Messenger& messenger) = delete;

    bool withMpirun();
    static void mpi_finalize();
    bool check();

private:
    int size;
    int rank;
    int color;
    bool localRanksFlag;

#ifdef USE_SHM
    ShmReduction* pshm;
#endif
    void* commHelperHanlde;
    int (*helperInit)(int*, int*, int*);
    void (*helperFreePCOMM)();
    void (*helperAllreduce)(float*, float*, size_t);
    void (*helperAllreduceBF16)(bfloat16_t*, bfloat16_t*, size_t);
    void (*helperBroadcast)(int*, size_t);
    void (*helperAllgatherv)(const float*, size_t, float*, const std::vector<long unsigned int>&);
    void (*helperWorldSendFP32)(const float*, int, int, int);
    void (*helperWorldRecvFP32)(float*, int, int, int);
    void (*helperWorldSendINT32)(const int32_t*, int, int, int);
    void (*helperWorldRecvINT32)(int32_t*, int, int, int);
};