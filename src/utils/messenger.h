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
#include <mpi.h>

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <sys/time.h>

#include "bfloat16.h"
#include "compile_util.h"
#include "environment.h"
#include "oneapi/ccl.hpp"
#include "shm_reduction.h"
#include "simple_mem_pool.h"
#include "timeline.h"
#include "verbose.h"

extern bool tunedComm();

class Messenger {
private:
    Messenger() {
        // User has set the SINGLE_INSTANCE environment variable
        // or program is not with MPI.
        if (Env::getInstance().getSingleInstance() || !withMpirun()) {
            std::cout << "[INFO] SINGLE_INSTANCE MODE." << std::endl;
#ifdef USE_SHM
            this->pshm = nullptr;
#endif
            this->rank = 0;
            this->size = 1;
            return;
        }

        commHelperHanlde = dlopen("libxft_comm_helper.so", RTLD_NOW | RTLD_LOCAL);
        if (commHelperHanlde == nullptr) {
            printf("Failed to load xft_comm_helper library from path error code: %s\n", dlerror());
            exit(-1);
        }

        helperInit = (int (*)(int *, int *, int *))dlsym(commHelperHanlde, "init");
        helperFreePCOMM = (void (*)())dlsym(commHelperHanlde, "freePCOMM");
        helperAllreduce = (void (*)(float *, float *, size_t))dlsym(commHelperHanlde, "allreduce");
        helperAllreduceBF16 = (void (*)(bfloat16_t *, bfloat16_t *, size_t))dlsym(commHelperHanlde, "allreduceBF16");
        helperBroadcast = (void (*)(int *, size_t))dlsym(commHelperHanlde, "broadcast");
        helperAllgatherv = (void (*)(const float *, size_t, float *, const std::vector<long unsigned int> &))dlsym(
                commHelperHanlde, "allgatherv");

        helperWorldSendFP32 = (void (*)(const float *, int, int, int))dlsym(commHelperHanlde, "worldSendFP32");
        helperWorldRecvFP32 = (void (*)(float *, int, int, int))dlsym(commHelperHanlde, "worldRecvFP32");
        helperWorldSendINT32 = (void (*)(const int32_t *, int, int, int))dlsym(commHelperHanlde, "worldSendINT32");
        helperWorldRecvINT32 = (void (*)(int32_t *, int, int, int))dlsym(commHelperHanlde, "worldRecvINT32");

        atexit(Messenger::mpi_finalize);

        color = Env::getInstance().getPipelineStage();
        int sameHostnames = (*helperInit)(&size, &rank, &color);

#ifdef USE_SHM
        if (sameHostnames && !Env::getInstance().getOneCCLEnabled()) {
            localRanksFlag = true;
            pshm = new ShmReduction(rank, size, [this](int *pidFd, size_t count) { this->broadcast(pidFd, count); });
        } else {
            localRanksFlag = false;
        }
#endif
    }

    ~Messenger() {
        if (helperFreePCOMM != nullptr) { (*helperFreePCOMM)(); }
        // if (commHelperHanlde != nullptr) { dlclose(commHelperHanlde); }
#ifdef USE_SHM
        delete pshm;
#endif
    }

public:
    static Messenger &getInstance() {
        static Messenger instance;
        return instance;
    }

    bool isMaster() { return rank == 0 && color == 0; }

    int getRank() { return rank; }

    int getSize() { return size; }

    int getColor() { return color; }

    template <typename T>
    void reduceAdd(T *sendBuf, T *recvBuf, size_t count) {
        if (!check()) return;
        TimeLine t("Messenger.reduceAdd");
        static std::unordered_map<size_t, int> tuned_map;
#ifdef USE_SHM
        if (tunedComm() && localRanksFlag) {
            size_t commSize = sizeof(T) * count;
            if (sizeof(T) * count > pshm->getSHMSize()) { pshm->ShmResize(rank, commSize); }

            auto it = tuned_map.find(commSize);
            if (it == tuned_map.end()) {
                T *commBuf = (T *)SimpleMemPool::instance().getBuffer("commBuf", commSize);
                int warmup = 1, nruns = 3;
                struct timeval start, end;
                float dur_shm = std::numeric_limits<float>::max(), dur_ccl = std::numeric_limits<float>::max();
                // tuned for the faster comm primitive
                for (int i = 0; i < warmup + nruns; ++i) {
                    if (i >= warmup) gettimeofday(&start, NULL);
                    pshm->reduceAdd(commBuf, commBuf, count, rank, size);
                }
                gettimeofday(&end, NULL);
                dur_shm = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;

                if (commSize < 1.0e9) {
                    for (int i = 0; i < warmup + nruns; ++i) {
                        if (i >= warmup) gettimeofday(&start, NULL);
                        cclAllreduce(commBuf, commBuf, count);
                    }
                    gettimeofday(&end, NULL);
                    dur_ccl = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
                }

                // default 0 means SHM Addreduce, 1 means CCL
                int comm_type = 0;
                if (rank == 0 && dur_ccl < dur_shm * 0.9) { comm_type = 1; }
                this->broadcast(&comm_type, 1);
                tuned_map[commSize] = comm_type;
            }

            if (tuned_map[commSize] == 0) {
                pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
            } else {
                cclAllreduce(sendBuf, recvBuf, count);
            }
        } else {
            reduceAddBase(sendBuf, recvBuf, count);
        }
#else
        reduceAddBase(sendBuf, recvBuf, count);
#endif
    }

    // inplace reducing is supported
    template <typename T>
    void reduceAddBase(T *sendBuf, T *recvBuf, size_t count) {
        TimeLine t("Messenger.reduceAdd");

#ifdef USE_SHM
        if (sizeof(T) * count > pshm->getSHMSize() || !localRanksFlag) {
            cclAllreduce(sendBuf, recvBuf, count);
        } else {
            pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
        }
#else
        cclAllreduce(sendBuf, recvBuf, count);
#endif
    }

    template <typename T>
    void cclAllreduce(T *sendBuf, T *recvBuf, size_t count) {
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            (*helperAllreduceBF16)(sendBuf, recvBuf, count);
        } else if constexpr (std::is_same_v<T, float>) {
            (*helperAllreduce)(sendBuf, recvBuf, count);
        } else {
            printf("Unsupported data type for reduceAdd.\n");
            exit(-1);
        }
    }

    // Only int is used now
    template <typename T>
    void broadcast(T *buf, size_t count) {
        if (check()) {
            // assume always broadcast from master (rank 0)
            (*helperBroadcast)(buf, count);
        }
    }

    // template <typename T>
    // void alltoall(const T *send_buf, T *recv_buf, size_t count) {
    //     if (check()) { ccl::alltoall(send_buf, recv_buf, count, *pcomm).wait(); }
    // }

    // void barrier() {
    //     if (check()) { ccl::barrier(*pcomm); }
    // }

    // Only float is used now
    void allgatherv(
            const float *send_buf, size_t count, float *recv_buf, const std::vector<long unsigned int> &recv_counts) {
        if (check()) { (*helperAllgatherv)(send_buf, count, recv_buf, recv_counts); }
    }

    void worldSendFP32(const float *buf, int count, int dest, int tag) {
        if (check()) { (*helperWorldSendFP32)(buf, count, dest, tag); }
    }

    void worldRecvFP32(float *buf, int count, int source, int tag) {
        if (check()) { (*helperWorldRecvFP32)(buf, count, source, tag); }
    }

    void worldSendINT32(const int32_t *buf, int count, int dest, int tag) {
        if (check()) { (*helperWorldSendINT32)(buf, count, dest, tag); }
    }

    void worldRecvINT32(int32_t *buf, int count, int source, int tag) {
        if (check()) { (*helperWorldRecvINT32)(buf, count, source, tag); }
    }

    bool withMpirun() {
        return (std::getenv("MPI_LOCALRANKID") || std::getenv("MPI_LOCALNRANKS") || std::getenv("PMI_RANK")
                       || std::getenv("PMI_SIZE") || std::getenv("PMIX_RANK"))
                ? true
                : false;
    }

private:
    Messenger(const Messenger &messenger) = delete;
    Messenger &operator=(const Messenger &messenger) = delete;

    static void mpi_finalize() {
        void *handle = dlopen("libxft_comm_helper.so", RTLD_NOW | RTLD_LOCAL);
        if (handle != nullptr) {
            void (*helperMpiFinalize)() = (void (*)())dlsym(handle, "mpiFinalize");
            (*helperMpiFinalize)();
            dlclose(handle);
        }
    }

    // Check if indeed need to communicate
    bool check() {
        if (unlikely(size > 1 && !commHelperHanlde)) {
            printf("Unable to call into ccl as of unsuccessful initialization.\n");
            exit(-1);
        }
        return size > 1;
    }

private:
    int size;
    int rank;
    int color; // Processes with the same color will be placed into the same sub-communicator
    bool localRanksFlag;

#ifdef USE_SHM
    ShmReduction *pshm;
#endif
    void *commHelperHanlde;
    int (*helperInit)(int *, int *, int *);
    void (*helperFreePCOMM)();
    void (*helperAllreduce)(float *, float *, size_t);
    void (*helperAllreduceBF16)(bfloat16_t *, bfloat16_t *, size_t);
    void (*helperBroadcast)(int *, size_t);
    void (*helperAllgatherv)(const float *, size_t, float *, const std::vector<long unsigned int> &);
    void (*helperWorldSendFP32)(const float *, int, int, int);
    void (*helperWorldRecvFP32)(float *, int, int, int);
    void (*helperWorldSendINT32)(const int32_t *, int, int, int);
    void (*helperWorldRecvINT32)(int32_t *, int, int, int);
};
