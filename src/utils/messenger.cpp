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
#include <mpi.h>

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>

#include "bfloat16.h"
#include "compile_util.h"
#include "messenger.h"
#include "oneapi/ccl.hpp"
#include "shm_reduction.h"
#include "timeline.h"
#include "verbose.h"

Messenger::Messenger() {
    // User has set the SINGLE_INSTANCE environment variable
    // or program is not with MPI.
    if (std::getenv("SINGLE_INSTANCE") != nullptr || !withMpirun()) {
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

    color = Env::getPipelineStage();
    int sameHostnames = (*helperInit)(&size, &rank, &color);

#ifdef USE_SHM
    if (sameHostnames && !std::getenv("XFT_ONECCL")) {
        localRanksFlag = true;
        pshm = new ShmReduction(rank, size, [this](int *pidFd, size_t count) { this->broadcast(pidFd, count); });
    } else {
        localRanksFlag = false;
    }
#endif
}

Messenger::~Messenger() {
    if (helperFreePCOMM != nullptr) { (*helperFreePCOMM)(); }
    // if (commHelperHanlde != nullptr) { dlclose(commHelperHanlde); }
#ifdef USE_SHM
    delete pshm;
#endif
}

Messenger &Messenger::getInstance() {
    static Messenger instance;
    return instance;
}

bool Messenger::isMaster() {
    return rank == 0;
}

int Messenger::getRank() {
    return rank;
}

int Messenger::getSize() {
    return size;
}

int Messenger::getColor() {
    return color;
}

// From some example code of oneCCL, inplace reducing is supported
// Only float is used now
void Messenger::reduceAdd(float *sendBuf, float *recvBuf, size_t count) {
    TimeLine t("Messenger.reduceAdd");

#ifdef USE_SHM
    if (count * sizeof(float) > pshm->getSHMSize() || !localRanksFlag) {
        (*helperAllreduce)(sendBuf, recvBuf, count);
    } else {
        pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
    }
#else
    (*helperAllreduce)(sendBuf, recvBuf, count);
#endif
}

void Messenger::reduceAdd(bfloat16_t *sendBuf, bfloat16_t *recvBuf, size_t count) {
    TimeLine t("Messenger.reduceAdd");

#ifdef USE_SHM
    if (count * sizeof(bfloat16_t) > pshm->getSHMSize() || !localRanksFlag) {
        (*helperAllreduceBF16)(sendBuf, recvBuf, count);
    } else {
        pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
    }
#else
    (*helperAllreduceBF16)(sendBuf, recvBuf, count);
#endif
}

// Only float is used now
void Messenger::allgatherv(
        const float *send_buf, size_t count, float *recv_buf, const std::vector<long unsigned int> &recv_counts) {
    if (check()) { (*helperAllgatherv)(send_buf, count, recv_buf, recv_counts); }
}

void Messenger::worldSendFP32(const float *buf, int count, int dest, int tag) {
    if (check()) { (*helperWorldSendFP32)(buf, count, dest, tag); }
}

void Messenger::worldRecvFP32(float *buf, int count, int source, int tag) {
    if (check()) { (*helperWorldRecvFP32)(buf, count, source, tag); }
}

void Messenger::worldSendINT32(const int32_t *buf, int count, int dest, int tag) {
    if (check()) { (*helperWorldSendINT32)(buf, count, dest, tag); }
}

void Messenger::worldRecvINT32(int32_t *buf, int count, int source, int tag) {
    if (check()) { (*helperWorldRecvINT32)(buf, count, source, tag); }
}

bool Messenger::withMpirun() {
    return (std::getenv("MPI_LOCALRANKID") || std::getenv("MPI_LOCALNRANKS") || std::getenv("PMI_RANK")
                   || std::getenv("PMI_SIZE") || std::getenv("PMIX_RANK"))
            ? true
            : false;
}

void Messenger::mpi_finalize() {
    void *handle = dlopen("libxft_comm_helper.so", RTLD_NOW | RTLD_LOCAL);
    if (handle != nullptr) {
        void (*helperMpiFinalize)() = (void (*)())dlsym(handle, "mpiFinalize");
        (*helperMpiFinalize)();
        dlclose(handle);
    }
}

// Check if indeed need to communicate
bool Messenger::check() {
    if (unlikely(size > 1 && !commHelperHanlde)) {
        printf("Unable to call into ccl as of unsuccessful initialization.\n");
        exit(-1);
    }
    return size > 1;
}
