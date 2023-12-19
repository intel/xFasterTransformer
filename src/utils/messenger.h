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

#include "compile_util.h"
#include "oneapi/ccl.hpp"
#include "shm_reduction.h"
#include "timeline.h"

class Messenger {
private:
    Messenger() {
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

        comm_helper_hanlde = dlopen("libxft_comm_helper.so", RTLD_NOW | RTLD_LOCAL);
        if (comm_helper_hanlde == nullptr) {
            printf("Failed to load xft_comm_helper library from path error code: %s\n", dlerror());
            exit(-1);
        }

        helper_init = (int (*)(int *, int *))dlsym(comm_helper_hanlde, "init");
        helper_freePCOMM = (void (*)())dlsym(comm_helper_hanlde, "freePCOMM");
        helper_allreduce = (void (*)(float *, float *, size_t))dlsym(comm_helper_hanlde, "allreduce");
        helper_broadcast = (void (*)(int *, size_t))dlsym(comm_helper_hanlde, "broadcast");
        helper_allgatherv = (void (*)(const float *, size_t, float *, const std::vector<long unsigned int> &))dlsym(
                comm_helper_hanlde, "allgatherv");
        void (*helper_mpi_finalize)() = (void (*)())dlsym(comm_helper_hanlde, "mpiFinalize");

        atexit(helper_mpi_finalize);

        int same_hostnames = (*helper_init)(&rank, &size);

#ifdef USE_SHM
        if (same_hostnames && !std::getenv("XFT_ONECCL")) {
            local_ranks_flag = true;
            pshm = new ShmReduction(rank, size, [this](int *pid_fd, size_t count) { this->broadcast(pid_fd, count); });
        } else {
            local_ranks_flag = false;
        }
#endif
    }

    ~Messenger() {
        if (helper_freePCOMM != nullptr) { (*helper_freePCOMM)(); }
        if (comm_helper_hanlde != nullptr) { dlclose(comm_helper_hanlde); }
#ifdef USE_SHM
        delete pshm;
#endif
    }

public:
    static Messenger &getInstance() {
        static Messenger instance;
        return instance;
    }

    bool isMaster() { return rank == 0; }

    int getRank() { return rank; }

    int getSize() { return size; }

    // From some example code of oneCCL, inplace reducing is supported
    // Only float is used now
    void reduceAdd(float *sendBuf, float *recvBuf, size_t count) {
        TimeLine t("Messenger.reduceAdd");

#ifdef USE_SHM
        if (count * sizeof(float) > pshm->getSHMSize() || !local_ranks_flag) {
            (*helper_allreduce)(sendBuf, recvBuf, count);
        } else {
            pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
        }
#else
        (*helper_allreduce)(sendBuf, recvBuf, count);
#endif
    }

    // Only int is used now
    template <typename T>
    void broadcast(T *buf, size_t count) {
        if (check()) {
            // assume always broadcast from master (rank 0)
            (*helper_broadcast)(buf, count);
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
        if (check()) { (*helper_allgatherv)(send_buf, count, recv_buf, recv_counts); }
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

    // Check if indeed need to communicate
    bool check() {
        if (unlikely(size > 1 && !comm_helper_hanlde)) {
            printf("Unable to call into ccl as of unsuccessful initialization.\n");
            exit(-1);
        }
        return size > 1;
    }

private:
    int size;
    int rank;
    bool local_ranks_flag;

#ifdef USE_SHM
    ShmReduction *pshm;
#endif
    void *comm_helper_hanlde;
    int (*helper_init)(int *, int *);
    void (*helper_freePCOMM)();
    void (*helper_allreduce)(float *, float *, size_t);
    void (*helper_broadcast)(int *, size_t);
    void (*helper_allgatherv)(const float *, size_t, float *, const std::vector<long unsigned int> &);
};
