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
#include <iostream>

#include "compile_util.h"
#include "oneapi/ccl.hpp"
#include "shm_reduction.h"
#include "timeline.h"

class Messenger {
private:
    Messenger() {
        // User has set the SINGLE_INSTANCE environment variable
        if (std::getenv("SINGLE_INSTANCE") != nullptr) {
            this->pcomm = nullptr;
#ifdef USE_SHM
            this->pshm = nullptr;
#endif
            this->rank = 0;
            this->size = 1;
            return;
        }

        ccl::init();

        MPI_Init(NULL, NULL);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        atexit(Messenger::mpi_finalize);

        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            main_addr = kvs->get_address();
            MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        } else {
            MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
            kvs = ccl::create_kvs(main_addr);
        }

        pcomm = new ccl::communicator(ccl::create_communicator(size, rank, kvs));

        rank = pcomm->rank();
        size = pcomm->size();

#ifdef USE_SHM
        char my_hostname[MPI_MAX_PROCESSOR_NAME];
        char all_hostnames[MPI_MAX_PROCESSOR_NAME * MPI_MAX_PROCESSOR_NAME];
        int hostname_len;

        // Check ranks are on the same physical machine
        MPI_Get_processor_name(my_hostname, &hostname_len);
        MPI_Allgather(my_hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                MPI_COMM_WORLD);

        int same_hostnames = 1;
        for (int i = 1; i < size; i++) {
            if (strcmp(my_hostname, &all_hostnames[i * MPI_MAX_PROCESSOR_NAME]) != 0) {
                same_hostnames = 0;
                break;
            }
        }

        if (same_hostnames && !std::getenv("XFT_ONECCL")) {
            local_ranks_flag = true;
            pshm = new ShmReduction(rank, size, [this](int *pid_fd, size_t count) { this->broadcast(pid_fd, count); });
        } else {
            local_ranks_flag = false;
        }
#endif
    }

    ~Messenger() {
        delete pcomm;
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
    template <typename T>
    void reduceAdd(T *sendBuf, T *recvBuf, size_t count) {
        TimeLine t("Messenger.reduceAdd");

#ifdef USE_SHM
        if (count * sizeof(T) > pshm->getSHMSize() || !local_ranks_flag) {
            ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *pcomm).wait();
        } else {
            pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
        }
#else
        ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *pcomm).wait();
#endif
    }

    template <typename T>
    void broadcast(T *buf, size_t count) {
        if (check()) {
            ccl::broadcast(buf, count, 0, *pcomm).wait(); // assume always broadcast from master (rank 0)
        }
    }

    template <typename T>
    void alltoall(const T *send_buf, T *recv_buf, size_t count) {
        if (check()) { ccl::alltoall(send_buf, recv_buf, count, *pcomm).wait(); }
    }

    void barrier() {
        if (check()) { ccl::barrier(*pcomm); }
    }

    template <typename T>
    void allgatherv(const T *send_buf, size_t count, T *recv_buf, const std::vector<long unsigned int> &recv_counts) {
        if (check()) { ccl::allgatherv(send_buf, count, recv_buf, recv_counts, *pcomm).wait(); }
    }

private:
    Messenger(const Messenger &messenger) = delete;
    Messenger &operator=(const Messenger &messenger) = delete;

    static void mpi_finalize() {
        int is_finalized = 0;
        MPI_Finalized(&is_finalized);

        if (!is_finalized) { MPI_Finalize(); }
    }

    // Check if indeed need to communicate
    bool check() {
        if (unlikely(size > 1 && !pcomm)) {
            printf("Unable to call into ccl as of unsuccessful initialization.\n");
            exit(-1);
        }
        return size > 1;
    }

private:
    int size;
    int rank;
    bool local_ranks_flag;

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;

    ccl::communicator *pcomm;

#ifdef USE_SHM
    ShmReduction *pshm;
#endif
};
