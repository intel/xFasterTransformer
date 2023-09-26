#pragma once
#include <mpi.h>

#include <cstdlib>
#include <iostream>

#include "compile_util.h"
#include "oneapi/ccl.hpp"
#include "shm_reduction.h"
#include "timeline.h"

class Messenger {
public:
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

        if (same_hostnames) {
            local_ranks_flag = true;
        } else {
            local_ranks_flag = false;
        }

        extern Messenger *gmessenger;
        extern void globalBarrier();

        gmessenger = this;
        pshm = new ShmReduction(globalBarrier, rank, size);
#endif
    }

    ~Messenger() {
        delete pcomm;
#ifdef USE_SHM
        delete pshm;
#endif
    }

    bool isMaster() { return rank == 0; }

    int getRank() { return rank; }

    int getSize() { return size; }

    // From some example code of oneCCL, inplace reducing is supported
    template <typename T>
    void reduceAdd(T *sendBuf, T *recvBuf, size_t count) {
        TimeLine t("Messenger.reduceAdd");

#ifdef USE_SHM
        if (count * sizeof(T) > pshm->getSHMSize() || ((count % 16) != 0 || !local_ranks_flag))
            ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *pcomm).wait();
        else
            pshm->reduceAdd(sendBuf, recvBuf, count, rank, size);
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
    Messenger(const Messenger &messenger);
    Messenger &operator=(const Messenger &messenger);

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
