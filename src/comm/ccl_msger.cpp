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
#include <cstdlib>
#include <mpi.h>
#include "oneapi/ccl.hpp"

ccl::communicator *pcomm;

int init(int *rank, int *size) {
    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;

    if (*rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    pcomm = new ccl::communicator(ccl::create_communicator(*size, *rank, kvs));

    *rank = pcomm->rank();
    *size = pcomm->size();

#ifdef USE_SHM
    char my_hostname[MPI_MAX_PROCESSOR_NAME];
    char all_hostnames[MPI_MAX_PROCESSOR_NAME * MPI_MAX_PROCESSOR_NAME];
    int hostname_len;

    // Check ranks are on the same physical machine
    MPI_Get_processor_name(my_hostname, &hostname_len);
    MPI_Allgather(my_hostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            MPI_COMM_WORLD);

    int same_hostnames = 1;
    for (int i = 1; i < *size; i++) {
        if (strcmp(my_hostname, &all_hostnames[i * MPI_MAX_PROCESSOR_NAME]) != 0) {
            same_hostnames = 0;
            break;
        }
    }
    return same_hostnames;
#endif
    return 0;
}

void mpiFinalized(int *is_finalized) {
    MPI_Finalized(is_finalized);
}

void mpiFinalize() {
    MPI_Finalize();
}

void freePCOMM() {
    delete pcomm;
}

void allreduce(float *sendBuf, float *recvBuf, size_t count) {
    ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *pcomm).wait();
}

void broadcast(int *buf, size_t count) {
    ccl::broadcast(buf, count, 0, *pcomm).wait(); // assume always broadcast from master (rank 0)
}

void allgatherv(
        const float *send_buf, size_t count, float *recv_buf, const std::vector<long unsigned int> &recv_counts) {
    ccl::allgatherv(send_buf, count, recv_buf, recv_counts, *pcomm).wait();
}