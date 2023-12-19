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

extern "C" int init(int *rank, int *size) {
    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, size);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type mainAddr;

    if (*rank == 0) {
        kvs = ccl::create_main_kvs();
        mainAddr = kvs->get_address();
        MPI_Bcast((void *)mainAddr.data(), mainAddr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Bcast((void *)mainAddr.data(), mainAddr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(mainAddr);
    }

    pcomm = new ccl::communicator(ccl::create_communicator(*size, *rank, kvs));

    *rank = pcomm->rank();
    *size = pcomm->size();

#ifdef USE_SHM
    char myHostname[MPI_MAX_PROCESSOR_NAME];
    char all_hostnames[MPI_MAX_PROCESSOR_NAME * MPI_MAX_PROCESSOR_NAME];
    int hostnameLen;

    // Check ranks are on the same physical machine
    MPI_Get_processor_name(myHostname, &hostnameLen);
    MPI_Allgather(myHostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            MPI_COMM_WORLD);

    int sameHostnames = 1;
    for (int i = 1; i < *size; i++) {
        if (strcmp(myHostname, &all_hostnames[i * MPI_MAX_PROCESSOR_NAME]) != 0) {
            sameHostnames = 0;
            break;
        }
    }
    return sameHostnames;
#endif
    return 0;
}

extern "C" void mpiFinalize() {
    int isFinalized = 0;
    MPI_Finalized(&isFinalized);
    if (!isFinalized) { MPI_Finalize(); }
}

extern "C" void freePCOMM() {
    delete pcomm;
}

extern "C" void allreduce(float *sendBuf, float *recvBuf, size_t count) {
    ccl::allreduce(sendBuf, recvBuf, count, ccl::reduction::sum, *pcomm).wait();
}

extern "C" void broadcast(int *buf, size_t count) {
    ccl::broadcast(buf, count, 0, *pcomm).wait(); // assume always broadcast from master (rank 0)
}

extern "C" void allgatherv(
        const float *sendBuf, size_t count, float *recvBuf, const std::vector<long unsigned int> &recvCounts) {
    ccl::allgatherv(sendBuf, count, recvBuf, recvCounts, *pcomm).wait();
}