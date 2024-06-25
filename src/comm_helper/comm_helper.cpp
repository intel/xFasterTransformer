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

static ccl::communicator *pcomm;

// world_color is initialized to pipeline_parallel_stages_num(pp_size)
// and will be re-assign to world_color of MPI == ppRank
extern "C" int init(int *world_size, int *world_rank, int *world_color) {
    ccl::init();

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, world_rank);

    // world_color = world_rank / tpSize = world_rank / (world_size / ppSize)
    // like: world_color = 0~7 / (8 / 4), XFT_PIPELINE_STAGE = ppSize = 4; tpSize = 2
    //       world_rank = 0, 1,  ->  world_color = ppRank = 0, 0,  ->  tpRank = 0, 1;
    //                    2, 3,                             1, 1,               0, 1;
    //                    4, 5,                             2, 2,               0, 1;
    //                    6, 7;                             3, 3;               0, 1;
    *world_color = *world_rank / (*world_size / *world_color);
    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, *world_color, *world_rank, &row_comm);

    int row_size, row_rank;
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(row_comm, &row_rank);

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type mainAddr;

    if (row_rank == 0) {
        kvs = ccl::create_main_kvs();
        mainAddr = kvs->get_address();
        MPI_Bcast((void *)mainAddr.data(), mainAddr.size(), MPI_BYTE, 0, row_comm);
    } else {
        MPI_Bcast((void *)mainAddr.data(), mainAddr.size(), MPI_BYTE, 0, row_comm);
        kvs = ccl::create_kvs(mainAddr);
    }

    pcomm = new ccl::communicator(ccl::create_communicator(row_size, row_rank, kvs));

    *world_size = pcomm->size();
    *world_rank = pcomm->rank();

#ifdef USE_SHM
    char myHostname[MPI_MAX_PROCESSOR_NAME];
    char all_hostnames[MPI_MAX_PROCESSOR_NAME * MPI_MAX_PROCESSOR_NAME];
    int hostnameLen;

    // Check ranks are on the same physical machine
    MPI_Get_processor_name(myHostname, &hostnameLen);
    MPI_Allgather(myHostname, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, all_hostnames, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
            MPI_COMM_WORLD);

    int sameHostnames = 1;
    for (int i = 1; i < *world_size; i++) {
        int id = (*world_rank + i) % (*world_size);
        if (strcmp(myHostname, &all_hostnames[id * MPI_MAX_PROCESSOR_NAME]) != 0) {
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

extern "C" void allreduceBF16(void *sendBuf, void *recvBuf, size_t count) {
    ccl::allreduce(sendBuf, recvBuf, count, ccl::datatype::bfloat16, ccl::reduction::sum, *pcomm).wait();
}

extern "C" void allreduceFP16(void *sendBuf, void *recvBuf, size_t count) {
    ccl::allreduce(sendBuf, recvBuf, count, ccl::datatype::float16, ccl::reduction::sum, *pcomm).wait();
}

extern "C" void broadcast(int *buf, size_t count) {
    ccl::broadcast(buf, count, 0, *pcomm).wait(); // assume always broadcast from master (rank 0)
}

extern "C" void allgatherv(
        const float *sendBuf, size_t count, float *recvBuf, const std::vector<long unsigned int> &recvCounts) {
    ccl::allgatherv(sendBuf, count, recvBuf, recvCounts, *pcomm).wait();
}

extern "C" void worldSendFP32(const float *buf, int count, int dest, int tag) {
    MPI_Send((const void *)buf, count, MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
}

extern "C" void worldRecvFP32(float *buf, int count, int source, int tag) {
    MPI_Recv((void *)buf, count, MPI_FLOAT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

extern "C" void worldSendINT32(const int32_t *buf, int count, int dest, int tag) {
    MPI_Send((const void *)buf, count, MPI_INT32_T, dest, tag, MPI_COMM_WORLD);
}

extern "C" void worldRecvINT32(int32_t *buf, int count, int source, int tag) {
    MPI_Recv((void *)buf, count, MPI_INT32_T, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}
