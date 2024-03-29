// Copyright (c) 2023-2024 Intel Corporation
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
#include "messenger.h"

#include "gtest/gtest.h"

class MyTestSuite : public ::testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    Messenger &myMessenger = Messenger::getInstance();
};

TEST_F(MyTestSuite, broadcast) {
    int ids[128];
    if (myMessenger.isMaster()) {
        for (int i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
            ids[i] = i + 1;
        }
    }

    myMessenger.broadcast(ids, sizeof(ids) / sizeof(ids[0]));

    for (int i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
        EXPECT_EQ(ids[i], i + 1);
    }
}

TEST_F(MyTestSuite, allreduce_ccl) {
    int batchSize = 1;
    int seqLen = 1024;
    int hiddenSize = 4096;

    int totalNums = batchSize * seqLen * hiddenSize + 1;
    float *data = (float *)malloc(totalNums * sizeof(float));
    for (int i = 0; i < totalNums; ++i) {
        data[i] = myMessenger.getRank() + 1;
    }

    myMessenger.reduceAdd(data, data, totalNums);

    int size = myMessenger.getSize();
    float expected = size * (size + 1) / 2;
    float eps = 1e-5;

    for (int i = 0; i < totalNums; ++i) {
        EXPECT_NEAR(data[i], expected, eps);
    }
    free(data);
}

TEST_F(MyTestSuite, allreduce_shm) {
    int batchSize = 1;
    int seqLen = 1024;
    int hiddenSize = 4096;

    int totalNums = batchSize * seqLen * hiddenSize;
    float *data = (float *)malloc(totalNums * sizeof(float));
    for (int i = 0; i < totalNums; ++i) {
        data[i] = myMessenger.getRank() + 1;
    }

    myMessenger.reduceAdd(data, data, totalNums);

    int size = myMessenger.getSize();
    float expected = size * (size + 1) / 2;
    float eps = 1e-5;

    for (int i = 0; i < totalNums; ++i) {
        EXPECT_NEAR(data[i], expected, eps);
    }
    free(data);
}

int main(int argc, char **argv) {
    // Use system clock for seed
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
