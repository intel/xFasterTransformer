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
#include <chrono>

#include "messenger.h"
#include "gtest/gtest.h"

class MyTestSuite : public ::testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    static void test(int bs, int seqLen, int hiddenSize) {
        Messenger &myMessenger = Messenger::getInstance();
        int totalNums = bs * seqLen * hiddenSize;
        float *data = (float *)malloc(totalNums * sizeof(float));
        float *tmp_data = (float *)malloc(totalNums * sizeof(float));
        for (int i = 0; i < totalNums; ++i) {
            data[i] = myMessenger.getRank() + 1;
        }

        // warm up
        for (int i = 0; i < 3; ++i) {
            myMessenger.reduceAdd(data, tmp_data, totalNums);
        }

        auto startTime = std::chrono::high_resolution_clock::now();
        myMessenger.reduceAdd(data, data, totalNums);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count() / 1000.0;
        std::cout << "node " << myMessenger.getRank() << "/" << myMessenger.getSize() - 1 << " runtime: " << duration
                  << " ms" << std::endl;

        int size = myMessenger.getSize();
        float expected = size * (size + 1) / 2;
        float eps = 1e-5;

        for (int i = 0; i < totalNums; ++i) {
            EXPECT_NEAR(data[i], expected, eps);
        }
        free(data);
        free(tmp_data);
    }
};

TEST_F(MyTestSuite, allreduce_128) {
    int batchSize = 1;
    int seqLen = 128;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

TEST_F(MyTestSuite, allreduce_256) {
    int batchSize = 1;
    int seqLen = 256;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

TEST_F(MyTestSuite, allreduce_512) {
    int batchSize = 1;
    int seqLen = 512;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

TEST_F(MyTestSuite, allreduce_1K) {
    int batchSize = 1;
    int seqLen = 1024;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

TEST_F(MyTestSuite, allreduce_2K) {
    int batchSize = 1;
    int seqLen = 2048;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

TEST_F(MyTestSuite, allreduce_4K) {
    int batchSize = 1;
    int seqLen = 4096;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

TEST_F(MyTestSuite, allreduce_8K) {
    int batchSize = 1;
    int seqLen = 8192;
    int hiddenSize = 5120;
    test(batchSize, seqLen, hiddenSize);
}

int main(int argc, char **argv) {
    // Use system clock for seed
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}