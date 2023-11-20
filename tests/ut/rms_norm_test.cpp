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
#include <cmath>
#include <type_traits>

#include "bfloat16.h"
#include "float16.h"
#include "layers_norm.h"
#include "gtest/gtest.h"

template <typename T>
static void refRmsNorm(T *output, const T *input, const T *weight, int rows, int cols, int iStride = -1,
        int oStride = -1, float epsilon = 1e-6) {

    // If iStride or oStride are not provided, set them to cols as default
    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

    for (int i = 0; i < rows; ++i) {
        float sum_of_squares = 0.0;

        for (int j = 0; j < cols; ++j) {
            T weighted_input = input[i * iStride + j] * weight[j];
            sum_of_squares += weighted_input * weighted_input;
        }

        float rms = std::sqrt(sum_of_squares / cols + epsilon);

        for (int j = 0; j < cols; ++j) {
            output[i * oStride + j] = input[i * iStride + j] * weight[j] / rms;
        }
    }
}

template <typename T>
static void compareRMSNorm(int rows, int cols) {

    T *input = (T *)aligned_alloc(64, rows * cols * sizeof(T));
    T *weight = (T *)aligned_alloc(64, cols * sizeof(T));
    T *ourOutput = (T *)aligned_alloc(64, rows * cols * sizeof(T));
    T *refOutput = (T *)aligned_alloc(64, rows * cols * sizeof(T));

    for (int i = 0; i < rows * cols; ++i) {
        input[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }

    for (int i = 0; i < cols; ++i) {
        weight[i] = static_cast<T>(1.0f);
    }

    if constexpr (std::is_same<T, float>::value) {
        xft::invokeRmsNorm(
                xft::DataType::fp32, (void *)ourOutput, (const void *)input, (const void *)weight, rows, cols);
        refRmsNorm<float>(refOutput, (const T *)input, (const T *)weight, rows, cols);
    } else if constexpr (std::is_same<T, float16_t>::value) {
        xft::invokeRmsNorm(
                xft::DataType::fp16, (void *)ourOutput, (const void *)input, (const void *)weight, rows, cols);
        refRmsNorm<float16_t>(refOutput, (const T *)input, (const T *)weight, rows, cols);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        xft::invokeRmsNorm(
                xft::DataType::bf16, (void *)ourOutput, (const void *)input, (const void *)weight, rows, cols);
        refRmsNorm<bfloat16_t>(refOutput, (const T *)input, (const T *)weight, rows, cols);
    }

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_LT(((float)refOutput[i] - (float)ourOutput[i]), 0.01);
    }

    free(input);
    free(weight);
    free(ourOutput);
    free(refOutput);
}

TEST(RMSNorm, float) {
    compareRMSNorm<float>(128, 128);
    compareRMSNorm<float>(5120, 5120);
    compareRMSNorm<float>(5120, 5120 * 3);
    compareRMSNorm<float>(rand() % 100 + 100, rand() % 100 + 100);
}

TEST(RMSNorm, bfloat16_t) {
    compareRMSNorm<bfloat16_t>(128, 128);
    compareRMSNorm<bfloat16_t>(5120, 5120);
    compareRMSNorm<bfloat16_t>(5120, 5120 * 3);
    compareRMSNorm<bfloat16_t>(rand() % 100 + 100, rand() % 100 + 100);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}