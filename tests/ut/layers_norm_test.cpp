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
static void layer_norm_ref(T *output, const T *input, const T *gamma, const T *beta, int rows, int cols,
        int iStride = -1, int oStride = -1, const float epsilon = 1e-5) {
    if (iStride == -1) iStride = cols;
    if (oStride == -1) oStride = cols;

    // Iterate over rows
    for (int i = 0; i < rows; ++i) {
        // Compute mean
        float mean = 0.0;
        for (int j = 0; j < cols; ++j) {
            mean += input[i * iStride + j];
        }
        mean /= cols;

        // Compute variance
        float variance = 0.0;
        for (int j = 0; j < cols; ++j) {
            T diff = input[i * iStride + j] - mean;
            variance += diff * diff;
        }
        variance /= cols;

        // Normalize
        T inv_std_dev = static_cast<T>(1.0 / std::sqrt(variance + epsilon));
        for (int j = 0; j < cols; ++j) {
            output[i * oStride + j] = gamma[j] * (input[i * iStride + j] - mean) * inv_std_dev + beta[j];
        }
    }
}

template <typename T>
static void compareLayerNorm(int rows, int cols) {

    T *input = (T *)aligned_alloc(64, rows * cols * sizeof(T));
    T *gamma = (T *)aligned_alloc(64, cols * sizeof(T));
    T *beta = (T *)aligned_alloc(64, cols * sizeof(T));
    T *ourOutput = (T *)aligned_alloc(64, rows * cols * sizeof(T));
    T *refOutput = (T *)aligned_alloc(64, rows * cols * sizeof(T));

    for (int i = 0; i < rows * cols; ++i) {
        input[i] = static_cast<T>(1.0f * rand() / RAND_MAX);
    }

    for (int i = 0; i < cols; ++i) {
        gamma[i] = static_cast<T>(1.0f);
    }

    for (int i = 0; i < cols; ++i) {
        beta[i] = static_cast<T>(0.0f);
    }

    if constexpr (std::is_same<T, float>::value) {
        xft::invokeLayerNorm(xft::DataType::fp32, (void *)ourOutput, (const void *)input, (const void *)gamma,
                (const void *)beta, rows, cols);
        layer_norm_ref<float>(refOutput, (const T *)input, (const T *)gamma, (const T *)beta, rows, cols);
    } else if constexpr (std::is_same<T, float16_t>::value) {
        xft::invokeLayerNorm(xft::DataType::fp16, (void *)ourOutput, (const void *)input, (const void *)gamma,
                (const void *)beta, rows, cols);
        layer_norm_ref<float16_t>(refOutput, (const T *)input, (const T *)gamma, (const T *)beta, rows, cols);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        xft::invokeLayerNorm(xft::DataType::bf16, (void *)ourOutput, (const void *)input, (const void *)gamma,
                (const void *)beta, rows, cols);
        layer_norm_ref<bfloat16_t>(refOutput, (const T *)input, (const T *)gamma, (const T *)beta, rows, cols);
    }

    for (int i = 0; i < rows * cols; ++i) {
        EXPECT_LT(std::abs((float)refOutput[i] - (float)ourOutput[i]), 0.01);
    }

    free(input);
    free(gamma);
    free(beta);
    free(ourOutput);
    free(refOutput);
}

TEST(LayerNorm, float) {
    compareLayerNorm<float>(128, 128);
    compareLayerNorm<float>(5120, 5120);
    compareLayerNorm<float>(5120, 5120 * 3);
    compareLayerNorm<float>(rand() % 100 + 100, rand() % 100 + 100);
}

TEST(LayerNorm, bfloat16_t) {
    compareLayerNorm<bfloat16_t>(128, 128);
    compareLayerNorm<bfloat16_t>(5120, 5120);
    compareLayerNorm<bfloat16_t>(5120, 5120 * 3);
    compareLayerNorm<bfloat16_t>(rand() % 100 + 100, rand() % 100 + 100);
}

TEST(LayerNorm, float16_t) {
    compareLayerNorm<float16_t>(128, 128);
    compareLayerNorm<float16_t>(5120, 5120);
    compareLayerNorm<float16_t>(5120, 5120 * 3);
    compareLayerNorm<float16_t>(rand() % 100 + 100, rand() % 100 + 100);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}