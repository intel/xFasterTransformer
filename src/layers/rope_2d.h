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
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

// 2D rotary embedding for ChatGLM
class RotaryEmbedding2D {
public:
    RotaryEmbedding2D(const int dim, const int max_position_embeddings = 2048, const float base = 10000);

    ~RotaryEmbedding2D() {}

    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *positions);
    void forward(float *query, float *key, int totSeqLen, int qStride, int kStride, int qHeads, int kHeads,
            int *positionIds);

private:
    void prepareEmbedding();

private:
    static bool initialized;
};
