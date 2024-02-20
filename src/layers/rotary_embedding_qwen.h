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
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_map>

#include "bfloat16.h"

/*  Sample:
        int bs = 2 headnum = 3 seq = 4  dim = 6;
        int max_len = 10;
        int pos_ids[4] = {2,0,1,3}; //  seq = 4 , Each batch have same value
        int pos_shape[2] = {bs, seq};
        float x[144] = {0, 1, 1,...}; // bs * h * seq * dim = 144
        int xshape[4] = {bs,headnum,seq,dim};
        Forward
        QwenRotaryEmbedding emb(dim, seq);
        float *embd = emb.forward(x, x_shape, pos_ids, pos_shape);
*/

class QwenRotaryEmbedding {
public:
    QwenRotaryEmbedding(const int dim, const int max_position_embeddings = 2048, const float base = 10000);

    ~QwenRotaryEmbedding();

    void forward(float *query, float *key, int qStride, int kStride, const int *qkShape, const int *positionIds);

    void forward(
            bfloat16_t *query, bfloat16_t *key, int qStride, int kStride, const int *qkShape, const int *positionIds);

    void init_logn(const int max_length = 2048);

private:
    float getNewBaseValue(const int true_seq_len, const int max_seq_length = -1);
    void QwenCalEmb(float *inv_freq, float base, std::unordered_map<float, std::tuple<float *, float *>> &embCosSin);

private:
    static bool initialized;
    static bool logn_initialized;
    static int inv_freq_size;
    static int max_seq_len_cached;
    int dim = 0;
    float base_initial = 10000.0;
    float base = 10000.0;
};
