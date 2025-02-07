// Copyright (c) 2025 Intel Corporation
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

#include "attention.h"
#include "llm_params.h"
#include "logger.h"
#include "my_types.h"
#include "rms_norm.h"

template <typename WeiT>
class DeepSeekAttention {
public:
    DeepSeekAttention(int layerId, DecoderContext *ctx) {}

    void setWeights(DecoderContext *ctx, const OriWeiT *queryWeight, const float *queryScale, const float *queryZero,
            const float *queryBias, const OriWeiT *keyWeight, const float *keyScale, const float *keyZero,
            const float *keyBias, const OriWeiT *valueWeight, const float *valueScale, const float *valueZero,
            const float *valueBias, const OriWeiT *attnOutWeight, const float *attnOutScale, const float *attnOutZero,
            const float *attnOutBias, bool doLNorm, const float *gamma1, const float *beta1, bool trans = true) {
        xft::Logger::error("Cannot use the old API to set weights in DeepSeekAttention.");
        exit(-1);
    }

    void setWeights(DecoderContext *ctx, xft::AttnParams *attnParams) {
        xft::MLAttnParams *mlap = dynamic_cast<xft::MLAttnParams *>(attnParams);
        if (mlap == nullptr) {
            xft::Logger::error("Cannot cast AttnParams to MLAttnParams.");
            exit(-1);
        }

        // Suppose the weight is not transposed (report error if it is transposed)
        if (mlap->q_a_proj.wtrans || mlap->kv_a_proj.wtrans) {
            xft::Logger::error("The weights should not be transposed.");
            exit(-1);
        }

        // Check the data type, so that we can safely merge or copy
        if (!isSameWeiType(mlap->q_a_proj.wtype) || !isSameWeiType(mlap->kv_a_proj.wtype)
                || !isSameWeiType(mlap->q_b_proj.wtype) || !isSameWeiType(mlap->kv_b_proj.wtype)) {
            xft::Logger::error("The weight type is not the same.");
            exit(-1);
        }

        // Merge q_a and kv_a
        int mergedDim = mlap->q_a_proj.output_dim + mlap->kv_a_proj.output_dim;
        WeiT *buffer = (WeiT *)aligned_alloc(64, ctx->hiddenSize * mergedDim * sizeof(WeiT));

#pragma omp parallel for
        for (int i = 0; i < ctx->hiddenSize; ++i) {
            memcpy(buffer + i * mergedDim, mlap->q_a_proj.weight + i * mlap->q_a_proj.output_dim,
                    mlap->q_a_proj.output_dim * sizeof(WeiT));
            memcpy(buffer + i * mergedDim + mlap->q_a_proj.output_dim,
                    mlap->kv_a_proj.weight + i * mlap->kv_a_proj.output_dim, mlap->kv_a_proj.output_dim * sizeof(WeiT));
        }

        // Pack the merged weights
        qkvAWeights.Resize(ctx->hiddenSize, mergedDim);
        xft::Matrix mergedW(buffer, ctx->hiddenSize, mergedDim, mergedDim);
        ctx->mmHelper->packWeight(mlap->q_a_proj.wtrans, mergedW, qkvAWeights);

        free(buffer);

        // Pack the weights for q_b and kv_b
        packDenseWeights(mlap->q_b_proj, qBWeights);
        packDenseWeights(mlap->kv_b_proj, kvBWeights);

        // Pack the weights for output
        packDenseWeights(mlap->o_proj, outWeights);

        // Norm params
        this->inputNorm.setWeight(mlap->input_norm.gamma, nullptr, ctx->hiddenSize);
        this->qANorm.setWeight(mlap->q_a_norm.gamma, nullptr, ctx->qLoraRank);
        this->kvANorm.setWeight(mlap->kv_a_norm.gamma, nullptr, ctx->kvLoraRank);
    }

private:
    void packDenseWeights(xft::DenseLayerParams &dense, xft::Matrix<WeiT> &packedW) {
        xft::Matrix<WeiT> w(dense.weight, dense.input_dim, dense.output_dim, dense.output_dim);
        packedW.Resize(dense.input_dim, dense.output_dim);
        ctx->mmHelper->packWeight(dense.wtrans, w, packedW);
    }

    bool isSameWeiType(xft::ParamType type) {
        if constexpr (std::is_same_v<WeiT, int8_t>) {
            return type == ParamType::Int8;
        } else if constexpr (std::is_same_v<WeiT, float16_t>) {
            return type == ParamType::FP16;
        } else if constexpr (std::is_same_v<WeiT, bfloat16_t>) {
            return type == ParamType::BF16;
        } else if constexpr (std::is_same_v<WeiT, float>) {
            return type == ParamType::FP32;
        }
        return false;
    }

private:
    xft::Matrix<WeiT> qkvAWeights; // merged q_a and kv_a
    xft::Matrix<WeiT> qBWeights;
    xft::Matrix<WeiT> kvBWeights;

    xft::Matrix<WeiT> outWeights;

    RmsNorm inputNorm;
    RmsNorm qANorm;
    RmsNorm kvANorm;
};