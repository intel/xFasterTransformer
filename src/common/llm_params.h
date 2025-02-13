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

#include <vector>

#include "dtype.h"

namespace xft {

struct DenseLayerParams {
    void *weight; // Flat array of weight (row-major)
    float *bias; // Bias (not quantized)
    ParamType wtype;
    bool wtrans; // Weight is transposed or not

    int input_dim; // Input dimension
    int output_dim; // Output dimension

    // Quantization parameters (optional)
    float *weight_scale; // Scale
    float *weight_zp; // Zero point

    DenseLayerParams()
        : weight(nullptr)
        , bias(nullptr)
        , weight_scale(nullptr)
        , weight_zp(nullptr)
        , wtype(ParamType::None)
        , wtrans(false)
        , input_dim(0)
        , output_dim(0) {}

    DenseLayerParams(int inputDim, int outputDim, ParamType weiType, bool wTrans = false)
        : wtype(weiType), wtrans(wTrans), input_dim(inputDim), output_dim(outputDim) {
        weight = aligned_alloc(64, (size_t)(inputDim * outputDim * getWidth(weiType)));

        // The bias can be optional, thus we don't allocate memory here
        bias = nullptr;
        weight_scale = nullptr;
        weight_zp = nullptr;
    }

    DenseLayerParams(DenseLayerParams &&other) noexcept
        : weight(other.weight)
        , bias(other.bias)
        , weight_scale(other.weight_scale)
        , weight_zp(other.weight_zp)
        , wtype(other.wtype)
        , wtrans(other.wtrans)
        , input_dim(other.input_dim)
        , output_dim(other.output_dim) {
        other.weight = nullptr;
        other.bias = nullptr;
        other.weight_scale = nullptr;
        other.weight_zp = nullptr;
    }

    DenseLayerParams &operator=(DenseLayerParams &other);

    ~DenseLayerParams() {
        if (weight) free((void *)weight);
        if (bias) free((void *)bias);
        if (weight_scale) free((void *)weight_scale);
        if (weight_zp) free((void *)weight_zp);
    }

    void setBiasValue(const float *biasValue) {
        if (!bias) bias = (float *)aligned_alloc(64, output_dim * sizeof(float));
        memcpy((void *)bias, (void *)biasValue, output_dim * sizeof(float));
    }

    void removeBias() {
        if (bias) {
            free((void *)bias);
            bias = nullptr;
        }
    }

    float getWidth(ParamType type) const {
        switch (type) {
            case ParamType::INT4: return sizeof(int8_t) / 2.0f;
            case ParamType::INT8: return sizeof(int8_t);
            case ParamType::FP16: return sizeof(uint16_t);
            case ParamType::BF16: return sizeof(uint16_t);
            case ParamType::FP32: return sizeof(float);
            default: return 0;
        }
    }
};

struct NormParams {
    float *gamma; // Scale
    float *beta; // Bias
    int hidden_size; // Hidden size

    NormParams() : gamma(nullptr), beta(nullptr), hidden_size(0) {}
    NormParams(int hiddenSize) : hidden_size(hiddenSize) {
        gamma = (float *)aligned_alloc(64, hiddenSize * sizeof(float));
        beta = (float *)aligned_alloc(64, hiddenSize * sizeof(float));
        // Bias can be optional, thus we initialize it to 0
        memset(beta, 0, hiddenSize * sizeof(float));
    }

    NormParams(NormParams &&other) noexcept : gamma(other.gamma), beta(other.beta), hidden_size(other.hidden_size) {
        other.gamma = nullptr;
        other.beta = nullptr;
    }

    NormParams &operator=(NormParams &other);

    ~NormParams() {
        if (gamma) free((void *)gamma);
        if (beta) free((void *)beta);
    }

    // Set the beta value to 0, if it is not provided
    void emptyBeta() { memset(beta, 0, hidden_size * sizeof(float)); }
};

struct AttnParams {
    virtual ~AttnParams() = default;
};

// GQA/MHA/MQA
struct GQAttnParams : public AttnParams {
    DenseLayerParams qkv;
    DenseLayerParams out;
    NormParams norm;

    GQAttnParams() : qkv {}, out {}, norm {} {}
    GQAttnParams(int hiddenSize, int qHeads, int kvHeads, int headSize, ParamType weiType, bool wTrans = false)
        : qkv(hiddenSize, headSize * (qHeads + kvHeads * 2), weiType, wTrans)
        , out(headSize * qHeads, hiddenSize, weiType, wTrans)
        , norm(hiddenSize) {}
};

// MLA/DeepSeek
struct MLAttnParams : public AttnParams {
    DenseLayerParams q_a_proj;
    DenseLayerParams q_b_proj;
    DenseLayerParams kv_a_proj;
    DenseLayerParams kv_b_proj;
    DenseLayerParams o_proj;
    NormParams q_a_norm;
    NormParams kv_a_norm;
    NormParams input_norm;

    MLAttnParams() {}

    // qLoraRank: The rank of the query projection, typical value is 1536
    // kvLoraRank: The rank of the key and value projection, typical value is 512
    // headNum: The number of heads, typical value is 128
    // nopeDim: The dimension of the nope vector, typical value is 128
    // ropeDim: The dimension of the rope vector, typical value is 64
    MLAttnParams(int hiddenSize, int qLoraRank, int kvLoraRank, int headNum, int nopeDim, int ropeDim, int vHeadDim,
            ParamType weiType, bool wTrans = false)
        : q_a_proj(hiddenSize, qLoraRank > 0 ? qLoraRank : headNum * (nopeDim + ropeDim), weiType, wTrans)
        , q_b_proj(qLoraRank, qLoraRank > 0 ? headNum * (nopeDim + ropeDim) : 0, weiType, wTrans)
        , kv_a_proj(hiddenSize, kvLoraRank + ropeDim, weiType, wTrans)
        , kv_b_proj(kvLoraRank, headNum * (nopeDim + vHeadDim), weiType, wTrans)
        , o_proj(headNum * vHeadDim, hiddenSize, weiType, wTrans)
        , q_a_norm(qLoraRank > 0 ? qLoraRank : 0)
        , kv_a_norm(kvLoraRank)
        , input_norm(hiddenSize) {}
};

struct FFNParams {
    virtual ~FFNParams() = default;
};

// GPT FFN
struct GptFFNParams : public FFNParams {
    DenseLayerParams fc1;
    DenseLayerParams fc2;
    NormParams norm;

    GptFFNParams() : fc1 {}, fc2 {}, norm {} {}
    GptFFNParams(int hiddenSize, int intermediateSize, ParamType denseWType, bool wTrans = false)
        : fc1(hiddenSize, intermediateSize, denseWType, wTrans)
        , fc2(intermediateSize, hiddenSize, denseWType, wTrans)
        , norm(hiddenSize) {}
};

// LLAMA FFN
struct LlamaFFNParams : public FFNParams {
    DenseLayerParams gate;
    DenseLayerParams up;
    DenseLayerParams down;
    NormParams norm;

    LlamaFFNParams() : gate {}, up {}, down {}, norm {} {}
    LlamaFFNParams(int hiddenSize, int intermediateSize, ParamType denseWType, bool wTrans = false)
        : gate(hiddenSize, intermediateSize, denseWType, wTrans)
        , up(hiddenSize, intermediateSize, denseWType, wTrans)
        , down(intermediateSize, hiddenSize, denseWType, wTrans)
        , norm(hiddenSize) {}
};

struct ExpertParams {
    DenseLayerParams gate;
    DenseLayerParams up;
    DenseLayerParams down;

    ExpertParams() : gate {}, up {}, down {} {}
    ExpertParams(int hiddenSize, int intermediateSize, ParamType denseWType, bool wTrans = false)
        : gate(hiddenSize, intermediateSize, denseWType, wTrans)
        , up(hiddenSize, intermediateSize, denseWType, wTrans)
        , down(intermediateSize, hiddenSize, denseWType, wTrans) {}
};

// Mixtral FFN
struct MixtralFFNParams : public FFNParams {
    NormParams norm;
    DenseLayerParams gating; // Gating for MOE
    std::vector<ExpertParams> experts; // List of experts

    MixtralFFNParams() : norm {}, gating {}, experts {} {}
    MixtralFFNParams(int expertNum, int hiddenSize, int intermediateSize, ParamType denseWType, bool wTrans = false)
        : norm(hiddenSize), gating(hiddenSize, expertNum, denseWType, wTrans) {
        for (int i = 0; i < expertNum; ++i) {
            experts.emplace_back(hiddenSize, intermediateSize, denseWType, wTrans);
        }
    }
};

// DeepSeek MOE
struct DeepSeekFFNParams : public FFNParams {
    NormParams norm;
    DenseLayerParams gating; // Gating for MOE
    ExpertParams mlp; // normal mlp layer
    std::vector<ExpertParams> routedExperts; // List of routed experts
    ExpertParams sharedExpert; // shared expert, assume it has been merged into 1 weight

    DeepSeekFFNParams() : norm {}, gating {}, mlp {}, routedExperts {}, sharedExpert {} {}
    DeepSeekFFNParams(int routedExpertsNum, int sharedExpertNum, int hiddenSize, int intermediateSize,
            int moeIntermediateSize, ParamType denseWType, bool wTrans = false)
        : norm(hiddenSize)
        , gating(hiddenSize, routedExpertsNum, denseWType, wTrans)
        , mlp(hiddenSize, intermediateSize, denseWType, wTrans)
        , sharedExpert(hiddenSize, sharedExpertNum * moeIntermediateSize, denseWType, wTrans) {
        // leave this to load weights?
        //for (int i = 0; i < routedExpertsNum; ++i) {
        //    routedExperts.emplace_back(hiddenSize, moeIntermediateSize, denseWType, wTrans);
        //}
    }
};

} // namespace xft
