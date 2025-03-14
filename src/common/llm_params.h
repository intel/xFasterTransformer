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

namespace xft {

enum class ParamType { None, INT4, Int8, FP16, BF16, FP32 };

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
            case ParamType::Int8: return sizeof(int8_t);
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
    DenseLayerParams qkv;
    DenseLayerParams out;
    NormParams norm;

    AttnParams() : qkv {}, out {}, norm {} {}
    AttnParams(int hiddenSize, int qHeads, int kvHeads, int headSize, ParamType weiType, bool wTrans = false)
        : qkv(hiddenSize, headSize * (qHeads + kvHeads * 2), weiType, wTrans)
        , out(headSize * qHeads, hiddenSize, weiType, wTrans)
        , norm(hiddenSize) {}
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

} // namespace xft