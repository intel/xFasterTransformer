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

#include <immintrin.h>
#include <vector>

#include "add_util.h"
#include "bfloat16.h"
#include "copy_util.h"
#include "debugger.h"
#include "decoder_util.h"
#include "llm_params.h"
#include "mlp_llama.h"
#include "timeline.h"

template <typename WeiT, typename InT = bfloat16_t, typename ImT = bfloat16_t, typename OutT = bfloat16_t>
class Qwen3MOE {
public:
    Qwen3MOE(int layerId, DecoderContext *ctx) : norm(ctx) {
        for (int i = 0; i < ctx->sparseExperts; ++i) {
            experts.emplace_back(new LlamaMLP<WeiT, InT, ImT, OutT>(layerId, ctx));
        }
    }

    ~Qwen3MOE() {
        for (auto &expert : experts) {
            delete expert;
        }
    }

    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float * /*unused*/, const OriWeiT *upW, const float *upS, const float *upZ, const float * /*unused*/,
            const float *normW, const float * /*unused*/, const OriWeiT *downW, const float *downS, const float *downZ,
            const float *downB, bool trans = true) {
        xft::Logger::error("Cannot use the original setWeights function in Qwen3MOE.");
        exit(-1);
    }

    template <typename WType>
    void setWeights(DecoderContext *ctx, FFNParams *ffnParams) {
        Qwen3MOEParams *ffn = dynamic_cast<Qwen3MOEParams *>(ffnParams);
        if (ffn == nullptr) {
            xft::Logger::error("Cannot cast FFNParams to Qwen3MOEParams.");
            exit(-1);
        }

        this->norm.setWeight(ffn->norm.gamma, nullptr, ctx->hiddenSize);

        prepareGateWeight<WType>(ctx, ffn->gating.weight);

#ifdef XFT_DEBUG
        dbg.debugPrint("gatingWeight:\n");
        dbg.dumpMatrix(gatingWeight);
#endif

        // Check the size of sparse experts
        if (ffn->experts.size() != this->experts.size() || ctx->sparseExperts != this->experts.size()) {
            xft::Logger::error("The number of experts is not consistent. %d %d %d", ffn->experts.size(),
                    this->experts.size(), ctx->sparseExperts);
            exit(-1);
        }

        for (int i = 0; i < ctx->sparseExperts; ++i) {
            xft::ExpertParams &expertParams = ffn->experts[i];
            experts[i]->template setWeights<WType>(ctx, expertParams);
        }
    }

#ifdef XFT_DEBUG
    void setDebugger(const Debugger &debugger) {
        this->dbg = debugger;
        for (auto &expert : experts) {
            expert->setDebugger(debugger);
        }
    }
#endif

    /**
     * # Fully Connected
     * residual = hidden_states
     * hidden_states = self.post_attention_layernorm(hidden_states)
     * hidden_states, router_logits = self.block_sparse_moe(hidden_states)
     * hidden_states = residual + hidden_states
     */
    void forward(DecoderContext *ctx, InT *input, OutT *output, int iStride, int oStride,
            bool doLnBefore = true /*not used*/, int totInSeqLen = 0) {
        TimeLine t("Qwen3MOE");

        const int M = totInSeqLen == 0 ? ctx->batchSize * ctx->inputSeqLen : totInSeqLen;
        const int hiddenSize = ctx->hiddenSize;
        const int expertNum = ctx->sparseExperts;

        static_assert(sizeof(ctx->normBuf.Data()[0]) >= sizeof(ImT), "normBuff is not big enough!");

        xft::Matrix<InT> inBuffer(input, M, hiddenSize, iStride);
        xft::Matrix<OutT> outBuffer(output, M, hiddenSize, oStride);
        xft::Matrix<ImT> normBuffer(
                (ImT *)ctx->normBuf.Data(), ctx->normBuf.Rows(), ctx->normBuf.Cols(), ctx->normBuf.Stride());

#ifdef XFT_DEBUG
        dbg.debugPrint("MOE Input: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
        dbg.dumpMatrix(inBuffer);
#endif

        // Copy if input and output are different, so that the residual connection can be done
        Messenger &messenger = Messenger::getInstance();
        if ((void *)input != (void *)output && messenger.isMaster()) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                xft::copy(output + i * oStride, input + i * iStride, hiddenSize);
            }
        } else {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                memset(output + i * oStride, 0, sizeof(OutT) * hiddenSize);
            }
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("MOE Outputbuffer init: [%d, %d] (%d)\n", outBuffer.Rows(), outBuffer.Cols(), outBuffer.Stride());
        dbg.dumpMatrix(outBuffer);
#endif

        // Normalize input
        if (doLnBefore == true) {
            norm.forward(inBuffer.Data(), normBuffer.Data(), M, inBuffer.Stride(), normBuffer.Stride(), 1e-6);
#ifdef XFT_DEBUG
            dbg.debugPrint("Norm: \n");
            dbg.dumpMatrix(normBuffer);
#endif
        }

        // Gating
        OutT *gateLogits = ctx->getBuffer<OutT>("gateLogits", M * expertNum, ctx->device);
        ctx->mmHelper->compute(false, M, expertNum, hiddenSize, 1.0f, normBuffer.Data(), normBuffer.Stride(),
                gatingWeight.Data(), nullptr, nullptr, nullptr, 0.0f, gateLogits, expertNum);
#ifdef XFT_DEBUG
        dbg.debugPrint("Gate: \n");
        dbg.dumpMatrix(gateLogits, M, expertNum, expertNum);
#endif

        // Softmax
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            DecoderUtil::computeSoftmax(gateLogits + i * expertNum, expertNum);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("Gate After Softmax: \n");
        dbg.dumpMatrix(gateLogits, M, expertNum, expertNum);
#endif

        // TODO: special case, we could do some optimization here
        if (M == 1) {}

        // Select top K experts for each token
        int *selExperts = reinterpret_cast<int *>(ctx->imOut.Data());
        float *expertWeight = reinterpret_cast<float *>(selExperts + M * ctx->numExpertsPerTok);
        selectTopKExperts(gateLogits, ctx->numExpertsPerTok, M, expertNum, selExperts, expertWeight);

#ifdef XFT_DEBUG
        dbg.debugPrint("Selected experts:\n");
        dbg.dumpMatrix(selExperts, M, ctx->numExpertsPerTok, ctx->numExpertsPerTok);
        dbg.debugPrint("Selected experts weights:\n");
        dbg.dumpMatrix(expertWeight, M, ctx->numExpertsPerTok, ctx->numExpertsPerTok);
#endif

        std::vector<int> idx[expertNum]; // index for each expert
        std::vector<float> weights[expertNum]; // weight for each expert
        // printf("M = %d, expertNum = %d\n", M, expertNum);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < ctx->numExpertsPerTok; ++j) {
                // printf("selExperts[%d] = %d, expertWeight[%d] = %f\n", j + i * ctx->numExpertsPerTok, selExperts[j + i * ctx->numExpertsPerTok],
                //         j + i * ctx->numExpertsPerTok, expertWeight[j + i * ctx->numExpertsPerTok]);
                idx[selExperts[j + i * ctx->numExpertsPerTok]].push_back(i);
                weights[selExperts[j + i * ctx->numExpertsPerTok]].push_back(
                        expertWeight[j + i * ctx->numExpertsPerTok]);
            }
        }

        // Call forward function of selected experts
        for (int i = 0; i < expertNum; ++i) {
            size_t rowNum = idx[i].size();
            if (idx[i].empty()) { continue; }

            // Gather input for expert i
            ImT *expertData = ctx->getBuffer<ImT>("expertData", rowNum * hiddenSize, ctx->device);
            gatherInput(expertData, hiddenSize, normBuffer.Data(), normBuffer.Stride(), idx[i]);

#ifdef XFT_DEBUG
            dbg.debugPrint("Expert %d, input[%d]:\n", i, rowNum);
            dbg.dumpMatrix(expertData, rowNum, hiddenSize, hiddenSize);
#endif
            // Call forward function of expert i
            experts[i]->forward(ctx, expertData, expertData, hiddenSize, hiddenSize, false, rowNum, true);
#ifdef XFT_DEBUG
            dbg.debugPrint("Expert %d, w=%f..., output:\n", i, weights[i][0]);
            dbg.dumpMatrix(expertData, rowNum, hiddenSize, hiddenSize);
#endif

            // Scatter output of expert i
            scatterOutput(output, oStride, expertData, hiddenSize, idx[i], weights[i]);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("MOE output:\n");
        dbg.dumpMatrix(output, M, hiddenSize, hiddenSize);
#endif
    }

private:
    template <typename WType>
    void prepareGateWeight(DecoderContext *ctx, const void *gateW) {
        int M = ctx->hiddenSize;
        int N = ctx->sparseExperts;

        using GateType = typename GateTypeSelector<WeiT>::type;
        Matrix<GateType> convertedGateW;
        xft::Vector<float> placeholder;

        ctx->mmHelper->convertWeight(false, M, N, (const WType *)gateW, nullptr, nullptr, convertedGateW, placeholder,
                placeholder, placeholder);

        convertedGateW.Resize(M, N, N);

#ifdef XFT_DEBUG
        dbg.debugPrint("convertedGateW:\n");
        dbg.dumpMatrix(convertedGateW);
#endif

        ctx->mmHelper->packWeight(false, convertedGateW, gatingWeight);
    }

    // Gather input for expert i
    void gatherInput(ImT *expertData, int hiddenSize, ImT *input, int iStride, const std::vector<int> &idx) {
#pragma omp parallel for
        for (size_t i = 0; i < idx.size(); ++i) {
            memcpy(expertData + i * hiddenSize, input + idx[i] * iStride, hiddenSize * sizeof(ImT));
        }
    }

    // Scatter output of expert i
    void scatterOutput(OutT *output, int oStride, ImT *expertData, int hiddenSize, const std::vector<int> &idx,
            const std::vector<float> &weights) {
#pragma omp parallel for
        for (size_t i = 0; i < idx.size(); ++i) {
            float scale = weights[i];
            xft::addto(output + idx[i] * oStride, expertData + i * hiddenSize, scale, hiddenSize);
        }
    }

    // TODO: abstract this function to a common place, which can be used by deepseek
    // Select top k element in N elements of array, store idx and value into selIdx and selWeight
    template <typename T>
    void topK(T *array, int N, int topk, int *selIdx, float *selWeight) {
        std::vector<std::pair<T, int>> vec;
        for (int i = 0; i < N; ++i) {
            vec.emplace_back(array[i], i);
        }
        std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<T, int>>());
        for (int i = 0; i < topk; ++i) {
            selIdx[i] = vec[i].second;
            if (selWeight != nullptr) selWeight[i] = vec[i].first;
        }
    }

    // logits: [M, N]
    template <typename T>
    void selectTopKExperts(T *logits, int K, int M, int N, int *selExperts, float *expertWeight) {
        // TODO: optimize
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            topK(logits + i * N, N, K, selExperts + i * K, expertWeight + i * K);
        }
        // Norm
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            // add a small value to avoid div 0
            float sum = 1e-20f;
            for (int j = 0; j < K; j += 16) {
                int remain = K - j;
                __mmask16 mask = remain >= 16 ? 0xffff : (1 << remain) - 1;
                auto v = xft::load_avx512(mask, expertWeight + i * K + j);
                sum = sum + _mm512_reduce_add_ps(v);
            }
            // add a small value to avoid div 0
            __m512 vsum = _mm512_set1_ps(1 / sum);
            for (int j = 0; j < K; j += 16) {
                int remain = K - j;
                __mmask16 mask = remain >= 16 ? 0xffff : (1 << remain) - 1;
                auto v = xft::load_avx512(mask, expertWeight + i * K + j);
                xft::store_avx512(expertWeight + i * K + j, mask, _mm512_mul_ps(v, vsum));
            }
        }
    }

private:
    xft::RmsNorm norm;
    xft::Matrix<typename GateTypeSelector<WeiT>::type> gatingWeight;
    std::vector<LlamaMLP<WeiT, InT, ImT, OutT> *> experts;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};
