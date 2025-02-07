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
#include "llm_params.h"
#include "mlp_llama.h"
#include "timeline.h"

template <typename WeiT, typename InT = bfloat16_t, typename ImT = bfloat16_t, typename OutT = bfloat16_t>
class MixtralMLP {
public:
    MixtralMLP(DecoderContext *ctx) : norm(ctx) {
        // TODO:
        ctx->sparseExperts = 8;
        for (int i = 0; i < ctx->sparseExperts; ++i) {
            experts.emplace_back(new LlamaMLP<WeiT, InT, ImT, OutT>(ctx));
        }
    }

    ~MixtralMLP() {
        for (auto &expert : experts) {
            delete expert;
        }
    }

    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float * /*unused*/, const OriWeiT *upW, const float *upS, const float *upZ, const float * /*unused*/,
            const float *normW, const float * /*unused*/, const OriWeiT *downW, const float *downS, const float *downZ,
            const float *downB, bool trans = true) {
        xft::Logger::error("Cannot use the original setWeights function in MixtralMLP.");
        exit(-1);
    }

    template <typename WType>
    void setWeights(DecoderContext *ctx, FFNParams *ffnParams) {
        MixtralFFNParams *ffn = dynamic_cast<MixtralFFNParams *>(ffnParams);
        if (ffn == nullptr) {
            xft::Logger::error("Cannot cast FFNParams to MixtralFFNParams.");
            exit(-1);
        }

        this->norm.setWeight(ffn->norm.gamma, nullptr, ctx->hiddenSize);

        prepareGateWeight(ctx, (WType *)ffn->gating.weight);

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
        TimeLine t("MixtralMLP");

        const int M = totInSeqLen == 0 ? ctx->batchSize * ctx->inputSeqLen : totInSeqLen;
        const int hiddenSize = ctx->hiddenSize;
        const int expertNum = ctx->sparseExperts;

        static_assert(sizeof(ctx->normBuf.Data()[0]) >= sizeof(ImT), "normBuff is not big enough!");

        xft::Matrix<InT> inBuffer(input, M, hiddenSize, iStride);
        xft::Matrix<OutT> outBuffer(output, M, hiddenSize, oStride);
        xft::Matrix<ImT> normBuffer(
                (ImT *)ctx->normBuf.Data(), ctx->normBuf.Rows(), ctx->normBuf.Cols(), ctx->normBuf.Stride());

#ifdef XFT_DEBUG
        dbg.debugPrint("MLP Input: [%d, %d] (%d)\n", inBuffer.Rows(), inBuffer.Cols(), inBuffer.Stride());
        dbg.dumpMatrix(inBuffer);
#endif

        // Copy if input and output are different, so that the residual connection can be done
        if ((void *)input != (void *)output) {
#pragma omp parallel for
            for (int i = 0; i < M; ++i) {
                xft::copy(output + i * oStride, input + i * iStride, hiddenSize);
            }
        }

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

        // TODO: special case, we could do some optimization here
        if (M == 1) {}

        // Select top 2 experts for each token
        int *selExperts = reinterpret_cast<int *>(ctx->imOut.Data());
        float *expertWeight = reinterpret_cast<float *>(selExperts + M * 2);
        selectTop2Experts(gateLogits, M, expertNum, selExperts, expertWeight);
#ifdef XFT_DEBUG
        dbg.debugPrint(
                "Selected experts: [%d]=%f [%d]=%f\n", selExperts[0], expertWeight[0], selExperts[1], expertWeight[1]);
#endif

        std::vector<int> idx[expertNum]; // index for each expert
        std::vector<float> weights[expertNum]; // weight for each expert
        for (int i = 0; i < M; ++i) {
            auto topExpert1 = selExperts[2 * i];
            auto topExpert2 = selExperts[2 * i + 1];
            idx[topExpert1].push_back(i);
            idx[topExpert2].push_back(i);
            weights[topExpert1].push_back(expertWeight[i * 2]);
            weights[topExpert2].push_back(expertWeight[i * 2 + 1]);
        }

        // Call forward function of selected experts
        for (int i = 0; i < expertNum; ++i) {
            size_t rowNum = idx[i].size();
            if (idx[i].empty()) { continue; }

            // Gather input for expert i
            ImT *expertData = ctx->getBuffer<ImT>("expertData", rowNum * hiddenSize, ctx->device);
            gatherInput(expertData, hiddenSize, normBuffer.Data(), normBuffer.Stride(), idx[i]);

#ifdef XFT_DEBUG
            dbg.debugPrint("Expert %d, input:\n", i);
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
    }

private:
    template <typename SrcType>
    void prepareGateWeight(DecoderContext *ctx, const SrcType *gateW) {
        int M = ctx->hiddenSize;
        int N = ctx->sparseExperts;

        using GateType = typename GateTypeSelector<WeiT>::type;
        Matrix<GateType> tmpW;
        tmpW.Resize(M, N, N);

        xft::copy(tmpW.Data(), gateW, M * N);

        ctx->mmHelper->packWeight(false, tmpW, gatingWeight);
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

    // logits: [M, N]
    template <typename T>
    void selectTop2Experts(T *logits, int M, int N, int *selExperts, float *expertWeight) {
#pragma omp parallel for
        for (int i = 0; i < M; i += 8) {
            // Process 8 tokens at a time
            if (i + 7 < M) {
#pragma unroll
                for (int t = 0; t < 8; ++t) {
                    selectTop2Experts(logits + (i + t) * N, N, selExperts + (i + t) * 2, expertWeight + (i + t) * 2);
                }
                auto w = xft::load_avx512(expertWeight + i * 2);
                float maxW = _mm512_reduce_max_ps(w);
                auto wExp = BertUtil::vexp(w - _mm512_set1_ps(maxW));
                // Shuffle the vector to get pairs
                __m512 shuffled = _mm512_permute_ps(wExp, _MM_SHUFFLE(2, 3, 0, 1));
                // Add the pairs
                __m512 sum = _mm512_add_ps(wExp, shuffled);
                // Normalize the weights and store
                xft::store_avx512(expertWeight + i * 2, 0xffff, _mm512_div_ps(wExp, sum));
            } else {
                // Process the remaining tokens
                while (i < M) {
                    selectTop2Experts(logits + i * N, N, selExperts + i * 2, expertWeight + i * 2);
                    auto maxW = std::max(expertWeight[i * 2], expertWeight[i * 2 + 1]);
                    auto wExp1 = std::exp(expertWeight[i * 2] - maxW);
                    auto wExp2 = std::exp(expertWeight[i * 2 + 1] - maxW);
                    expertWeight[i * 2] = wExp1 / (wExp1 + wExp2);
                    expertWeight[i * 2 + 1] = wExp2 / (wExp1 + wExp2);
                    ++i;
                }
            }
        }
    }

    // Select top 2 experts for one token
    template <typename T>
    void selectTop2Experts(T *logits, int N, int *selExperts, float *expertWeight) {
        float max1 = -std::numeric_limits<float>::infinity();
        float max2 = -std::numeric_limits<float>::infinity();
        int idx1 = -1;
        int idx2 = -1;
        for (int j = 0; j < N; ++j) {
            float val = logits[j];
            if (val > max1) {
                max2 = max1;
                idx2 = idx1;
                max1 = val;
                idx1 = j;
            } else if (val > max2) {
                max2 = val;
                idx2 = j;
            }
        }
        // Select top 2 experts
        selExperts[0] = idx1;
        selExperts[1] = idx2;
        // Record expert weights (not calculated yet)
        expertWeight[0] = max1;
        expertWeight[1] = max2;
    }

private:
    xft::RmsNorm norm;
    xft::Matrix<typename GateTypeSelector<WeiT>::type> gatingWeight;
    std::vector<LlamaMLP<WeiT, InT, ImT, OutT> *> experts;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};