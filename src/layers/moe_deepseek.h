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
class DeepSeekMoE {
public:
    DeepSeekMoE(DecoderContext *ctx) : norm(ctx) {
        //dense mlp or concatted all shared experts
        shared_expert = new LlamaMLP<WeiT, InT, ImT, OutT>(ctx);
        // routed experts
        for (int i = 0; i < ctx->sparseExperts; ++i) {
            experts.emplace_back(new LlamaMLP<WeiT, InT, ImT, OutT>(ctx));
        }
    }

    ~DeepSeekMoE() {
        delete shared_expert;

        for (auto &expert : experts) {
            delete expert;
        }
    }

    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float * /*unused*/, const OriWeiT *upW, const float *upS, const float *upZ, const float * /*unused*/,
            const float *normW, const float * /*unused*/, const OriWeiT *downW, const float *downS, const float *downZ,
            const float *downB, bool trans = true) {
        xft::Logger::error("Cannot use the original setWeights function in DeepSeekMoE.");
        exit(-1);
    }

    template <typename WType>
    void setWeights(DecoderContext *ctx, FFNParams *ffnParams) {
        DeepSeekFFNParams *ffn = dynamic_cast<DeepSeekFFNParams *>(ffnParams);
        if (ffn == nullptr) {
            xft::Logger::error("Cannot cast FFNParams to DeepSeekFFNParams.");
            exit(-1);
        }

        this->norm.setWeight(ffn->norm.gamma, nullptr, ctx->hiddenSize);

        prepareGateWeightBias(ctx, (WType *)ffn->gating.weight, (float *)ffn->gating.bias);

        // setWeights for mlp layer, mlp in firstKDenseReplace, moe for the rest
        if (ffn->mlp.down.weight != nullptr && ffn->mlp.up.weight != nullptr && ffn->mlp.gate.weight != nullptr) {
            shared_expert->template setWeights<WType>(ctx, ffn->mlp);
        } else {
            shared_expert->template setWeights<WType>(ctx, ffn->sharedExpert);
        }

        for (int i = 0; i < ffn->routedExperts.size(); ++i) {
            experts[i]->template setWeights<WType>(ctx, ffn->routedExperts[i]);
        }

        // Check the size of dense expert, im size is intermediateSize or moeIntermediateSize * n_shared_experts
        if (this->experts.size() > 0 && ffn->sharedExpert.down.input_dim != ctx->moeIntermediateSize * ctx->denseExperts) {
            xft::Logger::error("The im size of dense(shared) expert is not consistent. %d %d", ffn->sharedExpert.down.input_dim,
                ctx->moeIntermediateSize, ctx->denseExperts);
            exit(-1);
        }

        // Check the size of sparse experts (loading params vs init params)
        if (ffn->routedExperts.size() != this->experts.size()) {
            xft::Logger::error("The number of experts is not consistent. %d %d (config %d)", ffn->routedExperts.size(),
                    this->experts.size(), ctx->sparseExperts);
            exit(-1);
        }
    }

#ifdef XFT_DEBUG
    void setDebugger(const Debugger &debugger) {
        this->dbg = debugger;
        shared_expert->setDebugger(debugger);
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
        TimeLine t("DeepSeekMoE");

        const int M = totInSeqLen == 0 ? ctx->batchSize * ctx->inputSeqLen : totInSeqLen;
        const int hiddenSize = ctx->hiddenSize;
        const int expertNum = this->experts.size();

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

        shared_expert->forward(ctx, normBuffer.Data(), output, hiddenSize, hiddenSize, false, M, true);
        // first k dense replace
        if (expertNum == 0) { return; }

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

        // computing gating scores
        assert((ctx->scoringFunc) == "sigmoid");
        computingSigmoidWithBias(gateLogits, M, expertNum, gatingScoreCorrBias.Data());

        // topK method: noaux_tc
        assert((ctx->topkMethod) == "noaux_tc");
        int nGroups = ctx->nGroup;
        int topkGroup = ctx->topkGroup;
        int topkExpert = ctx->numExpertsPerTok;

        // 1. Select top 2 experts and sum-up for each group of tokens [M, n_group]
        float *groupWeight = reinterpret_cast<float *>(ctx->imOut.Data());
        scoresGroupExperts(gateLogits, M, expertNum, nGroups, groupWeight);

        // 2. Select top 4 groups for each token [M, topk_group]
        int *selGroups = reinterpret_cast<int *>(groupWeight + M * nGroups);
        selectTopKGroups(groupWeight, M, nGroups, selGroups, topkGroup);

        // 3. Select top 8 experts in selected 4 groups for each token idx-> [M, topk_expert] wei-> [M, topk_expert]
        int *selExperts = reinterpret_cast<int *>(selGroups + M * topkGroup);
        float *expertWeight = reinterpret_cast<float *>(selExperts + M * topkExpert);
        maskedSelectTopKExperts(gateLogits, M, expertNum, nGroups, selGroups, topkGroup, selExperts, expertWeight, topkExpert);

        // if ctx->normTopKProb is true, we need to normalize the expertWeight, so that they sum 1
        scaleNormTopKExpertsWeight(expertWeight, M, topkExpert, ctx->numExpertsPerTok > 1 && ctx->normTopKProb, ctx->routedScalingFac);

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
    void prepareGateWeightBias(DecoderContext *ctx, const SrcType *gateW, const float *gateB) {
        int M = ctx->hiddenSize;
        int N = ctx->sparseExperts;

        using GateType = typename GateTypeSelector<WeiT>::type;
        Matrix<GateType> tmpW;
        tmpW.Resize(M, N, N);
        xft::copy(tmpW.Data(), gateW, M * N);
        ctx->mmHelper->packWeight(false, tmpW, gatingWeight);

        gatingScoreCorrBias.Resize(N);
        xft::copy(gatingScoreCorrBias.Data(), gateB, N);
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
    void computingSigmoidWithBias(T *logits, int M, int N, float *scoreCorrBias) {
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            // compute sigmoid, 1.0 / (1.0 + exp(-x)) + scoreCorrBias
            int vecSize = 512 / sizeof(T);
            __m512 v1 = _mm512_set1_ps(1.0f);
            __m512 vzero = _mm512_set1_ps(0.0f);
            for (int j = 0; j < N; j += vecSize) {
                auto v = xft::load_avx512(logits + i * N + j);
                __m512 neg = _mm512_sub_ps(vzero, v);
                __m512 exp = BertUtil::vexp(neg);
                __m512 sgmd = _mm512_div_ps(v1, _mm512_add_ps(v1, exp));
                if (scoreCorrBias != nullptr) {
                    sgmd = _mm512_add_ps(sgmd, _mm512_set1_ps(scoreCorrBias[j]));
                }
                xft::store_avx512(logits + i * N + j, 0xffff, sgmd);
	    }
        }
    }

    // logits: [M, N]
    // output: [M, n_group]
    template <typename T>
    void scoresGroupExperts(T *logits, int M, int N, int nGroups, float *groupWeight) {
        int groupSize = N / nGroups;
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < nGroups; ++j) {
                sumTop2ExpertsInGroup(logits + i * N + j * groupSize, groupSize, groupWeight + i * nGroups + j);
            }
        }
    }

    template <typename T>
    void selectTopKGroups(T *groupWeight, int M, int nGroups, int *selGroups, int topkGroup) {
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            topK(groupWeight + i * nGroups, nGroups, topkGroup, selGroups + i * topkGroup, nullptr);
        }
    }

    template <typename T>
    void maskedSelectTopKExperts(T *gateLogits, int M, int N, int nGroups, int *selGroups, int topkGroup, int *selExperts,
            float *expertWeight, int topkExpert) {
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            topKMasked(gateLogits + i * N, N, topkExpert, selGroups + i * topkGroup, nGroups, selExperts + i * topkExpert,
                expertWeight + i * topkExpert);
        }
    }

    void scaleNormTopKExpertsWeight(float *expertWeight, int M, int topkExpert, bool normProb, float routedScalingFac) {
        // topkExpert is 8 (commonly less than 16), so use avx512 to speed up the normalization to make them sum 1
        int vecSize = 512 / sizeof(float);
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            __m512 vscale = _mm512_set1_ps(routedScalingFac);
            if (normProb) {
                __m512 sum = _mm512_set1_ps(0.0f);
                for (int j = 0; j < topkExpert; j += vecSize) {
                    auto v = xft::load_avx512(expertWeight + i * topkExpert + j);
                    sum = _mm512_add_ps(sum, v);
                }
                // add a small value to avoid div 0
                vscale = _mm512_div_ps(vscale, sum + _mm512_set1_ps(1e-20f));
            }
            for (int j = 0; j < topkExpert; j += vecSize) {
                auto v = xft::load_avx512(expertWeight + i * topkExpert + j);
                xft::store_avx512(expertWeight + i * topkExpert + j, 0xffff, _mm512_mul_ps(v, vscale));
            }
        }
    }

    // Select top 2 experts in one group for one token
    template <typename T>
    void sumTop2ExpertsInGroup(T *logits, int N, float *groupWeight) {
        float max1 = -std::numeric_limits<float>::infinity();
        float max2 = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
            float val = logits[j];
            if (val > max1) {
                max2 = max1;
                max1 = val;
            } else if (val > max2) {
                max2 = val;
            }
        }
        *groupWeight = max1 + max2;
    }

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
            if (selWeight != nullptr)
                selWeight[i] = vec[i].first;
        }
    }

    // the main difference between this topKMasked and topK is that this function will ignore the elements that are not in selGroups
    template <typename T>
    void topKMasked(T *array, int N, int topk, int *selGroups, int nGroups, int *selIdx, float *selWeight) {
        // groupId for each element is i / groupSize
        std::vector<std::pair<T, int>> vec;
        int groupSize = N / nGroups;
        for (int i = 0; i < N; ++i) {
            if (std::find(selGroups, selGroups + nGroups, i / groupSize) != selGroups + nGroups) {
                vec.emplace_back(array[i], i);
            }
        }
        std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<T, int>>());
        for (int i = 0; i < topk; ++i) {
            selIdx[i] = vec[i].second;
            if (selWeight != nullptr)
                selWeight[i] = vec[i].first;
        }
    }

private:
    xft::RmsNorm norm;
    xft::Matrix<typename GateTypeSelector<WeiT>::type> gatingWeight;
    xft::Vector<float> gatingScoreCorrBias;
    LlamaMLP<WeiT, InT, ImT, OutT> *shared_expert;
    std::vector<LlamaMLP<WeiT, InT, ImT, OutT> *> experts;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};
