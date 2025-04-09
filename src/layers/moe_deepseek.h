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
#include "copy_util.h"
#include "debugger.h"
#include "llm_params.h"
#include "mlp_llama.h"
#include "type_selector.h"
#include "timeline.h"

template <typename WeiT, typename InT = bfloat16_t, typename ImT = bfloat16_t, typename OutT = bfloat16_t>
class DeepSeekMoE {
public:
    DeepSeekMoE(int layerId, DecoderContext *ctx) : layerId(layerId), norm(ctx) {
        //dense mlp or concatted all shared experts
        shared_expert = new LlamaMLP<WeiT, InT, ImT, OutT>(layerId, ctx);
        if (layerId >= ctx->firstKDenseReplace) {
            for (int i = 0; i < ctx->sparseExperts; ++i)
                experts.emplace_back(new LlamaMLP<WeiT, InT, ImT, OutT>(layerId, ctx));
        }
    }

    ~DeepSeekMoE() {
        delete shared_expert;

        for (auto &expert : experts) {
            delete expert;
        }
    }

    static xft::DataType getWeightDataType() { return xft::getDataType<WeiT>(); }

    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float * /*unused*/, const OriWeiT *upW, const float *upS, const float *upZ, const float * /*unused*/,
            const float *normW, const float * /*unused*/, const OriWeiT *downW, const float *downS, const float *downZ,
            const float *downB, bool trans = true) {
        xft::Logger::error("Cannot use the original setWeights function in DeepSeekMoE.");
        exit(-1);
    }

    template <typename WType>
    void setWeights(DecoderContext *ctx, xft::FFNParams *ffnParams) {
        xft::DeepSeekFFNParams *ffn = dynamic_cast<xft::DeepSeekFFNParams *>(ffnParams);
        if (ffn == nullptr) {
            xft::Logger::error("Cannot cast FFNParams to DeepSeekFFNParams.");
            exit(-1);
        }

        const int expertNum = ffn->routedExperts.size();

        // setWeights for mlp layer, mlp in firstKDenseReplace, moe for the rest
        // ffn->routedExperts.size() == 0 means the firstKDenseReplace is used
        if (layerId < ctx->firstKDenseReplace || expertNum == 0) {
            memcpy(ffn->mlp.norm.gamma, ffn->norm.gamma, sizeof(float) * ffn->norm.hidden_size);
            shared_expert->template setWeights<WType>(ctx, &ffn->mlp);
        } else {
            prepareGateWeightBias(ctx, &(ffn->gating));
            if (ctx->denseExperts > 0) {
                this->norm.setWeight(ffn->norm.gamma, nullptr, ctx->hiddenSize);
                memcpy(ffn->sharedExpert.norm.gamma, ffn->norm.gamma, sizeof(float) * ffn->norm.hidden_size);
                shared_expert->template setWeights<WType>(ctx, &ffn->sharedExpert);
            }
            // setWeights for each expert
#pragma omp parallel for num_threads(std::min(omp_get_max_threads(), 64))
            for (int i = 0; i < expertNum; ++i) {
                experts[i]->template setWeights<WType>(ctx, ffn->routedExperts[i]);
            }
        }

        // Check the size of dense expert, im size is intermediateSize or moeIntermediateSize * n_shared_experts
        if (this->experts.size() > 0 && ffn->sharedExpert.down.input_dim != ctx->moeIntermediateSize * ctx->denseExperts) {
            xft::Logger::error("The im size of dense(shared) expert is not consistent. %d %d", ffn->sharedExpert.down.input_dim,
                ctx->moeIntermediateSize, ctx->denseExperts);
            exit(-1);
        }

        // Check the size of sparse experts (loading params vs init params)
        if (layerId >= ctx->firstKDenseReplace && ffn->routedExperts.size() != this->experts.size()) {
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

        {
            TimeLine t("MoE_Copy_Residual");
            // Copy if input and output are different, so that the residual connection can be done
            if ((void *)input != (void *)output) {
#pragma omp parallel for
                for (int i = 0; i < M; ++i) {
                    xft::copy(output + i * oStride, input + i * iStride, hiddenSize);
                }
            }
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("Output init: \n");
        dbg.dumpMatrix(output, M, hiddenSize, hiddenSize);
#endif

        {
            TimeLine t("MoE_SharedExpert");
            shared_expert->forward(ctx, input, output, iStride, oStride, doLnBefore, M, false);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("Output mlp/shared: \n");
        dbg.dumpMatrix(output, M, hiddenSize, hiddenSize);
#endif
        if (expertNum == 0) {
            return;
        }
        {
            TimeLine t("MoE_Normalize");
            // Normalize input
            if (doLnBefore == true) {
                norm.forward(inBuffer.Data(), normBuffer.Data(), M, inBuffer.Stride(), normBuffer.Stride(), 1e-6);
#ifdef XFT_DEBUG
                dbg.debugPrint("Norm: \n");
                dbg.dumpMatrix(normBuffer);
#endif
            }
        }

        ImT *normBuf = (doLnBefore ? normBuffer.Data() : inBuffer.Data());
        int normStride = (doLnBefore ? normBuffer.Stride() : inBuffer.Stride());

        // Gating
        OutT *gateLogits = ctx->getBuffer<OutT>("gateLogits", M * expertNum, ctx->device);
        {
            TimeLine t("MoE_Gating");
            ctx->mmHelper->compute(false, M, expertNum, hiddenSize, 1.0f, normBuf, normStride, gatingWeight.Data(),
                nullptr, nullptr, nullptr, 0.0f, gateLogits, expertNum);
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("Gate: \n");
        dbg.dumpMatrix(gateLogits, M, expertNum, expertNum);
#endif

        {
            TimeLine t("MoE_Sigmoid");
            // computing gating scores
            assert((ctx->scoringFunc) == "sigmoid");
            // TODO: fused with last compute MatMul
            computingSigmoid(gateLogits, M, expertNum);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("gateLogits sigmoid: \n");
        dbg.dumpMatrix(gateLogits, M, expertNum, expertNum);
#endif

        forwardExpertsWithLogits(ctx, normBuf, output, M, normStride, oStride, gateLogits);
    }


    void forwardExpertsWithLogits(DecoderContext *ctx, ImT *normBuf, OutT *output, int M, int normStride, int oStride, OutT *gateLogits) {
        int topkExpert = ctx->numExpertsPerTok;
        int *selExperts = ctx->getBuffer<int>("selExperts", M * topkExpert, ctx->device);
        float *expertWeight = ctx->getBuffer<float>("expertWeight", M * topkExpert, ctx->device);
        if (ctx->topkMethod == "noaux_tc") {
            topkNoauxTc(ctx, M, gateLogits, selExperts, expertWeight);
        } else {
            xft::Logger::error("Unsupported topk method: %s", ctx->topkMethod.c_str());
            exit(-1);
        }
        forwardExperts(ctx, normBuf, output, M, normStride, oStride, selExperts, expertWeight);
    }

    void topkNoauxTc(DecoderContext *ctx, int M, OutT *gateLogits, int *selExperts, float *expertWeight) {
        const int hiddenSize = ctx->hiddenSize;
        const int expertNum = this->experts.size();
        int nGroups = ctx->nGroup;
        int topkGroup = ctx->topkGroup;
        int topkExpert = ctx->numExpertsPerTok;

        // topK method: noaux_tc
        assert((ctx->topkMethod) == "noaux_tc");
        // 1. Select top 2 experts and sum-up for each group of tokens [M, n_group]
        float *groupWeight = ctx->getBuffer<float>("groupWeight", M * nGroups, ctx->device);
        {
            TimeLine t("MoE_GroupScore");
            scoresGroupExperts(gateLogits, M, expertNum, nGroups, groupWeight, gatingScoreCorrBias.Data());
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("Group scores: \n");
        dbg.dumpMatrix(groupWeight, M, nGroups, nGroups);
#endif

        // 2. Select top 4 groups for each token [M, topk_group]
        int *selGroups = ctx->getBuffer<int>("selGroups", M * topkGroup, ctx->device);
        {
            TimeLine t("MoE_SelectGroup");
            selectTopKGroups(groupWeight, M, nGroups, selGroups, topkGroup);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("seleted Group: \n");
        dbg.dumpMatrix(selGroups, M, topkGroup, topkGroup);
#endif

        // 3. Select top 8 experts in selected 4 groups for each token idx-> [M, topk_expert] wei-> [M, topk_expert]
        {
            TimeLine t("MoE_SelectExperts");
            maskedSelectTopKExperts(gateLogits, M, expertNum, nGroups, selGroups, topkGroup, selExperts, expertWeight, topkExpert);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("seleted Experts & Weight: \n");
        dbg.dumpMatrix(selExperts, M, topkExpert, topkExpert);
        dbg.dumpMatrix(expertWeight, M, topkExpert, topkExpert);
#endif

        {
            TimeLine t("MoE_ScaleNorm");
            // 4. if ctx->normTopKProb is true, Normalize the expertWeight, so that they sum 1
            scaleNormTopKExpertsWeight(expertWeight, M, topkExpert, ctx->numExpertsPerTok > 1 && ctx->normTopKProb, ctx->routedScalingFac);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("scaledNorm seleted Experts & Weight: \n");
        dbg.dumpMatrix(selExperts, M, topkExpert, topkExpert);
        dbg.dumpMatrix(expertWeight, M, topkExpert, topkExpert);
#endif

#ifdef XFT_DEBUG
        dbg.debugPrint(
            "Selected experts: [%d]=%f [%d]=%f\n", selExperts[0], expertWeight[0], selExperts[1], expertWeight[1]);
#endif
    }

    void forwardExperts(DecoderContext *ctx, ImT *normBuf, OutT *output, int M, int normStride, int oStride, int *selExperts, float *expertWeight) {
        const int hiddenSize = ctx->hiddenSize;
        const int expertNum = this->experts.size();
        int topkExpert = ctx->numExpertsPerTok;

        // Call forward function of selected experts
        // expert-wise for large M or bf16 for now
        if (M > 128 || std::is_same_v<WeiT, bfloat16_t> || Env::getInstance().getMoEEngine() == 0) {
            // 5. Reorder the input and weight for each expert
            std::vector<int> idx[expertNum]; // index for each expert
            std::vector<float> weights[expertNum]; // weight for each expert
            {
                TimeLine t("MoE_Reorder");
                for (int i = 0; i < M; ++i) {
                    // fill idx and weights for each expert
                    for (int j = 0; j < topkExpert; ++j) {
                        if (selExperts[i * topkExpert + j] < 0) break;
                        idx[selExperts[i * topkExpert + j]].push_back(i);
                        weights[selExperts[i * topkExpert + j]].push_back(expertWeight[i * topkExpert + j]);
                    }
                }
            }
            // call forward for each expert
            for (int i = 0; i < expertNum; ++i) {
                size_t rowNum = idx[i].size();
                if (idx[i].empty()) { continue; }

                TimeLine t("MoE_RoutedExpert");
                // Gather input for expert i
                ImT *expertData = ctx->getBuffer<ImT>("expertData", rowNum * hiddenSize, ctx->device);
                gatherInput(expertData, hiddenSize, normBuf, normStride, idx[i]);

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

                // Scatter output of expert i (critical section)
                scatterOutput(output, oStride, expertData, hiddenSize, idx[i], weights[i]);
            }
        } else if (Env::getInstance().getMoEEngine() == 1) {
            // call sparse mlp for each token
            for (int i = 0; i < M; ++i) {
                TimeLine t("MoE_TokenSparseFW");
                OutT *tokenData = ctx->getBuffer<OutT>("tokenData", 1 * hiddenSize, ctx->device);
                int nExperts = 0;
                for (int j = 0; j < topkExpert; ++j) {
                    if (selExperts[i * topkExpert + j] < 0) break;
                    ++nExperts;
                }
                sparseForward(ctx, normBuf + i * normStride, selExperts + i * topkExpert, expertWeight + i * topkExpert,
                    nExperts, tokenData, hiddenSize, output + i * oStride, oStride);
            }
        } else {
            xft::Logger::error("Unsupported MoE engine: %d", Env::getInstance().getMoEEngine());
            exit(-1);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("Output:\n");
        dbg.dumpMatrix(output, M, hiddenSize, hiddenSize);
#endif
    }

private:
    void prepareGateWeightBias(DecoderContext *ctx, xft::DenseLayerParams *denseParams) {
        int M = ctx->hiddenSize;
        int N = ctx->sparseExperts;

        if (denseParams->weight != nullptr) {
            using GateType = typename GateTypeSelector<WeiT>::type;

            xft::Matrix<GateType> quantizedGatingW;

            //ctx->mmHelper->convertWeight(ctx, false, M, N, (const GateType *)denseParams->weight, denseParams->weight_scale,
            //    denseParams->weight_zp, true, quantizedGatingW, gatingWScale, gatingWZero, gatingWSum);
            quantizedGatingW.Resize(M, N);
            xft::copy(quantizedGatingW.Data(), (const GateType *)denseParams->weight, M * N);
            gatingWeight.Resize(M, N);
            ctx->mmHelper->packWeight(false, quantizedGatingW, gatingWeight);
        }

        if (denseParams->bias != nullptr) {
            gatingScoreCorrBias.Resize(N);
            xft::copy(gatingScoreCorrBias.Data(), (const float *)(denseParams->bias), N);
        }
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
            //#pragma omp critical
            {
                xft::addto(output + idx[i] * oStride, expertData + i * hiddenSize, scale, hiddenSize);
            }
        }
    }

    // logits: [M, N]
    template <typename T>
    void computingSigmoid(T *logits, int M, int N) {
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            // compute sigmoid, 1.0 / (1.0 + exp(-x)) + scoreCorrBias
            __m512 v1 = _mm512_set1_ps(1.0f);
            __m512 vzero = _mm512_set1_ps(0.0f);
            for (int j = 0; j < N; j += 16) {
                int remain = N - j;
                __mmask16 mask = remain >= 16 ? 0xffff : (1 << remain) - 1;
                auto v = xft::load_avx512(mask, logits + i * N + j);
                __m512 neg = _mm512_sub_ps(vzero, v);
                __m512 exp = BertUtil::vexp(neg);
                __m512 sgmd = _mm512_div_ps(v1, _mm512_add_ps(v1, exp));
                xft::store_avx512(logits + i * N + j, mask, sgmd);
            }
        }
    }

    // logits: [M, N]
    // output: [M, n_group]
    template <typename T>
    void scoresGroupExperts(T *logits, int M, int N, int nGroups, float *groupWeight, float *scoreCorrBias) {
        int groupSize = N / nGroups;
#pragma omp parallel for collapse(2)
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < nGroups; ++j) {
                sumTop2ExpertsInGroupWithBias(logits + i * N + j * groupSize, groupSize, groupWeight + i * nGroups + j,
                    scoreCorrBias + j * groupSize);
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
#pragma omp parallel for
        for (int i = 0; i < M; ++i) {
            __m512 vscale = _mm512_set1_ps(routedScalingFac);
            if (normProb) {
                float sum = 0.0f;
                for (int j = 0; j < topkExpert; j += 16) {
                    int remain = topkExpert - j;
                    __mmask16 mask = remain >= 16 ? 0xffff : (1 << remain) - 1;
                    auto v = xft::load_avx512(mask, expertWeight + i * topkExpert + j);
                    sum = sum + _mm512_reduce_add_ps(v);
                }
                // add a small value to avoid div 0
                vscale = _mm512_div_ps(vscale, _mm512_set1_ps(sum + 1e-20f));
            }
            for (int j = 0; j < topkExpert; j += 16) {
                int remain = topkExpert - j;
                __mmask16 mask = remain >= 16 ? 0xffff : (1 << remain) - 1;
                auto v = xft::load_avx512(mask, expertWeight + i * topkExpert + j);
                xft::store_avx512(expertWeight + i * topkExpert + j, mask, _mm512_mul_ps(v, vscale));
            }
        }
    }

    // Select top 2 experts in one group for one token
    template <typename T>
    void sumTop2ExpertsInGroupWithBias(T *logits, int N, float *groupWeight, float *corrBias) {
        float max1 = -std::numeric_limits<float>::infinity();
        float max2 = -std::numeric_limits<float>::infinity();
        int idx1 = -1;
        int idx2 = -1;
        for (int j = 0; j < N; ++j) {
            float val = logits[j];
	    if (corrBias != nullptr)
                val += corrBias[j];
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
        for (int i = 0; i < nGroups; ++i) {
            if (std::find(selGroups, selGroups + nGroups, i) != selGroups + nGroups) {
                for (int j = 0; j < groupSize; ++j)
                    vec.emplace_back(array[i * groupSize + j], i * groupSize + j);
            }
        }
        std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(), std::greater<std::pair<T, int>>());
        for (int i = 0; i < topk; ++i) {
            selIdx[i] = vec[i].second;
            if (selWeight != nullptr)
                selWeight[i] = vec[i].first;
        }
    }

    void sparseForward(DecoderContext *ctx, ImT *input, int *selExperts, float *expertWeight, int nExperts, OutT *tokenData,
            int hiddenSize, OutT *output, int oStride) {
        const WeiT *weightsGUList[nExperts];
        const WeiT *weightsDList[nExperts];
        const float *scalesGUList[nExperts];
        const float *scalesDList[nExperts];
        int ldaGUScales[nExperts];
        int ldaDScales[nExperts];
        int blockSize = 128;
        float alpha[nExperts];
        OutT *imOuts[nExperts];

         // just for 1 token
        int M = 1;

        int K1 = hiddenSize;
        // concat gate and up weights
        int N1 = this->experts[0]->splitSize * 2;

        int K2 = this->experts[0]->splitSize;
        int N2 = hiddenSize;

        {
            TimeLine t("SparseFW_Prepare");
            for (int i = 0; i < nExperts; ++i) {
                weightsGUList[i] = this->experts[selExperts[i]]->catWeights.Data();
                scalesGUList[i] = this->experts[selExperts[i]]->catGUScales.Data();
                ldaGUScales[i] = (K1 + blockSize - 1) / blockSize;
                alpha[i] = 1.0;
                imOuts[i] = ctx->getBuffer<OutT>("sparseImOut_" + std::to_string(i), M * N1, ctx->device);

                weightsDList[i] = this->experts[selExperts[i]]->downWeight.Data();
                ldaDScales[i] = (K2 + blockSize - 1) / blockSize;
                scalesDList[i] = this->experts[selExperts[i]]->downScales.Data();
            }
        }

        int lda1 = hiddenSize;
        int ldc1[nExperts];
        for (int i = 0; i < nExperts; ++i) {
            ldc1[i] = N1;
        }

#ifdef XFT_DEBUG
        dbg.debugPrint("SparseFW_Input (%d %d):\n", 1, hiddenSize);
        dbg.dumpMatrix(input, 1, hiddenSize, lda1);
#endif
        {
            TimeLine t("SparseFW_GateUp");
            ctx->mmHelper->compute_batch_C(M, N1, K1, alpha, input, lda1, weightsGUList, scalesGUList, imOuts, ldc1, ldaGUScales,
                blockSize, nExperts);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("Sparse_GateUp %d x (%d %d):\n", nExperts, lda1, ldc1[0]);
        dbg.dumpMatrix(imOuts[0], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[1], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[2], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[3], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[4], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[5], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[6], 1, N1, ldc1[0]);
        dbg.dumpMatrix(imOuts[nExperts - 1], 1, N1, ldc1[0]);
#endif

        {
            TimeLine t("SparseFW_Silu");
            // Compute silu on the left half and then add it with the right half
            if (ctx->actType == DecoderContext::SILU) {
                DecoderUtil::siluSumBatch(imOuts, imOuts, nExperts, M, N1);
            } else {
                printf("ERROR: unsupported activation in MLP.\n");
                exit(-1);
            }
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("Sparse_Silu %d x (%d %d, %d):\n", nExperts, 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[0], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[1], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[2], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[3], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[4], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[5], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[6], 1, N1 / 2, ldc1[0]);
        dbg.dumpMatrix(imOuts[nExperts - 1], 1, N1 / 2, ldc1[0]);
#endif

        int lda2[nExperts];
        for (int i = 0; i < nExperts; ++i) {
            lda2[i] = ldc1[i];
        }

        int ldc2 = oStride;
        {
            TimeLine t("SparseFW_Down");
            ctx->mmHelper->compute_batch_A(M, N2, K2, expertWeight, (const bfloat16_t**)imOuts, lda2, weightsDList,
                scalesDList, tokenData, ldc2, ldaDScales, blockSize, nExperts);
            xft::addto(output, tokenData, 1.0, hiddenSize);
        }
#ifdef XFT_DEBUG
        dbg.debugPrint("tokenData (%d %d):\n", 1, hiddenSize);
        dbg.dumpMatrix(tokenData, 1, hiddenSize, lda1);
        dbg.debugPrint("Sparse_Down (%d %d):\n", lda2[0], ldc2);
        dbg.dumpMatrix(output, 1, N2, ldc2);
#endif
    }

private:
    xft::RmsNorm norm;
    xft::Matrix<typename GateTypeSelector<WeiT>::type> gatingWeight;
    xft::Vector<float> gatingWScale;
    xft::Vector<float> gatingWZero;
    xft::Vector<float> gatingWSum;
    xft::Vector<float> gatingScoreCorrBias;
    //dense mlp or concatted all shared experts, including norm
    LlamaMLP<WeiT, InT, ImT, OutT> *shared_expert;
    std::vector<LlamaMLP<WeiT, InT, ImT, OutT> *> experts;
    
    int layerId;

#ifdef XFT_DEBUG
    Debugger dbg;
#endif
};
