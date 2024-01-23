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

#include <immintrin.h>
#include <omp.h>

#include <cassert>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>

#include "attention.h"
#include "debugger.h"
#include "kvcache_tensor.h"
#include "timeline.h"
#include "type_selector.h"

// To get weight data type in attention class
template <typename T>
struct AttnTypeExtractor;
template <template <typename...> class ATTN_CLS, typename WeiT, typename QKPO_CLS, typename NORM_CLS>
struct AttnTypeExtractor<ATTN_CLS<WeiT, QKPO_CLS, NORM_CLS>> {
    using Tin = float;
    using Tim = float;
    using Tout = float;
};
template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, typename InT, typename ImT, typename OutT>
struct AttnTypeExtractor<Attention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, true>> {
    using Tin = InT;
    using Tim = ImT;
    using Tout = OutT;
};

template <typename ATTN_CLS, typename MLP_CLS>
class Decoder {
public:
    Decoder(DecoderContext *_ctx, int _layerIdx)
        : layerIdx(_layerIdx)
        , attn(_layerIdx, _ctx)
        , mlp(_ctx)
#ifdef DEBUG
        , dbg(Debugger::formatStr("%d_%d.csv", _layerIdx, _ctx->splitIdx))
#endif
    {
#ifdef DEBUG
        attn.setDebugger(dbg);
        mlp.setDebugger(dbg);
#endif
    }

    virtual ~Decoder() {}

    int getLayerId() { return layerIdx; }

    void setWeights(DecoderContext *ctx, std::vector<float *> &params, bool trans = true) {
        const float *queryWeight = params[0];
        const float *queryBias = params[1];
        const float *keyWeight = params[2];
        const float *keyBias = params[3];
        const float *valueWeight = params[4];
        const float *valueBias = params[5];
        const float *attnOutWeight = params[6];
        const float *attnOutBias = params[7];
        const float *gamma1 = params[8];
        const float *beta1 = params[9];

        attn.setWeights(ctx, queryWeight, queryBias, keyWeight, keyBias, valueWeight, valueBias, attnOutWeight,
                attnOutBias, gamma1, beta1, trans);

        std::vector<float *> mlpParams(params.begin() + 10, params.end());
        mlp.setWeights(ctx, mlpParams, trans);
    }

    template <typename InT, typename ImT, typename OutT, typename KVCacheT>
    void forwardAttention(DecoderContext *ctx, InT *input, ImT *imBuf, OutT *output, const float *attnMask,
            KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen,
            bool useSelfAttn, bool doLnBefore, int *positionIds = nullptr) {
        TimeLine t("Decoder.forwardAttention");

        using Ttarget = typename AttnTypeExtractor<ATTN_CLS>::Tim;
        static_assert(sizeof(ImT) >= sizeof(Ttarget), "Intermediate buffer is NOT big enough!");

        attn.forward(ctx, input, (Ttarget *)imBuf, output, attnMask, presentKey, presentValue, inputSeqLen, pastSeqLen,
                useSelfAttn, doLnBefore, positionIds);
    }

    template <typename InT, typename OutT>
    void forwardFFN(DecoderContext *ctx, InT *input, OutT *output, int iStride, int oStride, bool doLnBefore = true) {
        TimeLine t("Decoder.forwardFFN");
        mlp.forward(ctx, input, output, iStride, oStride, doLnBefore);
    }

private:
    void copyWeights(hpj::Matrix<float> &w, int start_col, int end_col, const float *data) {
        hpj::Matrix<float> subW(w, 0, w.Rows(), start_col, end_col - start_col);
        copyWeights(subW, data);
    }

    // Copy the transposed weight into the non-transposed matrix
    void copyWeights(hpj::Matrix<float> &w, const float *data) {
        for (int j = 0; j < w.Cols(); ++j) {
            for (int i = 0; i < w.Rows(); ++i) {
                w(i, j) = *data++;
            }
        }
    }

    void copyTransposed(hpj::Matrix<float> &dst, hpj::Matrix<float> &src) {
        dst.Resize(src.Cols(), src.Rows());
        for (int i = 0; i < dst.Rows(); ++i) {
            for (int j = 0; j < dst.Cols(); ++j) {
                dst(i, j) = src(j, i);
            }
        }
    }

    // Add bias to matrix
    void biasAdd(hpj::Matrix<float> &m, hpj::Vector<float> &bias) {
        float *pbias = bias.Data();
#pragma omp parallel for
        for (int i = 0; i < m.Rows(); ++i) {
            float *p = m.Row(i);
#pragma omp simd
            for (int j = 0; j < m.Cols(); ++j) {
                p[j] += pbias[j];
            }
        }
    }

private:
    // For debug usage
    int layerIdx;

    ATTN_CLS attn;
    MLP_CLS mlp;

#ifdef DEBUG
    Debugger dbg;
#endif
};
