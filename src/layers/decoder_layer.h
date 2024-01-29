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
#include "dtype.h"
#include "kvcache_tensor.h"
#include "timeline.h"
#include "type_selector.h"

// To get weight data type in attention class
template <typename T>
struct AttnParameters;
template <template <typename...> class ATTN_CLS, typename WeiT, typename QKPO_CLS, typename NORM_CLS>
struct AttnParameters<ATTN_CLS<WeiT, QKPO_CLS, NORM_CLS>> {
    using Tim = float;
};
template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, typename InT, typename ImT, typename OutT>
struct AttnParameters<Attention<WeiT, QKPO_CLS, NORM_CLS, InT, ImT, OutT, true>> {
    using Tim = ImT;
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

    // SrcT: float or int8_t
    template <typename SrcT>
    void setWeights(DecoderContext *ctx, const SrcT *queryWeight, const float *queryScale, const float *queryZero,
            const float *queryBias, const SrcT *keyWeight, const float *keyScale, const float *keyZero,
            const float *keyBias, const SrcT *valueWeight, const float *valueScale, const float *valueZero,
            const float *valueBias, const SrcT *attnOutWeight, const float *attnOutScale, const float *attnOutZero,
            const float *attnOutBias, const float *ln1Gamma, const float *ln1Beta, const SrcT *fc1Weight,
            const float *fc1Scales, const float *fc1Zeros, const float *fc1Bias, const SrcT *fc2Weight,
            const float *fc2Scales, const float *fc2Zeros, const float *fc2Bias, const float *ln2Gamma,
            const float *ln2Beta, const SrcT *fc3Weight, const float *fc3Scales, const float *fc3Zeros,
            bool trans = true) {
        attn.setWeights(ctx, queryWeight, queryScale, queryZero, queryBias, keyWeight, keyScale, keyZero, keyBias,
                valueWeight, valueScale, valueZero, valueBias, attnOutWeight, attnOutScale, attnOutZero, attnOutBias,
                ln1Gamma, ln1Beta, trans);

        mlp.setWeights(ctx, fc1Weight, fc1Scales, fc1Zeros, fc1Bias, fc2Weight, fc2Scales, fc2Zeros, fc2Bias, ln2Gamma,
                ln2Beta, fc3Weight, fc3Scales, fc3Zeros, trans);
    }

    template <typename InT, typename ImT, typename OutT, typename KVCacheT>
    void forwardAttention(DecoderContext *ctx, InT *input, ImT *imBuf, OutT *output, const float *attnMask,
            KVCacheTensor<KVCacheT> &presentKey, KVCacheTensor<KVCacheT> &presentValue, int inputSeqLen, int pastSeqLen,
            bool useSelfAttn, bool doLnBefore, int *positionIds = nullptr) {
        TimeLine t("Decoder.forwardAttention");

        using Ttarget = typename AttnParameters<ATTN_CLS>::Tim;
        static_assert(sizeof(ImT) >= sizeof(Ttarget), "Intermediate buffer is NOT big enough!");

        attn.forward(ctx, input, (Ttarget *)imBuf, output, attnMask, presentKey, presentValue, inputSeqLen, pastSeqLen,
                useSelfAttn, doLnBefore, positionIds);
    }

    void forwardFFN(
            DecoderContext *ctx, float *input, float *output, int iStride, int oStride, bool doLnBefore = true) {
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
