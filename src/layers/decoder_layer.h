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

#include "attention.h"
#include "debugger.h"
#include "timeline.h"

template <typename ATTN_CLS, typename MLP_CLS>
class Decoder {
public:
    Decoder(DecoderContext *_ctx, int _layerIdx)
        : ctx(_ctx)
        , layerIdx(_layerIdx)
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

    DecoderContext *getCtx() { return this->ctx; }

    int getLayerId() { return layerIdx; }

    void setWeights(std::vector<float *> &params, bool trans = true) {
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

    template <typename KVCacheT>
    void forwardAttention(float *input, float *output, const float *attnMask, KVCacheT *presentKeys,
            KVCacheT *presentValues, int inputSeqLen, int pastSeqLen, bool useSelfAttn, bool doLnBefore,
            bool returnAttn, bool returnKVs, bool forPT = true, int *positionIds = nullptr) {
        TimeLine t("Decoder.forwardAttention");
        attn.forward(ctx, input, output, attnMask, presentKeys, presentValues, inputSeqLen, pastSeqLen, useSelfAttn,
                doLnBefore, returnAttn, returnKVs, forPT, positionIds);
    }

    void forwardFFN(float *input, float *output, int iStride, int oStride, bool doLnBefore = true) {
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
    DecoderContext *ctx;

    // For debug usage
    int layerIdx;

    ATTN_CLS attn;
    MLP_CLS mlp;

#ifdef DEBUG
    Debugger dbg;
#endif
};
