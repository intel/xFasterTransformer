#pragma once
#include <numeric>
#ifdef USE_MKLML
#include <mkl.h>
#endif

#include "bfloat16.h"
#include "debugger.h"
#include "decoder_util.h"
#include "float16.h"
#include "gemm_kernel_ext.h"
#include "hgemm_f32f16f32_simple.h"
#include "matmul_helper.h"
#include "sgemm_f32f16f32_simple.h"
#include "sgemm_simple.h"
#include "transformer_ctx.h"
#include "transformer_util.h"

// WeiT: weight data type
// QKPO_CLS: class for post operation of query/key, it is generally the rotary embedding
// NORM_CLS: class for layernorm or other norms
// INPUT_AS_RESID: input as residential or not, most models use input as residential,
//                 but there are exceptions like ChatGLM use values after layernorm as residential
template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, bool INPUT_AS_RESID = true>
class Attention {
public:
    Attention(int layerId, DecoderContext *ctx) : layerId(layerId), qkpo(ctx->attHeadSize) {}

    // The inerface is for PyTorch, thus the weights are already transposed
    void setWeights(DecoderContext *ctx, const float *_queryWeight, const float *_queryBias, const float *_keyWeight,
            const float *_keyBias, const float *_valueWeight, const float *_valueBias, const float *_attnOutputWeight,
            const float *_attnOutputBias, const float *_gamma1, const float *_beta1, bool trans = true) {
        int hiddenSize = ctx->hiddenSize;
        int q_hiddenSize = ctx->hiddenSize;
        int k_hidden_size = ctx->attHeadSize * ctx->kvHeadNum;
        int v_hidden_size = ctx->attHeadSize * ctx->kvHeadNum;
        int qkv_hidden_size = q_hiddenSize + k_hidden_size + v_hidden_size;
        // Merged weights, dimension is like: 768*(768 + k_hidden_size + v_hidden_size)
        // Vertically split the QKV weights
        int q_colsPerSplit = q_hiddenSize / ctx->numSplit;
        int k_colsPerSplit = k_hidden_size / ctx->numSplit;
        int v_colsPerSplit = v_hidden_size / ctx->numSplit;
        int colsPerSplit = q_colsPerSplit + k_colsPerSplit + v_colsPerSplit;
        qkvWeight.Resize(hiddenSize, colsPerSplit);

        float *concatBuf = (float *)malloc(hiddenSize * colsPerSplit * sizeof(float));
        if (trans) {
            memcpy(concatBuf, _queryWeight + ctx->splitIdx * hiddenSize * q_colsPerSplit,
                    hiddenSize * q_colsPerSplit * sizeof(float));
            memcpy(concatBuf + hiddenSize * q_colsPerSplit, _keyWeight + ctx->splitIdx * hiddenSize * k_colsPerSplit,
                    hiddenSize * k_colsPerSplit * sizeof(float));
            memcpy(concatBuf + hiddenSize * (q_colsPerSplit + k_colsPerSplit),
                    _valueWeight + ctx->splitIdx * hiddenSize * v_colsPerSplit,
                    hiddenSize * v_colsPerSplit * sizeof(float));
        } else {
            for (int i = 0; i < hiddenSize; ++i)
                for (int j = 0; j < q_colsPerSplit; ++j)
                    concatBuf[i * colsPerSplit + j]
                            = _queryWeight[ctx->splitIdx * q_colsPerSplit + i * qkv_hidden_size + j];
            for (int i = 0; i < hiddenSize; ++i)
                for (int j = 0; j < k_colsPerSplit; ++j)
                    concatBuf[q_colsPerSplit + i * colsPerSplit + j]
                            = _keyWeight[ctx->splitIdx * k_colsPerSplit + i * qkv_hidden_size + j];
            for (int i = 0; i < hiddenSize; ++i)
                for (int j = 0; j < v_colsPerSplit; ++j)
                    concatBuf[q_colsPerSplit + k_colsPerSplit + i * colsPerSplit + j]
                            = _valueWeight[ctx->splitIdx * v_colsPerSplit + i * qkv_hidden_size + j];
        }

        hpj::Matrix<WeiT> quantizedqkvWeight;
        MMHelper::convertWeight(
                trans, hiddenSize, colsPerSplit, concatBuf, quantizedqkvWeight, qkvWeightScale, qkvWeightZero);
        MMHelper::packWeight(trans, quantizedqkvWeight, qkvWeight);

        free(concatBuf);

        // Merged bias
        if (_queryBias && _keyBias && _valueBias) {
            //colsPerSplit = hiddenSize / ctx->numSplit;
            qkvBias.Resize(colsPerSplit);
            memcpy(qkvBias.Data(), _queryBias + ctx->splitIdx * q_colsPerSplit, sizeof(float) * colsPerSplit);
            memcpy(qkvBias.Data() + q_colsPerSplit, _keyBias + ctx->splitIdx * k_colsPerSplit,
                    sizeof(float) * k_colsPerSplit);
            memcpy(qkvBias.Data() + q_colsPerSplit + k_colsPerSplit, _valueBias + ctx->splitIdx * v_colsPerSplit,
                    sizeof(float) * v_colsPerSplit);
        }

        // Weights for attention output
        // Horizontally split the weight, as the source (PyTorch weight) is transposed, thus looks like vertically
        hpj::Matrix<WeiT> quantizedWeight;
        MMHelper::convertWeight(ctx, trans, hiddenSize, hiddenSize, _attnOutputWeight, false, quantizedWeight,
                attnOutputWeightScale, attnOutputWeightZero);
        MMHelper::packWeight(trans, quantizedWeight, attnOutputWeight);

        // Attention output bias
        if (_attnOutputBias) {
            attnOutputBias.Resize(hiddenSize);
            if (ctx->splitIdx == 0) {
                memcpy(attnOutputBias.Data(), _attnOutputBias, sizeof(float) * hiddenSize);
            } else { // For other splits, set bias to 0, to avoid duplicated calculation
                memset(attnOutputBias.Data(), 0, sizeof(float) * hiddenSize);
            }
        }

        // LayerNorm
        this->norm.setWeight(_gamma1, _beta1, hiddenSize);
    }

#ifdef DEBUG
    void setDebugger(const Debugger &debugger) { this->dbg = debugger; }
#endif

    // Do the forward computing for the whole BERT layer
    // inputBuffer: (bs * seq_len) x hidden_size
    // outBuffer: (bs * seq_len) x hidden_size
    // attnMask: (bs, 1, tgt_len, src_len), tgt_len is the length of query, src_len is the length of key
    // presentKeys, presentValues: past key/values concats current key/values
    // pastSeqLen: the sequence length in pastKeys and pastValues
    // useSelfAttn: use self attention or not, for example, self attention is used to gen the first token
    // doLnBefore: Do layer norm before or not. If true, will do layer norm as the first step
    // returnAttn: return attention values or not
    // returnKVs: return present key/values or not
    // forPT: is it for PyTorch or not, if not for PyTorch, then the cached keys/values are controlled by us
    template <typename KVCacheT>
    void forward(DecoderContext *ctx, float *input, float *output, const float *attnMask, KVCacheT *presentKeys,
            KVCacheT *presentValues, int inputSeqLen, int pastSeqLen, bool useSelfAttn, bool doLnBefore,
            bool returnAttn, bool returnKVs, bool forPT = true, int *positionIds = nullptr) {
        if (forPT) {
            printf("For better perf, need to manage cached key/vaues by ourself, PyTorch extension is not supported "
                   "any more.\n");
            exit(-1);
        }

        hpj::Matrix<float> inputBuffer(input, ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        hpj::Matrix<float> outBuffer(output, ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);

        auto hiddenSize = ctx->hiddenSize;
        auto &qkvMatMul = ctx->qkvMatMul;
        auto &resultBuffer1 = outBuffer;
        auto &resultBuffer2 = ctx->tmpBuf;

#ifdef DEBUG
        dbg.debugPrint("---- DecoderLayer.forward (useSelfAttn=%d) ----\n", useSelfAttn);
        dbg.debugPrint("input:\n");
        dbg.dumpMatrix(inputBuffer);
#endif

        if (doLnBefore) {
            TimeLine t1("input.layer_norm");
            norm.forward(inputBuffer.Data(), resultBuffer1.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    resultBuffer1.Stride());
            t1.release();
        }
#ifdef DEBUG
        dbg.debugPrint("layer norm:\n");
        dbg.dumpMatrix(resultBuffer1);
        dbg.debugPrint("qkvWeight [%d, %d]:\n", this->qkvWeight.Rows(), this->qkvWeight.Cols());
        dbg.dumpMatrix(this->qkvWeight);
#endif

        // Query, Key, Value computed together
        TimeLine t2("QKV.linear");
        DecoderUtil::dense(resultBuffer1, qkvWeight, qkvWeightScale, qkvWeightZero, qkvBias, qkvMatMul);
        t2.release();

        int cols = hiddenSize / ctx->numSplit;
        hpj::Matrix<float> query(qkvMatMul, 0, inputBuffer.Rows(), 0, cols);
        hpj::Matrix<float> key(qkvMatMul, 0, inputBuffer.Rows(), cols, cols);
        hpj::Matrix<float> value(qkvMatMul, 0, inputBuffer.Rows(), cols * 2, cols);

#ifdef DEBUG
        dbg.debugPrint("Q:\n");
        dbg.dumpMatrix(query);
        dbg.debugPrint("K:\n");
        dbg.dumpMatrix(key);
#endif

        // Apply post operattions on query and key
        TimeLine t3("QKPO");
        int qk_shape[4] = {ctx->batchSize, ctx->inputSeqLen, ctx->attHeadNum / ctx->numSplit, ctx->attHeadSize};
        if (positionIds != nullptr) {
            qkpo.forward(query.Data(), key.Data(), query.Stride(), key.Stride(), qk_shape, positionIds);
        } else {
            // Use the default position ids
            std::vector<int> position_ids(ctx->inputSeqLen);
            if (inputSeqLen == 1) {
                position_ids[0] = pastSeqLen;
            } else {
                std::iota(position_ids.begin(), position_ids.end(), 0);
            }
            qkpo.forward(query.Data(), key.Data(), query.Stride(), key.Stride(), qk_shape, position_ids.data());
        }
        t3.release();

#ifdef DEBUG
        dbg.debugPrint("Q after post op:\n");
        dbg.dumpMatrix(query);
        dbg.debugPrint("K after post op:\n");
        dbg.dumpMatrix(key);
#endif

        // Revise attnFactor before softmax (for some models, attnFactor may be not the default value)
        // We initially introduced the code for ChatGLM, but eventually found it has no difference and was unnecessary.
        // However, we have chosen to keep it in the codebase in case it becomes useful for future models.
        if (getScalingCoeff() != 0) { ctx->attFactor = getScalingCoeff(); }

        TimeLine t4("MHA");
#ifndef STEP_BY_STEP_ATTN
        // Use the fused impl.
        if (!useSelfAttn) {
            if constexpr (!INPUT_AS_RESID) {
                auto presult = resultBuffer1.Data();
                int rows = resultBuffer1.Rows(), cols = resultBuffer1.Cols(), stride = resultBuffer1.Stride();
                resultBuffer1.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
                inputBuffer.Assign(presult, rows, cols, stride);
            }
            fusedCrossAttention(
                    ctx, query, key, value, resultBuffer1, presentKeys, presentValues, attnMask, pastSeqLen);
            goto attn_output;
        }
#endif

        // First BMM with the key managed by ourself
        bmm_ours_keys(ctx, query, key, presentKeys, ctx->qkScores, useSelfAttn ? 0 : pastSeqLen);

#ifdef DEBUG
        { // to bypass "crosses initialization" error
            dbg.debugPrint("Q * K:\n");
            int scoreStride = (pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen);
            int debugRows[] = {0, 1, 2, -1, ctx->batchSize * (ctx->attHeadNum / ctx->numSplit) * ctx->inputSeqLen - 1};
            for (int i = 0; i < sizeof(debugRows) / sizeof(int); ++i) {
                if (debugRows[i] < 0) {
                    dbg.debugPrint("...\n");
                    continue;
                }
                auto p = ctx->qkScores + debugRows[i] * scoreStride;
                dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0] * ctx->attFactor, p[1] * ctx->attFactor,
                        p[2] * ctx->attFactor, p[inputSeqLen - 3] * ctx->attFactor, p[inputSeqLen - 2] * ctx->attFactor,
                        p[inputSeqLen - 1] * ctx->attFactor);
            }
        }
#endif

        if (useSelfAttn) {
            DecoderUtil::computeSoftmax(ctx, attnMask, ctx->inputSeqLen, ctx->inputSeqLen);
        } else {
            DecoderUtil::computeSoftmax(
                    ctx, attnMask, ctx->inputSeqLen, pastSeqLen + 1, (pastSeqLen + 1 + 15) / 16 * 16);
        }

#ifdef DEBUG
        { // to bypass "crosses initialization" error
            dbg.debugPrint("attention softmax:\n");
            int scoreStride = (pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen);
            int debugRows[] = {0, 1, 2, -1, ctx->batchSize * (ctx->attHeadNum / ctx->numSplit) * ctx->inputSeqLen - 1};
            for (int i = 0; i < sizeof(debugRows) / sizeof(int); ++i) {
                if (debugRows[i] < 0) {
                    dbg.debugPrint("...\n");
                    continue;
                }
                auto p = ctx->qkScores + debugRows[i] * scoreStride;
                dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[inputSeqLen - 3], p[inputSeqLen - 2],
                        p[inputSeqLen - 1]);
            }
        }
#endif

        // Use the layernorm result (in resultBuffer1) as residential
        // As below code will overwrite resultBuffer1, here we swap inputBuffer and resultBuffer1
        // to make sure values in resultBuffer1 is not overwritten
        if constexpr (!INPUT_AS_RESID) {
            auto presult = resultBuffer1.Data();
            int rows = resultBuffer1.Rows(), cols = resultBuffer1.Cols(), stride = resultBuffer1.Stride();
            resultBuffer1.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
            inputBuffer.Assign(presult, rows, cols, stride);
        }

        // Second BMM with the value managed by ourself
        bmm_our_values(ctx, ctx->qkScores, value, presentValues, resultBuffer1, useSelfAttn ? 0 : pastSeqLen);

    attn_output:
        t4.release();

        hpj::Matrix<float> attnSplit(resultBuffer1.Data(), resultBuffer1.Rows(), resultBuffer1.Cols() / ctx->numSplit,
                resultBuffer1.Stride());

#ifdef DEBUG
        dbg.debugPrint("attention_%d (softmax * value):\n", ctx->splitIdx);
        dbg.dumpMatrix(attnSplit);
#endif

        TimeLine t5("Output");
        // Output/projection in attention, only add the input in the first split
        if (ctx->splitIdx == 0) {
            float gamma = getResidentialScale();

            // denseWithScaledSum should be enough, but as the performance of denseWithScaledSum is not verified,
            // So here still use denseWithSum
            if (gamma == 1) {
                DecoderUtil::denseWithSum(attnSplit, attnOutputWeight, attnOutputWeightScale, attnOutputWeightZero,
                        attnOutputBias, inputBuffer, resultBuffer2);
            } else {
                DecoderUtil::denseWithScaledSum(attnSplit, attnOutputWeight, attnOutputWeightScale,
                        attnOutputWeightZero, attnOutputBias, gamma, inputBuffer, resultBuffer2);
            }
        } else {
            DecoderUtil::dense(attnSplit, attnOutputWeight, attnOutputWeightScale, attnOutputWeightZero, attnOutputBias,
                    resultBuffer2);
        }
        t5.release();

#ifdef DEBUG
        dbg.debugPrint("attention output/projection:\n");
        dbg.dumpMatrix(resultBuffer2);
#endif

        if (!doLnBefore) {
            TimeLine t6("result.layer_norm");
            norm.forward(resultBuffer2.Data(), resultBuffer1.Data(), resultBuffer2.Rows(), resultBuffer2.Stride(),
                    resultBuffer1.Stride());
            t6.release();
        }
    }

protected:
    // The first BatchMatMul inside self attention, need to copy key to presentKeys
    // query: batchSize(any) x seqLength(1/32) x hiddenSize(5120 / 2)
    // key: batchSize(any) x seqLength(1/32) x hiddenSize(5120 / 2)
    // presentKeys: seqLength(32) x batchSize(any) x hiddenSize(5120 / 2)
    // scores: batchSize(1) x attHeadNum(40 / 2) x seqLength(32) x seqLength(32)
    // Note: for self-attention, there is no pad in scores; for cross-attention, with pad
    // TODO: bs > 1 not considered
    template <typename KVCacheT>
    void bmm_ours_keys(DecoderContext *ctx, hpj::Matrix<float> &query, hpj::Matrix<float> &key, KVCacheT *presentKeys,
            float *scores, int pastSeqLen) {
        int sizePerSplit = ctx->hiddenSize / ctx->numSplit;
        int headNumOnDuty = ctx->attHeadNum / ctx->numSplit; // how many head number this task should do
        int batchSize = ctx->batchSize;

        // Copy current key to cached keys
        // Re-layout is needed: (bs, seq, hidden_size) -> (seq, bs, hidden_size)
        auto curKey = presentKeys + pastSeqLen * batchSize * sizePerSplit;
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
                auto src = key.Row(b * ctx->inputSeqLen + seq);
                auto dst = curKey + (seq * batchSize + b) * sizePerSplit;
                if constexpr (std::is_same_v<KVCacheT, float>) {
                    memcpy(dst, src, sizePerSplit * sizeof(float));
                } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                    float16_t::cvt_float_to_float16(src, dst, sizePerSplit);
                }
            }
        }

#ifdef USE_MKLML
#define GRP_COUNT 1
        MKL_INT m[GRP_COUNT] = {ctx->inputSeqLen};
        MKL_INT k[GRP_COUNT] = {ctx->attHeadSize};
        MKL_INT n[GRP_COUNT] = {pastSeqLen + ctx->inputSeqLen};

        MKL_INT lda[GRP_COUNT] = {query.Stride()};
        MKL_INT ldb[GRP_COUNT] = {sizePerSplit * batchSize};
        int strideC = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
        MKL_INT ldc[GRP_COUNT] = {strideC};

        CBLAS_TRANSPOSE transA[GRP_COUNT] = {CblasNoTrans};
        CBLAS_TRANSPOSE transB[GRP_COUNT] = {CblasTrans};

        float alpha[GRP_COUNT] = {1.0};
        float beta[GRP_COUNT] = {0.0};

        const int group_count = headNumOnDuty * batchSize;
        const MKL_INT size_per_grp[GRP_COUNT] = {group_count};

        // Total number of multiplications: headNumOnDuty * batchSize
        const float *a_array[group_count];
        const float *b_array[group_count];
        float *c_array[group_count];
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < headNumOnDuty; ++i) {
                a_array[b * headNumOnDuty + i] = query.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;
                b_array[b * headNumOnDuty + i] = presentKeys + b * sizePerSplit + i * ctx->attHeadSize;
                c_array[b * headNumOnDuty + i] = scores + (b * headNumOnDuty + i) * ctx->inputSeqLen * strideC;
            }
        }

        // Call cblas_sgemm_batch
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc,
                GRP_COUNT, size_per_grp);
#else
        int m = ctx->inputSeqLen;
        int k = ctx->attHeadSize;
        int n = pastSeqLen + ctx->inputSeqLen;
        int lda = query.Stride();
        int ldb = sizePerSplit * batchSize;
        int strideC = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
        int ldc = strideC;
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < headNumOnDuty; ++i) {
                auto A = query.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;
                auto B = presentKeys + b * sizePerSplit + i * ctx->attHeadSize;
                auto C = scores + (b * headNumOnDuty + i) * ctx->inputSeqLen * strideC;
                small_gemm_transb(A, B, C, m, n, k, lda, ldb, ldc);
            }
        }
#endif
    }

    // The second BatchMatMul
    template <typename KVCacheT>
    void bmm_our_values(DecoderContext *ctx, float *scores, hpj::Matrix<float> &value, KVCacheT *presentValues,
            hpj::Matrix<float> &result, int pastSeqLen) {
        int sizePerSplit = ctx->hiddenSize / ctx->numSplit;
        int headNumOnDuty = ctx->attHeadNum / ctx->numSplit; // how many head number this task should do
        int batchSize = ctx->batchSize;

        // Copy current value to cached values
        // Re-layout is needed: (bs, seq, hidden_size) -> (seq, bs, hidden_size)
        auto curValue = presentValues + pastSeqLen * batchSize * sizePerSplit;
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int seq = 0; seq < ctx->inputSeqLen; ++seq) {
                auto src = value.Row(b * ctx->inputSeqLen + seq);
                auto dst = curValue + (seq * batchSize + b) * sizePerSplit;
                if constexpr (std::is_same_v<KVCacheT, float>) {
                    memcpy(dst, src, sizePerSplit * sizeof(float));
                } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                    float16_t::cvt_float_to_float16(src, dst, sizePerSplit);
                }
            }
        }

#ifdef USE_MKLML
#define GRP_COUNT 1
        MKL_INT m[GRP_COUNT] = {ctx->inputSeqLen};
        MKL_INT k[GRP_COUNT] = {pastSeqLen + ctx->inputSeqLen};
        MKL_INT n[GRP_COUNT] = {ctx->attHeadSize};

        int strideA = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
        MKL_INT lda[GRP_COUNT] = {strideA};
        MKL_INT ldb[GRP_COUNT] = {sizePerSplit * batchSize};
        MKL_INT ldc[GRP_COUNT] = {result.Stride()};

        CBLAS_TRANSPOSE transA[GRP_COUNT] = {CblasNoTrans};
        CBLAS_TRANSPOSE transB[GRP_COUNT] = {CblasNoTrans};

        float alpha[GRP_COUNT] = {1.0};
        float beta[GRP_COUNT] = {0.0};

        const int group_count = headNumOnDuty * batchSize;
        const MKL_INT size_per_grp[GRP_COUNT] = {group_count};

        // Total number of multiplications: attHeadNum * batchSize
        const float *a_array[group_count];
        const float *b_array[group_count];
        float *c_array[group_count];
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < headNumOnDuty; ++i) {
                a_array[b * headNumOnDuty + i] = scores + (b * headNumOnDuty + i) * ctx->inputSeqLen * strideA;
                b_array[b * headNumOnDuty + i] = presentValues + b * sizePerSplit + i * ctx->attHeadSize;
                c_array[b * headNumOnDuty + i] = result.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;
            }
        }

        // Call cblas_sgemm_batch
        cblas_sgemm_batch(CblasRowMajor, transA, transB, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc,
                GRP_COUNT, size_per_grp);
#else
        int m = ctx->inputSeqLen;
        int k = pastSeqLen + ctx->inputSeqLen;
        int n = ctx->attHeadSize;

        int strideA = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
        int lda = strideA;
        int ldb = sizePerSplit * batchSize;
        int ldc = result.Stride();

#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < headNumOnDuty; ++i) {
                auto A = scores + (b * headNumOnDuty + i) * ctx->inputSeqLen * strideA;
                auto B = presentValues + b * sizePerSplit + i * ctx->attHeadSize;
                auto C = result.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;

                if constexpr (std::is_same_v<KVCacheT, float>) {
                    small_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
                } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                    small_sgemm_f32f16f32(m, n, k, A, lda, B, ldb, C, ldc);
                } else {
                    printf("Type %s for KV Cache is not supported!\n", typeid(KVCacheT).name());
                    exit(-1);
                }
            }
        }
#endif
    }

    template <typename KVCacheT>
    void fusedCrossAttention(DecoderContext *ctx, hpj::Matrix<float> &query, hpj::Matrix<float> &key,
            hpj::Matrix<float> &value, hpj::Matrix<float> &result, KVCacheT *presentKeys, KVCacheT *presentValues,
            const float *attnMask, int pastSeqLen) {
        int sizePerSplit = ctx->hiddenSize / ctx->numSplit;
        int headNumOnDuty = ctx->attHeadNum / ctx->numSplit; // how many heads this task should do
        int batchSize = ctx->batchSize;

        auto curKey = presentKeys + pastSeqLen * batchSize * sizePerSplit;
        auto curValue = presentValues + pastSeqLen * batchSize * sizePerSplit;

#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < headNumOnDuty; ++i) {
                // Copy current key to cached keys
                // Re-layout is needed: (bs, seq=1, hidden_size) -> (seq=1, bs, hidden_size)
                auto src = key.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;
                auto dst = curKey + b * sizePerSplit + i * ctx->attHeadSize;
                if constexpr (std::is_same_v<KVCacheT, float>) {
                    memcpy(dst, src, ctx->attHeadSize * sizeof(float));
                } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                    float16_t::cvt_float_to_float16(src, dst, ctx->attHeadSize);
                }

                // Q * K
                int m = ctx->inputSeqLen;
                int k = ctx->attHeadSize;
                int n = pastSeqLen + ctx->inputSeqLen;
                int lda = query.Stride();
                int ldb = sizePerSplit * batchSize;
                int strideC = pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen;
                int ldc = strideC;
                auto A = query.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;
                auto B = presentKeys + b * sizePerSplit + i * ctx->attHeadSize;
                auto C = ctx->qkScores + (b * headNumOnDuty + i) * ctx->inputSeqLen * strideC;
                small_gemm_transb(A, B, C, m, n, k, lda, ldb, ldc);

                // Softmax(Q * K)
                const int keyLen = pastSeqLen + 1;
                DecoderUtil::computeSoftmax(ctx, C, attnMask + b * keyLen, keyLen);

                // Copy current value to cached values
                // Re-layout is needed: (bs, seq, hidden_size) -> (seq, bs, hidden_size)
                src = value.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;
                dst = curValue + b * sizePerSplit + i * ctx->attHeadSize;
                if constexpr (std::is_same_v<KVCacheT, float>) {
                    memcpy(dst, src, ctx->attHeadSize * sizeof(float));
                } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                    float16_t::cvt_float_to_float16(src, dst, ctx->attHeadSize);
                }

                // Softmax * V
                std::swap(k, n);
                lda = strideC;
                ldc = result.Stride();
                A = ctx->qkScores + (b * headNumOnDuty + i) * ctx->inputSeqLen * strideC;
                B = presentValues + b * sizePerSplit + i * ctx->attHeadSize;
                C = result.Row(b * ctx->inputSeqLen) + i * ctx->attHeadSize;

                if constexpr (std::is_same_v<KVCacheT, float>) {
                    small_sgemm(m, n, k, A, lda, B, ldb, C, ldc);
                } else if constexpr (std::is_same_v<KVCacheT, float16_t>) {
                    small_sgemm_f32f16f32(m, n, k, A, lda, B, ldb, C, ldc);
                }
            } // end for i
        } // end for b
    }

protected:
    virtual float getResidentialScale() {
        return 1; // directly add the residential
    }

    // Used in computeSoftmax
    virtual float getScalingCoeff() {
        return 0; // 0 means using the default value
    }

    // query, key, value weighs
    hpj::Matrix<WeiT> qkvWeight;
    hpj::Vector<float> qkvWeightScale; // if weighs is int8
    hpj::Vector<float> qkvWeightZero; // if weighs is int8
    // query, key, value bias
    hpj::Vector<float> qkvBias;

    hpj::Matrix<WeiT> attnOutputWeight;
    hpj::Vector<float> attnOutputWeightScale; // if weighs is int8
    hpj::Vector<float> attnOutputWeightZero; // if weighs is int8
    hpj::Vector<float> attnOutputBias;

    // Query/Key post op
    QKPO_CLS qkpo;

    // layerNorm param
    NORM_CLS norm;
    int layerId;

#ifdef DEBUG
    Debugger dbg;
#endif
};
