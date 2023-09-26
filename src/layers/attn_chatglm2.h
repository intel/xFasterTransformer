#pragma once
#include <cmath>

#include "attention.h"

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS, bool INPUT_AS_RESID>
class ChatGLM2Attention : public Attention<WeiT, QKPO_CLS, NORM_CLS, INPUT_AS_RESID> {
public:
    ChatGLM2Attention(int layerId, DecoderContext *ctx)
        : Attention<WeiT, QKPO_CLS, NORM_CLS, INPUT_AS_RESID>(layerId, ctx) {
        residScale = 1; // std::sqrt(2 * ctx->layers);
        scalingCoeff = 1.0f / (std::sqrt(ctx->attHeadSize) * (layerId + 1));
        // TODO: remove internal buffer,
        //       using qkvBuffer to direct compute BMM
        //       without expand grouped QKV in multi-head query
        //       these functions need to upgrade :
        //           fusedCrossAttention, bmm_ours_keys, bmm_our_values
        this->qkvGroupBuffer = NULL;
        this->qkvGroupBufferLen = 0;
    }
    virtual ~ChatGLM2Attention() {
        if (this->qkvGroupBuffer != NULL) {
            free(this->qkvGroupBuffer);
            this->qkvGroupBuffer = NULL;
        }
        this->qkvGroupBufferLen = 0;
    }

protected:
    float getResidentialScale() override { return residScale; }

private:
    // Residential scale
    float residScale;
    // query_key_layer_scaling_coeff
    float scalingCoeff;
    hpj::Matrix<float> qkvGroupMatMul;
    float *qkvGroupBuffer;
    int qkvGroupBufferLen;

protected:
    void expand_to_qkv(hpj::Matrix<float> &qkvMatMul, hpj::Matrix<float> &qkvGroupMatMul, int qkvRows, int q_cols,
            int kv_cols, int group_num) {
        int copy_num = q_cols / kv_cols;
        int k_cols = kv_cols / group_num;
#pragma omp parallel for
        for (int i = 0; i < qkvRows; i++) {
            const float *rowFrom = qkvGroupMatMul.Row(i);
            float *rowTo = qkvMatMul.Row(i);
            memcpy(rowTo, rowFrom, q_cols * sizeof(float));
            rowFrom += q_cols;
            rowTo += q_cols;
            for (int j = 0; j < group_num; j++) {
                for (int k = 0; k < copy_num; k++) {
                    memcpy(rowTo, rowFrom, k_cols * sizeof(float));
                    rowTo += k_cols;
                }
                rowFrom += k_cols;
            }
            for (int j = 0; j < group_num; j++) {
                for (int k = 0; k < copy_num; k++) {
                    memcpy(rowTo, rowFrom, k_cols * sizeof(float));
                    rowTo += k_cols;
                }
                rowFrom += k_cols;
            }
        }
    }
    // Do the forward computing for the whole BERT layer
    // inputBuffer: (bs * seq_len) x hidden_size
    // outBuffer: (bs * seq_len) x hidden_size
    // attnMask: (bs, 1, tgt_len, src_len), tgt_len is the length of query, src_len is the length of key
    // pastKeys, pastValues: keys and values in the past; shape: batchSize(4) x attHeadNum(40) x pastSeqLen(N) x attHeadSize(128)
    // presentKeys, presentValues: past key/values concats current key/values
    // pastSeqLen: the sequence length in pastKeys and pastValues
    // useSelfAttn: use self attention or not, for example, self attention is used to gen the first token
    // doLnBefore: Do layer norm before or not. If true, will do layer norm as the first step
    // returnAttn: return attention values or not
    // returnKVs: return present key/values or not
    // forPT: is it for PyTorch or not, if not for PyTorch, then the cached keys/values are controlled by us

public:
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
        auto &resultBuffer1 = (ctx->numSplit == 1 ? outBuffer : ctx->normBuf);
        auto &resultBuffer2 = ctx->tmpBuf;
        float epsilon = ctx->epsilon;

        // //init group_qkvBuffer
        int attHeadSize = ctx->attHeadSize;
        int kvHeadNum = ctx->kvHeadNum;
        int attHeadNum = ctx->attHeadNum;
        int qkvRows = ctx->batchSize * inputSeqLen;
        // multi query attention
        int q_cols = attHeadNum * attHeadSize;
        int kv_cols = kvHeadNum * attHeadSize;
        int qkvCols = q_cols + kv_cols * 2;

        int tmp_pad = 4;
        //No need pad for AMX
        if constexpr (std::is_same_v<WeiT, bfloat16_t>) { tmp_pad = 0; }
        const int pad = tmp_pad;
        int qkvStride = (qkvCols % 512 == 0 ? qkvCols + pad : qkvCols); // stride for the concated QKV
        int newLen = qkvRows * qkvStride;
        // printf("qkvGroupMatMul qkvRows=%d, qkvCols=%d, qkvStride=%d, new_len=%d\n", qkvRows, qkvCols, qkvStride, new_len);
        if (this->qkvGroupBufferLen < newLen) {
            this->qkvGroupBufferLen = newLen;
            if (this->qkvGroupBuffer != NULL) {
                free(this->qkvGroupBuffer);
                this->qkvGroupBuffer = NULL;
            }
            this->qkvGroupBuffer = (float *)aligned_alloc(64, sizeof(float) * this->qkvGroupBufferLen);
        }
        this->qkvGroupMatMul.Assign(this->qkvGroupBuffer, qkvRows, qkvCols, qkvStride);

#ifdef DEBUG
        this->dbg.debugPrint("---- GLM2 DecoderLayer.forward (useSelfAttn=%d) ----\n", useSelfAttn);
        this->dbg.debugPrint("input [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        this->dbg.dumpMatrix(inputBuffer);
#endif

        if (doLnBefore) {
            this->norm.forward(inputBuffer.Data(), resultBuffer1.Data(), inputBuffer.Rows(), inputBuffer.Stride(),
                    resultBuffer1.Stride(), epsilon);
        }
#ifdef DEBUG
        this->dbg.debugPrint(
                "layer norm [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, ctx->hiddenSize);
        this->dbg.dumpMatrix(resultBuffer1);
#endif
        // Query, Key, Value computed together
        DecoderUtil::dense(resultBuffer1, this->qkvWeight, this->qkvWeightScale, this->qkvWeightZero, this->qkvBias,
                qkvGroupMatMul);

#ifdef DEBUG
        this->dbg.debugPrint("dense [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, qkvCols);
        this->dbg.dumpMatrix(qkvGroupMatMul);
#endif

        // Apply post operattions on query and key
        int q_shape[4] = {ctx->batchSize, inputSeqLen, attHeadNum, attHeadSize};
        int k_shape[4] = {ctx->batchSize, inputSeqLen, kvHeadNum, attHeadSize};
        if (positionIds != nullptr) {
            this->qkpo.forward(qkvGroupMatMul.Data(), qkvGroupMatMul.Stride(), ctx->batchSize, inputSeqLen, attHeadNum,
                    kvHeadNum, attHeadSize, positionIds);
        } else {
            std::vector<int> position_ids(ctx->inputSeqLen);
            if (inputSeqLen == 1) {
                position_ids[0] = pastSeqLen;
            } else {
                std::iota(position_ids.begin(), position_ids.end(), 0);
            }
            this->qkpo.forward(qkvGroupMatMul.Data(), qkvGroupMatMul.Stride(), ctx->batchSize, inputSeqLen, attHeadNum,
                    kvHeadNum, attHeadSize, position_ids.data());
        }

#ifdef DEBUG
        this->dbg.debugPrint("qkpo [%d, %d, %d]:\n", ctx->batchSize * inputSeqLen, ctx->hiddenSize, qkvCols);
        this->dbg.dumpMatrix(qkvGroupMatMul);
#endif

        this->expand_to_qkv(qkvMatMul, qkvGroupMatMul, qkvRows, q_cols, kv_cols, kvHeadNum);

        int cols = q_cols / ctx->numSplit;
        hpj::Matrix<float> query(qkvMatMul, 0, inputBuffer.Rows(), 0, cols);
        hpj::Matrix<float> key(qkvMatMul, 0, inputBuffer.Rows(), cols, cols);
        hpj::Matrix<float> value(qkvMatMul, 0, inputBuffer.Rows(), cols * 2, cols);

#ifdef DEBUG
        this->dbg.debugPrint("Q:\n");
        this->dbg.dumpMatrix(query);
        this->dbg.debugPrint("K:\n");
        this->dbg.dumpMatrix(key);
        this->dbg.debugPrint("V:\n");
        this->dbg.dumpMatrix(value);
#endif

        if (this->getScalingCoeff() != 0) { ctx->attFactor = this->getScalingCoeff(); }
        TimeLine t4("MHA");
        if (!useSelfAttn) {
            if constexpr (!INPUT_AS_RESID) {
                auto presult = resultBuffer1.Data();
                int rows = resultBuffer1.Rows(), cols = resultBuffer1.Cols(), stride = resultBuffer1.Stride();
                resultBuffer1.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
                inputBuffer.Assign(presult, rows, cols, stride);
            }
            this->fusedCrossAttention(
                    ctx, query, key, value, resultBuffer1, presentKeys, presentValues, attnMask, pastSeqLen);
        } else {
            // First BMM with the key managed by ourself
            this->bmm_ours_keys(
                    ctx, query, key, presentKeys, ctx->qkScores, useSelfAttn ? 0 : pastSeqLen); //, ctx->attFactor);

            // Revise attnFactor before softmax (for some models, attnFactor may be not the default value)
            // We initially introduced the code for ChatGLM, but eventually found it has no difference and was unnecessary.
            // However, we have chosen to keep it in the codebase in case it becomes useful for future models.

#ifdef DEBUG
            this->dbg.debugPrint("Q * K:\n");
            int scoreStride = (pastSeqLen > 0 ? (pastSeqLen + ctx->inputSeqLen + 15) / 16 * 16 : ctx->inputSeqLen);
            int debugRows[] = {0, 1, 2, -1, ctx->batchSize * ctx->attHeadNum * ctx->inputSeqLen - 1};
            for (int i = 0; i < sizeof(debugRows) / sizeof(int); ++i) {
                if (debugRows[i] < 0) {
                    this->dbg.debugPrint("...\n");
                    continue;
                }
                auto p = ctx->qkScores + debugRows[i] * scoreStride;
                this->dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0] * ctx->attFactor, p[1] * ctx->attFactor,
                        p[2] * ctx->attFactor, p[inputSeqLen - 3] * ctx->attFactor, p[inputSeqLen - 2] * ctx->attFactor,
                        p[inputSeqLen - 1] * ctx->attFactor);
            }
#endif

            if (useSelfAttn) {
                DecoderUtil::computeSoftmax(ctx, attnMask, ctx->inputSeqLen, ctx->inputSeqLen);
            } else {
                DecoderUtil::computeSoftmax(
                        ctx, attnMask, ctx->inputSeqLen, pastSeqLen + 1, (pastSeqLen + 1 + 15) / 16 * 16);
            }
#ifdef DEBUG
            this->dbg.debugPrint("attention softmax:\n");
            for (int i = 0; i < sizeof(debugRows) / sizeof(int); ++i) {
                if (debugRows[i] < 0) {
                    this->dbg.debugPrint("...\n");
                    continue;
                }
                auto p = ctx->qkScores + debugRows[i] * scoreStride;
                this->dbg.debugPrint("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[inputSeqLen - 3],
                        p[inputSeqLen - 2], p[inputSeqLen - 1]);
            }
#endif

            // Use the layernorm result (in resultBuffer1) as residential
            // As below code will overwrite resultBuffer1, here we swap inputBuffer and resultBuffer1
            // to make sure values in resultBuffer1 is not overwritten
            if constexpr (!INPUT_AS_RESID) {
                printf("replace input\n");
                auto presult = resultBuffer1.Data();
                int rows = resultBuffer1.Rows(), cols = resultBuffer1.Cols(), stride = resultBuffer1.Stride();
                resultBuffer1.Assign(inputBuffer.Data(), inputBuffer.Rows(), inputBuffer.Cols(), inputBuffer.Stride());
                inputBuffer.Assign(presult, rows, cols, stride);
            }

            // Second BMM with the value managed by ourself
            this->bmm_our_values(ctx, ctx->qkScores, value, presentValues, resultBuffer1, useSelfAttn ? 0 : pastSeqLen);
        }
        t4.release();
        hpj::Matrix<float> attnSplit(resultBuffer1.Data(), resultBuffer1.Rows(), resultBuffer1.Cols() / ctx->numSplit,
                resultBuffer1.Stride());

#ifdef DEBUG
        this->dbg.debugPrint("attention_%d (softmax * value):\n", ctx->splitIdx);
        this->dbg.dumpMatrix(attnSplit);
#endif
        // Output/projection in attention, only add the input in the first split
        if (ctx->splitIdx == 0) {
            float gamma = getResidentialScale();

            // denseWithScaledSum should be enough, but as the performance of denseWithScaledSum is not verified,
            // So here still use denseWithSum
            if (gamma == 1) {
                DecoderUtil::denseWithSum(attnSplit, this->attnOutputWeight, this->attnOutputWeightScale,
                        this->attnOutputWeightZero, this->attnOutputBias, inputBuffer, resultBuffer2);
            } else {
                DecoderUtil::denseWithScaledSum(attnSplit, this->attnOutputWeight, this->attnOutputWeightScale,
                        this->attnOutputWeightZero, this->attnOutputBias, gamma, inputBuffer, resultBuffer2);
            }
        } else {
            DecoderUtil::dense(attnSplit, this->attnOutputWeight, this->attnOutputWeightScale,
                    this->attnOutputWeightZero, this->attnOutputBias, resultBuffer2);
        }

#ifdef DEBUG
        this->dbg.debugPrint("attention output/projection:\n");
        this->dbg.dumpMatrix(resultBuffer2);
#endif

        if (!doLnBefore) {
            this->norm.forward(resultBuffer2.Data(), resultBuffer1.Data(), resultBuffer2.Rows(), resultBuffer2.Stride(),
                    resultBuffer1.Stride());
        }
    }
};
