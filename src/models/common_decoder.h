#pragma once

#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "INIReader.h"
#include "abstract_decoder.h"
#include "attention.h"
#include "debugger.h"
#include "decoder_layer.h"
#include "dist_linear.h"
#include "messenger.h"
#include "timeline.h"
#include "transformer_ctx.h"
#include "transpose_util.h"
#include "weight_util.h"

using namespace xft;

struct QKPO_Dummy {
    QKPO_Dummy(int dim) {}
    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *position_ids) {}
};

/******************** Start functions used by reorderCache *******************/

static void swapValues(float *p1, float *p2, int size) {
    int i = 0;
    for (; i + 15 < size; i += 16) {
        __m512 v1 = _mm512_loadu_ps(p1 + i);
        __m512 v2 = _mm512_loadu_ps(p2 + i);
        _mm512_storeu_ps(p1 + i, v2);
        _mm512_storeu_ps(p2 + i, v1);
    }

    if (i < size) {
        int remain = size - i;
        __mmask16 mask = (1 << remain) - 1;

        __m512 v1 = _mm512_maskz_loadu_ps(mask, p1 + i);
        __m512 v2 = _mm512_maskz_loadu_ps(mask, p2 + i);
        _mm512_mask_storeu_ps(p1 + i, mask, v2);
        _mm512_mask_storeu_ps(p2 + i, mask, v1);
    }
}

static void swapValues(float16_t *p1, float16_t *p2, int size) {
    int i = 0;
    for (; i + 31 < size; i += 32) {
        __m512i v1 = _mm512_loadu_si512((__m512i *)(p1 + i));
        __m512i v2 = _mm512_loadu_si512((__m512i *)(p2 + i));
        _mm512_storeu_si512(p1 + i, v2);
        _mm512_storeu_si512(p2 + i, v1);
    }

    if (i < size) {
        int remain = size - i;
        __mmask32 mask = (1 << remain) - 1;

        __m512i v1 = _mm512_maskz_loadu_epi16(mask, (__m512i *)(p1 + i));
        __m512i v2 = _mm512_maskz_loadu_epi16(mask, (__m512i *)(p2 + i));
        _mm512_mask_storeu_epi16(p1 + i, mask, v2);
        _mm512_mask_storeu_epi16(p2 + i, mask, v1);
    }
}

template <typename T>
static void skippableCopy(T *dst, T *src, int size) {
    // Copy only when different
    // TODO: check if there are any risks
    if (*(uint64_t *)dst != *(uint64_t *)src) { memcpy(dst, src, size * sizeof(T)); }
}

template <typename T>
static bool valueExist(T *arr, int size, T val) {
    for (int i = 0; i < size; ++i) {
        if (arr[i] == val) { return true; }
    }
    return false;
}

/******************** end functions used by reorderCache *******************/

// Template parameters:
// ATTN_CLS - class for attention impl.
// MLP_CLS - MLP implementation
// KVCacheT - data type of the cached keys/values
// ATTN_MLP_PARALLEL - true means attention and MLP are in parallel, using the same initial input
template <typename ATTN_CLS, typename MLP_CLS, typename KVCacheT = float, bool ATTN_MLP_PARALLEL = false>
class CommonDecoder : public AbstractDecoder {
public:
    CommonDecoder(const std::string &modelPath, const std::string &modelType)
#ifdef DEBUG
        : dbg("model_decoder.csv")
#endif
    {
        std::string configPath = modelPath + "/config.ini";
        INIReader reader = INIReader(configPath);

        const int attHeadNum = reader.GetInteger(modelType, "head_num");
        // Use the same head number for the default multi-head attention
        const int kvHeadNum = reader.GetInteger(modelType, "kv_head_num", attHeadNum);
        const int size_per_head = reader.GetInteger(modelType, "size_per_head");
        const int imSize = reader.GetInteger(modelType, "inter_size");
        const int layers = reader.GetInteger(modelType, "num_layer");
        const int vocabSize = reader.GetInteger(modelType, "vocab_size");
        // Use 2k as default value
        const int maxPositions = reader.GetInteger(modelType, "max_pos_seq_len", 2048);
        const int hiddenSize = attHeadNum * size_per_head;
        const int embeddingSize = hiddenSize;
        const int multi_query_group_num = reader.GetInteger(modelType, "multi_query_group_num", attHeadNum);
        const float epsilon = reader.GetFloat(modelType, "layernorm_eps", 1e-6);

        std::string act = reader.Get(modelType, "activation_type");
        std::transform(act.begin(), act.end(), act.begin(), ::tolower);

        this->startId = reader.GetInteger(modelType, "start_id", 0);
        this->endId = reader.GetInteger(modelType, "end_id", startId);

        this->initSeqLen = 0;
        this->accSeqLen = 0;

        // Buffer related (not initialized)
        this->inputTokens = nullptr;
        this->maskSize = 0;
        this->attnMask = nullptr;
        this->embRows = 0;
        this->embBuf = nullptr;
        this->outBuf = nullptr;

        this->cacheSizePerLayer = 0;
        this->cachedKeys = new KVCachePointer[layers];
        this->cachedValues = new KVCachePointer[layers];
        for (int i = 0; i < layers; ++i) {
            this->cachedKeys[i] = nullptr;
            this->cachedValues[i] = nullptr;
        }

        // Context
        this->context = nullptr;
        DecoderContext *ctx = getDecoderContext(layers, hiddenSize, attHeadNum, kvHeadNum, imSize, act, epsilon,
                vocabSize, embeddingSize, maxPositions);

        // Decoder
        for (int i = 0; i < layers; ++i) {
            auto pdec = new DECODER(ctx, i);
            this->setDecoderWeights(pdec, modelPath, i);
            this->decoders.push_back(pdec);
        }

        // Predictor
        int workers = messenger.getSize();
        int rank = messenger.getRank();
        this->predictor = new DistLinear<float16_t>(hiddenSize, vocabSize, rank, workers);
        this->setPredictorWeight(modelPath);
    }

    virtual ~CommonDecoder() {
        for (int i = 0; i < decoders.size(); ++i) {
            if (this->cachedKeys[i]) free(this->cachedKeys[i]);
            if (this->cachedValues[i]) free(this->cachedValues[i]);
        }

        delete[] this->cachedKeys;
        delete[] this->cachedValues;

        if (this->inputTokens) free(this->inputTokens);
        if (this->attnMask) free(this->attnMask);
        if (this->embBuf) free(this->embBuf);
        if (this->outBuf) free(this->outBuf);

        delete this->predictor;

        for (auto dec : this->decoders) {
            delete dec;
        }

        delete this->context;
    }

    std::tuple<float *, int, int> forward(int *ids, int64_t *dims, int step, bool logitsAll = false) {
        TimeLine t("Decoder.forward");
        int batchSize = dims[0] * dims[1];
        int seqLen = dims[2];
        // The caller does not give us the IDs, have to use our own
        if (ids == nullptr) {
            if (step == 0) {
                if (this->inputTokens) free(this->inputTokens);
                this->inputTokens = (int *)aligned_alloc(64, batchSize * seqLen * sizeof(int));
                ids = this->inputTokens;
            } else {
                ids = this->inputTokens;
            }
        }

        // Broadcast the IDs from master to other workers
        if (this->messenger.getSize() > 1) { this->messenger.broadcast(ids, batchSize * seqLen); }

        // Prepare context
        DecoderContext *ctx = this->getContext();
        ctx->resize(batchSize, seqLen, (step == 0 ? 0 : this->accSeqLen));

        if (step == 0) {
            // Enlarge buffer if needed
            prepareBuffers(ctx, logitsAll);

            // Reset initial and accumulated sequence length at the first step
            this->initSeqLen = seqLen;
            this->accSeqLen = 0;
        }

        // Embedding
        this->embeddingForward(ids, this->embBuf, batchSize, seqLen);
        this->accSeqLen += seqLen;

        // Prepare attention mask
        this->prepareAttnMask(ids, step);

        // Token position ids, note: different models may have different impl.
        int *positionIds = this->getPositionIds(ids, batchSize, seqLen, step);

        // Decoder: forward
        int hiddenSize = ctx->hiddenSize;
        for (int i = 0; i < this->decoders.size(); ++i) {
            int workers = this->messenger.getSize();
            KVCacheT *presentKeys = this->cachedKeys[i];
            KVCacheT *presentValues = this->cachedValues[i];

            this->decoders[i]->forwardAttention(this->embBuf, this->outBuf, attnMask,
                    presentKeys, // presentKeys,
                    presentValues, // presentValues,
                    seqLen, // inputSeqLen,
                    this->accSeqLen - seqLen, // pastSeqLen
                    step == 0, // useSelfAttn,
                    true, // doLnBefore,
                    false, // returnAttn,
                    false, // returnKVs
                    false, // forPT
                    positionIds);

            auto &attnOut = this->getContext()->tmpBuf;

            // Merge the result of attention
            // When attention and FFN/MLP are in parallel, do not need to reduce after attention
            if constexpr (!ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->messenger.reduceAdd(attnOut.Data(), attnOut.Data(), batchSize * seqLen * attnOut.Stride());
                }
            }

            // When attention and FFN/MLP are in parallel, use the initial embedding as input
            if constexpr (ATTN_MLP_PARALLEL) {
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(this->embBuf, this->outBuf, hiddenSize, hiddenSize, true);
                    this->messenger.reduceAdd(this->outBuf, this->embBuf, batchSize * seqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(this->embBuf, this->embBuf, hiddenSize, hiddenSize, true);
                }
            } else {
                // FFN (for multiple workers, output into outBuf and then reduce add to embBuf)
                if (this->messenger.getSize() > 1) {
                    this->decoders[i]->forwardFFN(attnOut.Data(), this->outBuf, attnOut.Stride(), hiddenSize, true);
                    this->messenger.reduceAdd(this->outBuf, this->embBuf, batchSize * seqLen * hiddenSize);
                } else {
                    this->decoders[i]->forwardFFN(attnOut.Data(), this->embBuf, attnOut.Stride(), hiddenSize, true);
                }
            }
        }

        // Prepare input for final Layer Norm (only care about the last row in outBuf)
        // Shape of embBuf: (bs, seqLen, hiddenSize)
        float *lnIn = this->embBuf;
        if (seqLen > 1 && !logitsAll) { // copy is not needed when seqLen = 1 or logitsAll is true
            lnIn = this->outBuf;
#pragma omp parallel for
            for (int b = 0; b < batchSize; ++b) {
                memcpy(lnIn + b * hiddenSize, this->embBuf + ((b + 1) * seqLen - 1) * hiddenSize,
                        hiddenSize * sizeof(float));
            }
        }

#ifdef DEBUG
        dbg.debugPrint("LayerNorm In:\n");
        dbg.dumpMatrix(lnIn, batchSize, hiddenSize, hiddenSize);
#endif

        // LN, as it supports inplace computing, input and output can be the same
        float *lnOut = this->embBuf;
        if (!logitsAll)
            lastLayerNormForward(lnIn, lnOut, batchSize);
        else
            lastLayerNormForward(lnIn, lnOut, batchSize * seqLen);

#ifdef DEBUG
        dbg.debugPrint("LayerNorm Out:\n");
        dbg.dumpMatrix(lnOut, batchSize, hiddenSize, hiddenSize);
#endif

        // Predictor
        if (!logitsAll)
            this->predictor->forward(lnOut, this->outBuf, batchSize);
        else
            this->predictor->forward(lnOut, this->outBuf, batchSize * seqLen);

#ifdef DEBUG
        auto splitSize = this->predictor->getSplitSize();
        dbg.debugPrint("outBuf:\n");
        dbg.dumpMatrix(outBuf, batchSize, splitSize, splitSize);
#endif

        return std::tuple<float *, int, int>(
                this->outBuf, this->predictor->getSplitOffset(), this->predictor->getSplitSize());
    }

    // Reorder cached keys and values, size=batchSize*beamSize
    void reorderCache(int *idx, int size) {
        int workers = messenger.getSize();
        auto ctx = getContext();
        int batchSize = ctx->batchSize;
        int cols = ctx->hiddenSize / workers;
        int maxSeqLen = ctx->maxPositions;

        REQUIRES(size == batchSize, "reorderCache: size is not correct.");

        // Reorder for all the layers
#pragma omp parallel for
        for (int layer = 0; layer < getLayers(); ++layer) {
            KVCacheT *keys = cachedKeys[layer] + initSeqLen * batchSize * cols;
            KVCacheT *values = cachedValues[layer] + initSeqLen * batchSize * cols;

            // Temporary buffer used for reorder
            KVCacheT *extraKeyBuf = (KVCacheT *)aligned_alloc(64, 2 * (batchSize - 1) * cols * sizeof(KVCacheT));
            KVCacheT *extraValBuf = extraKeyBuf + (batchSize - 1) * cols;

            for (int seq = initSeqLen; seq < accSeqLen; ++seq) { // Reorder is not needed for the first few lines
                int extraBufIdx = 0;
                int remapped[batchSize];
                memcpy(remapped, idx, batchSize * sizeof(int));

                for (int i = 0; i < batchSize; ++i) {
                    int from = remapped[i];
                    if (from < i) { // The source line already reordered
                        // Current line will be used in future, thus save to extra buffer
                        if (valueExist(remapped + i + 1, batchSize - i - 1, i)) {
                            memcpy(extraKeyBuf + extraBufIdx * cols, keys + i * cols, cols * sizeof(KVCacheT));
                            memcpy(extraValBuf + extraBufIdx * cols, values + i * cols, cols * sizeof(KVCacheT));

                            // When need line i, should look into temporary buffer, (extraBufIdx - batchSize) < 0, always
                            std::replace(remapped + i + 1, remapped + batchSize, i, extraBufIdx - batchSize);
                            extraBufIdx += 1;
                        }

                        if (from < 0) { // copy from extraBuf
                            skippableCopy(keys + i * cols, extraKeyBuf + (from + batchSize) * cols, cols);
                            skippableCopy(values + i * cols, extraValBuf + (from + batchSize) * cols, cols);
                        } else {
                            skippableCopy(keys + i * cols, keys + from * cols, cols);
                            skippableCopy(values + i * cols, values + from * cols, cols);
                        }
                    } else if (from > i) {
                        // Just need to swap
                        if (remapped[from] == i) {
                            swapValues(keys + i * cols, keys + from * cols, cols);
                            swapValues(values + i * cols, values + from * cols, cols);

                            // Update the map information
                            std::transform(remapped + i + 1, remapped + batchSize, remapped + i + 1, [&](int num) {
                                if (num == i) {
                                    return from;
                                } else if (num == from) {
                                    return i;
                                }
                                return num;
                            });
                        }
                        // Current line will be used in future, thus save to extra buffer
                        else if (valueExist(remapped + i + 1, batchSize - i - 1, i)) {
                            memcpy(extraKeyBuf + extraBufIdx * cols, keys + i * cols, cols * sizeof(KVCacheT));
                            memcpy(extraValBuf + extraBufIdx * cols, values + i * cols, cols * sizeof(KVCacheT));

                            // When need line i, should look into temporary buffer, (extraBufIdx - batchSize) < 0, always
                            std::replace(remapped + i + 1, remapped + batchSize, i, extraBufIdx - batchSize);
                            extraBufIdx += 1;

                            skippableCopy(keys + i * cols, keys + from * cols, cols);
                            skippableCopy(values + i * cols, values + from * cols, cols);

                            // When need line 'from', should look into line i
                            std::replace(remapped + i + 1, remapped + batchSize, from, i);
                        }
                        // Current line will never be used in futre, just overwrite it
                        else {
                            skippableCopy(keys + i * cols, keys + from * cols, cols);
                            skippableCopy(values + i * cols, values + from * cols, cols);

                            // When need line 'from', should look into line i
                            std::replace(remapped + i + 1, remapped + batchSize, from, i);
                        }
                    }
                }

                keys += batchSize * cols;
                values += batchSize * cols;
            }

            // Clean up
            free(extraKeyBuf);
        }
    }

    // Get decoder context
    DecoderContext *getContext() { return context; }

    // How many layers
    int getLayers() { return decoders.size(); }

    Messenger &getMessenger() { return messenger; }

    int getRank() { return messenger.getRank(); }

protected:
    using DECODER = Decoder<ATTN_CLS, MLP_CLS>;

    static bool fileExists(const std::string &filename) {
        std::ifstream file(filename);
        return file.good();
    }

    DecoderContext *getDecoderContext(int layers, const int hiddenSize, const int attHeadNum, const int kvHeadNum,
            const int imSize, const std::string &act, const float epsilon, int vocabSize, int embeddingSize,
            int maxPositions) {
        int splits = messenger.getSize();
        int splitIdx = messenger.getRank();

        if (context != nullptr) {
            if (context->hiddenSize == hiddenSize && context->attHeadNum == attHeadNum
                    && context->kvHeadNum == kvHeadNum && context->intermediateSize == imSize
                    && context->splitIdx == splitIdx) {
                return context;
            } else {
                printf("Different context size not unsupported!\n");
                exit(-1);
            }
        } else {
            this->context = new DecoderContext(layers, hiddenSize, attHeadNum, kvHeadNum, imSize, act, epsilon,
                    vocabSize, embeddingSize, maxPositions, splitIdx, splits);
        }

        return this->context;
    }

    void setDecoderWeights(DECODER *pdecoder, const std::string &modelPath, int layerIdx) {
        const int hiddenSize = pdecoder->getCtx()->hiddenSize;
        const int imSize = pdecoder->getCtx()->intermediateSize;
        const int kvHeadNum = pdecoder->getCtx()->kvHeadNum;
        const int attHeadSize = pdecoder->getCtx()->attHeadSize;
        const int mlpFactor = (pdecoder->getCtx()->actType == DecoderContext::SWIGLU) ? 2 : 1;
        int qSize = hiddenSize;
        int kvSize = attHeadSize * kvHeadNum;
        int qkvSize = qSize + kvSize + kvSize;

#define ALLOC(size, alignment) aligned_alloc((alignment), (size))
        float *qkvWeight = (float *)ALLOC(hiddenSize * qkvSize * sizeof(float), 64);
        float *qkvBias = (float *)ALLOC(qkvSize * sizeof(float), 64);
        float *attnOutWeight = (float *)ALLOC(hiddenSize * hiddenSize * sizeof(float), 64);
        float *attnOutBias = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *fc1Weight = (float *)ALLOC(hiddenSize * imSize * mlpFactor * sizeof(float), 64);
        float *fc1Bias = (float *)ALLOC(imSize * sizeof(float), 64);
        float *fc2Weight = (float *)ALLOC(hiddenSize * imSize * sizeof(float), 64);
        float *fc2Bias = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln1Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln1Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Gamma = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *ln2Beta = (float *)ALLOC(hiddenSize * sizeof(float), 64);
        float *fc3Weight = nullptr;

        // printf("hiddenSize=%d, qkvSize=%d\n", hiddenSize, qkvSize);
        REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx)
                                 + ".attention.query_key_value.weight.0.bin",
                         qkvWeight, hiddenSize * qkvSize)
                        == hiddenSize * qkvSize,
                "read QKV weight error");
        REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.weight.0.bin",
                         attnOutWeight, hiddenSize * hiddenSize)
                        == hiddenSize * hiddenSize,
                "read attn dense weight error");

        // Stardard 2 layer MLP
        if (fileExists(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.weight.0.bin")) {
            REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx)
                                     + ".mlp.dense_h_to_4h.weight.0.bin",
                             fc1Weight, hiddenSize * imSize * mlpFactor)
                            == hiddenSize * imSize * mlpFactor,
                    "read FC1 weight error");
            REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx)
                                     + ".mlp.dense_4h_to_h.weight.0.bin",
                             fc2Weight, hiddenSize * imSize)
                            == hiddenSize * imSize,
                    "read FC2 weight error");
        }
        // gate, up, down weights for Llama like model
        else {
            fc3Weight = (float *)ALLOC(hiddenSize * imSize * sizeof(float), 64);
            REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.gate_proj.weight.0.bin",
                             fc1Weight, hiddenSize * imSize * mlpFactor)
                            == hiddenSize * imSize * mlpFactor,
                    "read gate weight error");
            REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.up_proj.weight.0.bin",
                             fc2Weight, hiddenSize * imSize)
                            == hiddenSize * imSize,
                    "read up weight error");
            REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.down_proj.weight.0.bin",
                             fc3Weight, hiddenSize * imSize)
                            == hiddenSize * imSize,
                    "read down weight error");
        }

        REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.weight.bin",
                         ln1Gamma, hiddenSize)
                        == hiddenSize,
                "read LN1 gamma error");
        REQUIRES(readFile(modelPath + "/model.layers." + std::to_string(layerIdx)
                                 + ".post_attention_layernorm.weight.bin",
                         ln2Gamma, hiddenSize)
                        == hiddenSize,
                "read LN2 gamma error");

#define READ_OPTIONAL(filename, addr, size, errmsg)           \
    {                                                         \
        int ret = readFile((filename), (addr), (size));       \
        if (ret == 0) {                                       \
            free(addr);                                       \
            addr = nullptr;                                   \
        } else {                                              \
            if (ret != (size)) {                              \
                printf("%s\n", (errmsg));                     \
                exit(-1);                                     \
            }                                                 \
        }                                                     \
    }

        // The bias is optional
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.query_key_value.bias.0.bin",
                qkvBias, qkvSize, "read QKV bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".attention.dense.bias.bin",
                attnOutBias, hiddenSize, "read attn dense bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".input_layernorm.bias.bin", ln1Beta,
                hiddenSize, "read LN1 beta error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".post_attention_layernorm.bias.bin",
                ln2Beta, hiddenSize, "read LN2 beta error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_h_to_4h.bias.0.bin",
                fc1Bias, imSize, "read FC1 bias error");
        READ_OPTIONAL(modelPath + "/model.layers." + std::to_string(layerIdx) + ".mlp.dense_4h_to_h.bias.bin", fc2Bias,
                hiddenSize, "read FC2 bias error");

#define FREE(x) if ((x)) free((x))
        // Need the tranposed weights in our interface
        // ordering, trans, rows, cols, alpha, a, lda, b, ldb
#ifdef TRANS_FOR_DEBUG
        float *qkvWeightT = (float *)ALLOC(hiddenSize * qkvSize * sizeof(float), 64);
        TransposeUtil::transpose(qkvWeight, qkvWeightT, hiddenSize, qkvSize);
        FREE(qkvWeight);
        float *attnOutWeightT = (float *)ALLOC(hiddenSize * hiddenSize * sizeof(float), 64);
        TransposeUtil::transpose(attnOutWeight, attnOutWeightT, hiddenSize, hiddenSize);
        FREE(attnOutWeight);
        float *fc1WeightT = (float *)ALLOC(hiddenSize * imSize * mlpFactor * sizeof(float), 64);
        float *fc2WeightT = (float *)ALLOC(hiddenSize * imSize * sizeof(float), 64);
        float *fc3WeightT = (float *)ALLOC(hiddenSize * imSize * sizeof(float), 64);
        if (fc3Weight) {
            TransposeUtil::transpose(fc1Weight, fc1WeightT, hiddenSize, imSize * mlpFactor);
            TransposeUtil::transpose(fc2Weight, fc2WeightT, hiddenSize, imSize);
            TransposeUtil::transpose(fc3Weight, fc3WeightT, imSize, hiddenSize);
        } else {
            TransposeUtil::transpose(fc1Weight, fc1WeightT, hiddenSize, imSize * mlpFactor);
            TransposeUtil::transpose(fc2Weight, fc2WeightT, imSize, hiddenSize);
        }
        FREE(fc1Weight);
        FREE(fc2Weight);
        FREE(fc3Weight);
        std::vector<float *> params = {qkvWeightT, qkvBias, qkvWeightT + qSize * hiddenSize, qkvBias + qSize,
                qkvWeightT + (qSize + kvSize) * hiddenSize, qkvBias + qSize + kvSize, attnOutWeightT, attnOutBias,
                ln1Gamma, ln1Beta, fc1WeightT, fc1Bias, fc2WeightT, fc2Bias, ln2Gamma, ln2Beta, fc3WeightT};
        pdecoder->setWeights(params, true);
        FREE(qkvWeightT);
        FREE(attnOutWeightT);
        FREE(fc1WeightT);
        FREE(fc2WeightT);
        FREE(fc3WeightT);
#else
        std::vector<float *> params = {qkvWeight, qkvBias, qkvWeight + qSize, qkvBias + qSize,
                qkvWeight + qSize + kvSize, qkvBias + qSize + kvSize, attnOutWeight, attnOutBias, ln1Gamma, ln1Beta,
                fc1Weight, fc1Bias, fc2Weight, fc2Bias, ln2Gamma, ln2Beta, fc3Weight};
        pdecoder->setWeights(params, false);
        FREE(qkvWeight);
        FREE(attnOutWeight);
        FREE(fc1Weight);
        FREE(fc2Weight);
        FREE(fc3Weight);
#endif
        FREE(qkvBias);
        FREE(attnOutBias);
        FREE(fc1Bias);
        FREE(fc2Bias);
        FREE(ln1Gamma);
        FREE(ln1Beta);
        FREE(ln2Gamma);
        FREE(ln2Beta);
    }

    void setPredictorWeight(const std::string &modelPath) {
        int inputSize = predictor->getInputSize();
        int outputSize = predictor->getOutputSize();

        float *weight = (float *)malloc(inputSize * outputSize * sizeof(float));
        float *bias = nullptr;

        REQUIRES(readFile(modelPath + "/model.lm_head.weight.bin", weight, inputSize * outputSize)
                        == inputSize * outputSize,
                "read predictor weight error");

        predictor->setWeight(weight, bias);

        free(weight);
    }

    virtual void prepareBuffers(DecoderContext *ctx, bool logitsAll = false) {
        int batchSize = ctx->batchSize;
        int hiddenSize = ctx->hiddenSize;
        int seqLen = ctx->inputSeqLen;
        int vocabSize = ctx->vocabSize;
        int maxPositions = ctx->maxPositions;
        int layers = this->decoders.size();
        int workers = this->messenger.getSize();

        // Prepare buffers (embBuf & outBuf)
        int logitsLen = logitsAll ? batchSize * seqLen : batchSize;
        int requiredRows = batchSize * seqLen;
        // The required output buffer size is bigger than the embedding size
        if (logitsLen * vocabSize > batchSize * seqLen * hiddenSize) {
            requiredRows = logitsLen * vocabSize / hiddenSize + 1;
        }
        if (requiredRows > this->embRows) {
            if (this->embBuf) free(this->embBuf);
            if (this->outBuf) free(this->outBuf);
            this->embBuf = (float *)aligned_alloc(64, requiredRows * hiddenSize * sizeof(float));
            this->outBuf = (float *)aligned_alloc(64, requiredRows * hiddenSize * sizeof(float));
            this->embRows = requiredRows;
        }

        // Attention mask
        int sizeRequired = batchSize * seqLen * seqLen;
        getAttnMask(sizeRequired);

        // Cached keys/values
        // The maximum sequence length is to be the same as maxPositions, at most
        sizeRequired = maxPositions * batchSize * (hiddenSize / workers);
        if (this->cacheSizePerLayer < sizeRequired) {
            for (int i = 0; i < layers; ++i) {
                if (this->cachedKeys[i]) free(this->cachedKeys[i]);
                if (this->cachedValues[i]) free(this->cachedValues[i]);

                this->cachedKeys[i] = (KVCacheT *)aligned_alloc(64, sizeRequired * sizeof(KVCacheT));
                this->cachedValues[i] = (KVCacheT *)aligned_alloc(64, sizeRequired * sizeof(KVCacheT));

                if (this->cachedKeys[i] == nullptr || this->cachedValues[i] == nullptr) {
                    printf("Cannot allocate buffer for cached keys/values.\n");
                    exit(-1);
                }
            }

            this->cacheSizePerLayer = sizeRequired;
        }
    }

    float *getAttnMask(int sizeRequired) {
        if (this->maskSize < sizeRequired) {
            if (this->attnMask) free(this->attnMask);
            this->attnMask = (float *)aligned_alloc(64, sizeRequired * sizeof(float));
            this->maskSize = sizeRequired;
        }
        return this->attnMask;
    }

    int getStartId() { return startId; }
    int getEndId() { return endId; }

    virtual void embeddingForward(int *ids, float *output, int batchSize, int seqLen) = 0;
    virtual void lastLayerNormForward(float *input, float *output, int rows) = 0;
    virtual void prepareAttnMask(int *ids, int step) = 0;
    virtual int *getPositionIds(int *ids, int batchSize, int seqLen, int step) { return nullptr; }

protected:
    // For communication
    Messenger messenger;

    // Execution context
    DecoderContext *context;

    // The initial input sequence length, which is the prompt token size
    int initSeqLen;
    // Accumulated sequence length, = past_seq_len + current_seq_len
    int accSeqLen;
    // Record the mapped index for each sequence in keys/values
    std::vector<std::vector<int>> kvMapping;

    // If not the master, need to receive token IDs from the master
    int *inputTokens;

    KVCacheT **cachedKeys; // all accumulated keys
    KVCacheT **cachedValues; // all accumulated values
    int cacheSizePerLayer; // size of keys/values per layer

    int embRows; // allocated rows in embBuf
    float *embBuf; // used to store the embedding result
    float *outBuf; // output buffer for decoder layers, same shape with embBuf

protected:
    typedef KVCacheT *KVCachePointer;

    // Components most LLMs may use
    std::vector<DECODER *> decoders;
    DistLinear<float16_t> *predictor;

private:
    int maskSize; // size of allocated attnMask
    float *attnMask; // attention mask, set as private as may need to enlarge

    int startId;
    int endId;
#ifdef DEBUG
    Debugger dbg;
#endif
};