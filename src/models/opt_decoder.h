#pragma once
#include <map>
#include <string>
#include <vector>

#include "abstract_decoder.h"
#include "attention.h"
#include "common_decoder.h"
#include "dist_linear.h"
#include "float16.h"
#include "layer_norm.h"
#include "messenger.h"
#include "mlp_standard.h"
#include "opt_embedding.h"
#include "transformer_ctx.h"

template <typename WeiT>
class OptDecoder : public CommonDecoder<Attention<WeiT, QKPO_Dummy, LayerNorm>, MLP<WeiT>> {
public:
    OptDecoder(const std::string &modelPath);
    ~OptDecoder();

    void prepareAttnMask(int *ids, int step);
    void embeddingForward(int *ids, float *output, int batchSize, int seqLen);
    void lastLayerNormForward(float *input, float *output, int rows);

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    OptEmbedding<float16_t> *embedding;
    LayerNorm finalLN;
};