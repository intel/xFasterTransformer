#pragma once

#include "common_decoder.h"
#include "mlp_llama.h"
#include "rms_norm.h"
#include "rotary_embedding.h"
#include "token_embedding.h"

template <typename WeiT>
class LlamaLLM : public CommonDecoder<Attention<WeiT, LlamaRotaryEmbedding, RmsNorm>, LlamaMLP<WeiT>, float> {
public:
    LlamaLLM(const std::string &modelPath);
    ~LlamaLLM();

    void prepareAttnMask(int *ids, int step);
    void embeddingForward(int *ids, float *output, int batchSize, int seqLen);
    void lastLayerNormForward(float *input, float *output, int rows);

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    RmsNorm finalLN;
};