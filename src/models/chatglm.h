#pragma once

#include <vector>
#include "attn_chatglm.h"
#include "common_decoder.h"
#include "layers_norm.h"
#include "mlp_chatglm.h"
#include "rope_2d.h"
#include "token_embedding.h"

template <typename WeiT>
class ChatGLM : public CommonDecoder<ChatGlmAttention<WeiT, RotaryEmbedding2D, LayerNorm>, ChatGlmMLP<WeiT>> {
public:
    ChatGLM(const std::string &modelPath);
    ~ChatGLM();

    void prepareAttnMask(int *ids, int step);
    void embeddingForward(int *ids, float *output, int batchSize, int seqLen);
    void lastLayerNormForward(float *input, float *output, int rows);
    int *getPositionIds(int *ids, int batchSize, int seqLen, int step) override;

private:
    void setEmbeddingWeights(const std::string &modelPath);
    void setFinalLnWeight(const std::string &modelPath);

private:
    TokenEmbedding<float16_t> *embedding;
    LayerNorm finalLN;

    // Mask token ID from configuration
    int maskTokenId;
    int gmaskTokenId;

    // Mask position in current sequence, used to generate position_ids
    // mask_positions, use_gmasks = [], []
    // for seq in seqs:
    //     mask_token = gMASK if gMASK in seq else MASK
    //     use_gmask = mask_token == gMASK
    //     mask_positions.append(seq.index(mask_token))
    //     use_gmasks.append(use_gmask)
    std::vector<int> maskPositions;

    // Record last block positions
    std::vector<int> lastBlockPositions;

    // position_ids + block_position_ids
    // For input_ids [ 74747,  83400,  66846, 130001, 130004], position_ids is:
    // [0, 1, 2, 3, 3] + [0, 0, 0, 0, 1], as gmask_token_id = 130001
    // and next is [3] + [2], ...
    int *positionIds;
    int posBufSize;
};