#pragma once
#include <cmath>

#include "attention.h"

template <typename WeiT, typename QKPO_CLS, typename NORM_CLS>
class ChatGlmAttention : public Attention<WeiT, QKPO_CLS, NORM_CLS, false> {
public:
    ChatGlmAttention(int layerId, DecoderContext *ctx) : Attention<WeiT, QKPO_CLS, NORM_CLS, false>(layerId, ctx) {
        residScale = std::sqrt(2 * ctx->layers);
        scalingCoeff = 1.0f / (std::sqrt(ctx->attHeadSize) * (layerId + 1));
    }

protected:
    float getResidentialScale() override { return residScale; }

    // Do NOT needed, in ChatGLM, query_layer is divided by 'query_key_layer_scaling_coeff',
    // but 'attention_scores' is multiplied by the same factor before softmax
    // float getScalingCoeff() override {
    //     return scalingCoeff;
    // }

private:
    // Residential scale
    float residScale;
    // query_key_layer_scaling_coeff
    float scalingCoeff;
};