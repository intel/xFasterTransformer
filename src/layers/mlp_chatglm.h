#pragma once
#include <cmath>

#include "mlp_standard.h"

template <typename WeiT>
class ChatGlmMLP : public MLP<WeiT, false> {
public:
    ChatGlmMLP(DecoderContext *ctx) : MLP<WeiT, false>(ctx) { residScale = std::sqrt(2 * ctx->layers); }

protected:
    float getResidentialScale() override { return residScale; }

private:
    // Residential scale
    float residScale;
};