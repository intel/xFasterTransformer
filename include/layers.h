#pragma once

#include "dtype.h"

namespace xft {

void invokeLayerLLaMA(DataType dt, int batchSize, int inputSeqLen, int attHeadDim, int attHeadNum, int kvHeadNum,
        int maxPositions, int maxPosEmbed, int maxSeqLength, int pastSeqLen, int currentSeqLen, int step,
        int hiddenSize, int intermediateSize, void *output, int outputStride, const void *input, int inputStride,
        const void *queryWeight, const void *keyWeight, const void *valueWeight, const void *attnOutWeight,
        const void *gateWeight, const void *upWeight, const void *downWeight, const void *queryBias = nullptr,
        const void *keyBias = nullptr, const void *valueBias = nullptr, const void *attnOutBias = nullptr);

} // namespace xft