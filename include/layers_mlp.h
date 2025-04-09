// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include "dtype.h"

namespace xft {

void invokeMLPLLaMA(DataType dt, ActivationType at, int numTokens, int hiddenSize, int intermediateSize, void *output,
        int outputStride, const void *input, int inputStride, const void *gateWeight, const void *upWeight,
        const void *downWeight);

void invokeMoEDeepSeek(DataType dt, ActivationType at, int numTokens, int hiddenSize, int intermediateSize, int moeIntermediateSize,
        int numSharedExperts, int numRoutedExperts, void *output, int outputStride, const void *input, int inputStride, const void *gatingWeight,
        const void *gatingBias, const void *gateWeight, const void *upWeight, const void *downWeight);

// Create an object of DeepSeekMoE
// need expert map?
void *createDeepSeekMoE(int layerId, int numExperts, int numExpPerTok, int hiddenSize, int intermediateSize, bool normTopKProb, int nGroup,
        int topkGroup, const void *gateUpWeights, const void *downWeights, const void *gateUpScales, const void *downScales,
        const void *gatingCorrBias=nullptr, int blockSize=128, int tpRank=0, int tpSize=1);

// Foward function for DeepSeekMoE
void forwardDeepSeekMoE(void *moe, void *input, void *output, int nTokens, void *routingLogits, int iStride=0, int oStride=0);

void forwardDeepSeekMoE(void *moe, void *input, void *output, int nTokens, int *selExperts, float *expertWeights, int iStride=0, int oStride=0);

// Destroy the object of DeepSeekMoE
void destroyDeepSeekMoE(void *moe);

} // namespace xft
