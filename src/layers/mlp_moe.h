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
#include "bert_util.h"
#include "copy_util.h"
#include "debugger.h"
#include "decoder_util.h"
#include "matmul_helper.h"
#include "mlp_llama.h"
#include "singleton.h"
#include "timeline.h"

#include "weight_util.h"

template <typename WeiT, typename InT = float, typename ImT = float, typename OutT = float>
class MoeMLP : public SingletonBase<MoeMLP<WeiT>> {
public:
    MoeMLP() {}

    MoeMLP(DecoderContext *ctx) {
        // check the config.ini
        if (ctx->numExpertsPerTok < 1 || ctx->numLocalExperts < 1) {
            printf("[ERROR] The expert configuration obtained is incorrect. \nnum_experts_per_tok = %d "
                   "num_local_experts = %d \n",
                    ctx->numExpertsPerTok, ctx->numLocalExperts);
            exit(-1);
        }
        expertsWeight = new LlamaMLP<WeiT, InT, ImT, OutT>[ctx->numLocalExperts];
    }

    ~MoeMLP() {
        if (expertsWeight) {
            delete[] expertsWeight;
            expertsWeight = nullptr;
        }
    }

    // OriWeiT: float or int8_t
    template <typename OriWeiT>
    void setWeights(DecoderContext *ctx, const OriWeiT *gateW, const float *gateS, const float *gateZ,
            const float *gateB, const OriWeiT *upW, const float *upS, const float *upZ, const float *upB,
            const float *normW, const float *normBeta, const OriWeiT *downW, const float *downS, const float *downZ,
            bool trans = true) {
        for (int i = 0; i < ctx->numLocalExperts; i++) {
            expertsWeight[i].setWeights(
                    ctx, gateW, gateS, gateZ, gateB, upW, upS, upZ, upB, normW, normBeta, downW, downS, downZ, trans);
        }
    }

    // Forward for FFN (Feed Forward Network)
    void forward(DecoderContext *ctx, InT *input, OutT *output, int iStride, int oStride,
            bool doLnBefore = true /*not used*/) {}

private:
protected:
    hpj::Matrix<WeiT> expertsGateWeight;
    LlamaMLP<WeiT, InT, ImT, OutT> *expertsWeight;
};
