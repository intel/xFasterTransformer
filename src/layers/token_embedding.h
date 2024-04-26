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
#include "allocator.h"
#include "float16.h"
#include "token_embedding_kernels.h"
#include "transformer_ctx.h"

template <typename T>
class TokenEmbedding {
public:
    TokenEmbedding(DecoderContext *ctx) {
        this->vocabSize = ctx->vocabSize;
        this->hiddenSize = ctx->hiddenSize;
    }

    void setWeights(float *tokenEmb) {
        int size = vocabSize * hiddenSize;
        embTable = (T *)xft::alloc(size * sizeof(T));

        if constexpr (std::is_same_v<T, float>) {
            memcpy(embTable, tokenEmb, size * sizeof(T));
        } else if constexpr (std::is_same_v<T, float16_t>) {
            float16_t::cvt_float_to_float16(tokenEmb, embTable, size);
        } else {
            printf("Type %s not supported!\n", typeid(T).name());
            exit(-1);
        }
    }

    void setWeights(const std::string &weightPath) { loadWeight(weightPath, embTable, vocabSize * hiddenSize); }

    // tokenIds ia a 2-dimension array with batchSize rows, and seqLen cols
    template <typename OutT>
    void forward(int *tokenIds, OutT *output, int tokenSize) {
        xft::tokenEmbedding<OutT, T>(output, tokenIds, embTable, tokenSize, hiddenSize);
    }

    int getVocabSize() { return vocabSize; }

    int getHiddenSize() { return hiddenSize; }

private:
    // Embedding like:
    // self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    int vocabSize;
    int hiddenSize;

    T *embTable = nullptr;
};
