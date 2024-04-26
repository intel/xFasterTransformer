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
#include <cmath>
#include "float16.h"
#include "transformer_ctx.h"

template <typename T>
class GemmaTokenEmbedding {
public:
    GemmaTokenEmbedding(DecoderContext *ctx) {
        this->vocabSize = ctx->vocabSize;
        this->hiddenSize = ctx->hiddenSize;
    }

    void setWeights(float *tokenEmb) {
        int size = vocabSize * hiddenSize;
        embTable = (T *)aligned_alloc(64, size * sizeof(T));

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
        __m512 vdim = _mm512_set1_ps(sqrtf(this->hiddenSize));
        constexpr int kStep = 16;
        int blockSize = hiddenSize / kStep;
        int remainder = hiddenSize % kStep;

#pragma omp parallel for
        for (int i = 0; i < tokenSize; ++i) {
            int id = tokenIds[i];
            auto src = this->embTable + id * hiddenSize;
            auto dst = output + i * hiddenSize;
            for (int j = 0; j < blockSize; ++j) {
                __m512 v = load_avx512(0xffff, src + j * kStep);
                // normalized as https://github.com/huggingface/transformers/blob/2a9b1f80c45cab19b542bc7cc004937d39d6f6fb/src/transformers/models/gemma/modeling_gemma.py#L880-L882
                store_avx512(dst + j * kStep, 0xffff, _mm512_mul_ps(v, vdim));
            }

            if (remainder != 0) {
                __mmask16 mask = 0xFFFF >> (kStep - remainder);
                __m512 v = load_avx512(mask, src + hiddenSize - remainder);
                store_avx512(dst + hiddenSize - remainder, mask, _mm512_mul_ps(v, vdim));
            }
        }
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
