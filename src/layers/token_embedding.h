#pragma once
#include "float16.h"
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

    // tokenIds ia a 2-dimension array with batchSize rows, and seqLen cols
    void forward(int *tokenIds, float *output, int batchSize, int seqLen) {
        if constexpr (std::is_same_v<T, float>) {
            for (int i = 0; i < batchSize * seqLen; ++i) {
                int id = tokenIds[i];
                memcpy(output + i * hiddenSize, embTable + id * hiddenSize, hiddenSize * sizeof(float));
            }
        } else if constexpr (std::is_same_v<T, float16_t>) {
            for (int i = 0; i < batchSize * seqLen; ++i) {
                int id = tokenIds[i];
                float16_t::cvt_float16_to_float(embTable + id * hiddenSize, output + i * hiddenSize, hiddenSize);
            }
        } else {
            printf("Type %s not supported!\n", typeid(T).name());
            exit(-1);
        }
    }

    int getVocabSize() { return vocabSize; }

    int getHiddenSize() { return hiddenSize; }

private:
    // Embedding like:
    // self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
    int vocabSize;
    int hiddenSize;

    T *embTable;
};
