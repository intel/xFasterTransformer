#pragma once
#include "float16.h"
#include "transformer_ctx.h"

template <typename T>
class OptEmbedding {
public:
    OptEmbedding(DecoderContext *ctx) {
        this->vocabSize = ctx->vocabSize;
        this->embeddingSize = ctx->embeddingSize;
        this->maxPositions = ctx->maxPositions;
        this->hiddenSize = ctx->hiddenSize;
    }

    void setWeights(float *tokenEmb, float *positionEmb) {
        int size1 = vocabSize * embeddingSize;
        embTable = (T *)aligned_alloc(64, size1 * sizeof(T));

        int size2 = maxPositions * hiddenSize;
        positionalTable = (T *)aligned_alloc(64, size2 * sizeof(T));

        if constexpr (std::is_same_v<T, float>) {
            memcpy(embTable, tokenEmb, size1 * sizeof(T));
            memcpy(positionalTable, positionEmb, size2 * sizeof(T));
        } else if constexpr (std::is_same_v<T, float16_t>) {
            float16_t::cvt_float_to_float16(tokenEmb, embTable, size1);
            float16_t::cvt_float_to_float16(positionEmb, positionalTable, size2);
        } else {
            printf("Type %s not supported!\n", typeid(T).name());
            exit(-1);
        }
    }

    // TODO: mask is not considered
    // tokenIds and positions are 2-dimension array with batchSize rows, and seqLen cols
    void forward(int *tokenIds, int *positions, float *output, int batchSize, int seqLen) {
        if (embeddingSize != hiddenSize) {
            printf("Not supported yet: embeddingSize != hiddenSize\n");
            exit(-1);
        }

        int row = 0;

        if constexpr (std::is_same_v<T, float16_t>) {
            for (int i = 0; i < batchSize; ++i) {
                for (int j = 0; j < seqLen; ++j) {
                    // Embedding
                    int id = tokenIds[i * seqLen + j];
                    float16_t::cvt_float16_to_float(
                            embTable + id * embeddingSize, output + row * hiddenSize, embeddingSize);

                    // Positional embedding
                    int pos = positions[i * seqLen + j];
                    // # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
                    // # and adjust num_embeddings appropriately. Other models don't have this hack
                    // Do not add the offset if the embedding table is already handled it (like FasterTransformer)
                    //pos += 2;
                    float16_t::float_add_float16(output + row * hiddenSize, positionalTable + pos * hiddenSize,
                            output + row * hiddenSize, hiddenSize);

                    row += 1;
                }
            }
        } else {
            printf("Type %s not supported!\n", typeid(T).name());
            exit(-1);
        }
    }

    int getVocabSize() { return vocabSize; }

    int getEmbeddingSize() { return embeddingSize; }

    int getMaxPositions() { return maxPositions; }

    int getHiddenSize() { return hiddenSize; }

private:
    // Embedding in OPT models are like:
    // self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
    // self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
    int vocabSize;
    int embeddingSize;
    int maxPositions;
    int hiddenSize;

    T *embTable;
    T *positionalTable;
};
