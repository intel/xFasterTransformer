#include "alibi_embedding.h"
#include <cmath>
#include "compile_util.h"

bool AlibiEmbedding::initialized = false;

AlibiEmbedding::AlibiEmbedding(const int heads_num, const int seq_len)
{
    max_len = seq_len;
    max_head_nums = heads_num;
    alibi_get_relative_pos(max_len);
    alibi_get_slope(max_head_nums);
    initialized = true;
}

void AlibiEmbedding::alibi_get_bias(const int headIdx, const int seq_len, float *bias_matrx)
{
    REQUIRES(initialized == true, "Alibi Embedding ERROR, Alibi is not initialized.");
    REQUIRES(headIdx < max_head_nums, "Alibi Embedding ERROR, headIdx is exceeds max head nums.");
    if (seq_len > max_len)
    {
        max_len = seq_len;
        alibi_get_relative_pos(max_len);
    }
    for (size_t i = 0; i < seq_len; i++)
    {
        for (size_t j = 0; j < seq_len; j++)
        {
            int index = i * seq_len + j;
            bias_matrx[index] = pos_matrix[index] * slope_m[headIdx];
        }
    }
}

void AlibiEmbedding::alibi_get_relative_pos(const int seq_len)
{
    pos_matrix = (int *)aligned_alloc(64, seq_len * seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++)
    {
        for (int j = 0; j < seq_len; j++)
        {
            pos_matrix[i * seq_len + j] = j - i;
        }
    }
}

void AlibiEmbedding::alibi_get_slope(const int heads_num)
{
    slope_m = (float *)aligned_alloc(64, heads_num * sizeof(float));
    float x = std::pow(2, 8);
    x = std::pow(x, 1.0 / heads_num);
    for (int i = 0; i < heads_num; i++)
    {
        slope_m[i] = 1 / std::pow(x, i + 1);
    }
}