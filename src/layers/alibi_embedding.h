#pragma once
#include <iostream>

class AlibiEmbedding {
public:
    AlibiEmbedding(const int heads_num, const int seq_len);

    ~AlibiEmbedding() {
        max_len = 0;
        max_head_nums = 0;
        free(pos_matrix);
        free(slope_m);
    }

    void alibi_get_relative_pos(const int seq_len);

    void alibi_get_slope(const int heads_num);

    // headIdx is [0,n]
    void alibi_get_bias(const int headIdx, const int seq_len, float *bias_matrx);

private:
    static bool initialized;
    int max_len = 0;
    int max_head_nums = 0;
    int *pos_matrix;
    float *slope_m;
};
