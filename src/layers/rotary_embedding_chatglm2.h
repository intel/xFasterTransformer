#pragma once
#include <cmath>
#include <cstring>
#include <iostream>

/*  Sample:
        int bs = 2 headnum = 3 seq = 4  dim = 6;
        int max_len = 10;
        int pos_ids[4] = {2,0,1,3}; //  seq = 4 , Each batch have same value
        int pos_shape[2] = {bs, seq};
        float x[144] = {0, 1, 1,...}; // bs * h * seq * dim = 144
        int xshape[4] = {bs,headnum,seq,dim};
        Forward
        LlamaRotaryEmbedding emb(dim, seq);
        float *embd = emb.forward(x, x_shape, pos_ids, pos_shape);
*/

class ChatGLM2RotaryEmbedding {
public:
    ChatGLM2RotaryEmbedding(const int dim, const int max_position_embeddings = 2048, const float base = 10000);

    ~ChatGLM2RotaryEmbedding() {}

    void forward(float *buf, int bufStride, int batch_size, int seq_len, int head_num, int multi_query_group_num,
            int hidden_size_per_attention_head, const int *position_ids);

private:
    void glm2CalEmb();

private:
    static bool initialized;
};