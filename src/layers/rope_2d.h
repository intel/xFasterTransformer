#pragma once
#include <cmath>
#include <cstring>
#include <iostream>
using namespace std;

// 2D rotary embedding for ChatGLM
class RotaryEmbedding2D {
public:
    RotaryEmbedding2D(const int dim, const int max_position_embeddings = 2048, const float base = 10000);

    ~RotaryEmbedding2D() {}

    void forward(float *query, float *key, int qStride, int kStride, const int *qk_shape, const int *positions);

private:
    void prepareEmbedding();

private:
    static bool initialized;
};
