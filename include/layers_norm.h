#pragma once

namespace xft {

void invokeLayerNorm(float *output, const float *input, const float *gamma, const float *beta, const int rows,
        const int size, int iStride = -1, int oStride = -1, const float epsilon = 1e-5);

void invokeRmsNorm(float *output, const float *input, const float *weight, int rows, int cols, int iStride = -1,
        int oStride = -1, float epsilon = 1e-6);

// Layer normalization: only support the norm along last dimension
class LayerNorm {
public:
    LayerNorm();
    ~LayerNorm();

    void setWeight(const float *gamma, const float *beta, int size);

    // input and output are in shape of (rows, normSize)
    // TODO: column-wise parallel
    void forward(const float *input, float *output, int rows, int iStride = -1, int oStride = -1);

private:
    int normSize;

    // the weights contains gamma and beta concated together
    float *weights;
};

// Layer normalization: only support the norm along last dimension
class RmsNorm {
public:
    RmsNorm();
    ~RmsNorm();

    void setWeight(const float *w, const float *, int size);

    // input and output are in shape of (rows, normSize)
    void forward(const float *input, float *output, int rows, int iStride = -1, int oStride = -1, float epsilon = 1e-6);

private:
    int normSize;

    // the scale weight
    float *weight;
};

} // namespace xft