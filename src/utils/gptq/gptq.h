#ifndef __LLM_OPT_GPTQ_H__
#define __LLM_OPT_GPTQ_H__

#include "quantizer.h"

class GPTQ {
public:
    GPTQ(const Tensor<float>& weight, int wbits): wbits(wbits) {
        W.set(std::move(weight));
        quantizer = new Quantizer(wbits, perchannel, sym, mse);
        H.set(W.rows, W.rows, true);
    }

    void add_batch(const Tensor<float>& input);

    void fasterquant(Tensor<int>& Q_int, Tensor<float>& Q_float,  Tensor<float>& scale, Tensor<int>& zero,
                     int blocksize = 128, float percdamp = 0.01, int groupsize = -1, bool actorder = false);
    int wbits;
    bool perchannel = true;
    bool sym = false;
    bool mse = false;
    Tensor<float> W;
    Tensor<float> H;
    int nsamples = 0;
    Quantizer* quantizer;
};

# endif