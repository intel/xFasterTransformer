#include "quantizer.h"
#include <vector>

void Quantizer::find_params(const Tensor<float>& weight, Tensor<float>& scale, Tensor<int>& zero) {
    Tensor<float> wmin(weight.columns);
    Tensor<float> wmax(weight.columns);    
    for (int w = 0; w < weight.columns; ++w) {
        wmin[w] = weight[w];
        wmax[w] = weight[w];
    }
    for (int h = 1; h < weight.rows; ++h) {
        for (int w = 0; w < weight.columns; ++w) {
            wmin[w] = weight.get(h, w) < wmin[w] ? weight.get(h, w) : wmin[w];
            wmax[w] = weight.get(h, w) > wmax[w] ? weight.get(h, w) : wmax[w];
        }
    }
    for (int w = 0; w < weight.columns; ++w) {
        wmin[w] = wmin[w] < 0 ? wmin[w] : 0;
        wmax[w] = wmax[w] > 0 ? wmax[w] : 0;
    }

    for (int i = 0; i < wmin.size; ++i) {
        if (sym) {
            float wmin_abs = abs(wmin[i]);
            wmax[i] = wmin_abs > wmax[i] ? wmin_abs : wmax[i];
            if (wmin[i] < 0) {
                wmin[i] = -1 * wmax[i];
            }
        }
        if (wmin[i] == 0 && wmax[i] == 0) {
            wmin[i] -= 1;
            wmax[i] += 1;
        }
    }

    for (int i = 0; i < weight.columns; ++i) {
        scale[i] = (wmax[i] - wmin[i]) / maxq;
        if (sym) {
            zero[i] = (maxq + 1) / 2;
        } else {
            zero[i] = round(-1 * wmin[i] / scale[i]);
        }
    }

    if (mse) {
        Tensor<float> scale1(weight.columns);
        Tensor<int> zero1(weight.columns);
        Tensor<int> Qint(weight.rows, weight.columns);
        Tensor<float> Qfloat(weight.rows, weight.columns);
        Tensor<float> err(weight.rows);
        std::vector<bool> best_visited(weight.rows, false);
        std::vector<float> best(weight.rows);
        for (int i = 0; i < round(maxshrink * grid); ++i) {
            float p = 1 - 1.0 * i / grid;
            minus_multiply(scale1, wmax, wmin,  p / maxq);
            if (sym) {
                zero1.set(zero);
            } else {
                divide_multiply(zero1, wmin, scale1, -1.0 * p);
            }
            quantize_to_int(Qint, weight, scale1, zero1, maxq);
            dequantize_to_float(Qfloat, Qint, scale1, zero1);
            self_minus(Qfloat, weight);
            self_abs(Qfloat);
            self_pow(Qfloat, norm);
            row_sum(err, Qfloat);
            for (int i = 0; i < weight.rows; ++i) {
                if (!best_visited[i]) {
                    best[i] = err[i];
                    best_visited[i] = true;
                } else {
                    if (err[i] < best[i]) {
                        best[i] = err[i];
                        scale[i] = scale1[i];
                        zero[i] = zero1[i];
                    }
                }
            }
        }
        scale1.free();
        zero1.free();
        Qint.free();
        Qfloat.free();
        err.free();
    }

    if (!perchannel) {
        // TODO
    }

    wmin.free();
    wmax.free();
}