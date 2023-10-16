#include "gptq.h"

void GPTQ::add_batch(const Tensor<float> &input) {
    int tmp = input.rows;
    self_multiply(H, nsamples / (nsamples + tmp));
    nsamples += tmp;
    matmul_AAT(2.0 / nsamples, input, 1.0, H);
}

// Matrix W:
//  ___________________________________
// |                                   |
// |                                   |
// |___________________________________| <--- i1
// |                                   |
// |___________________________________| <--- i
// |                                   |
// |            W_part1                |
// |___________________________________| <--- i2
// |                                   |
// |            W_part2                |
// |___________________________________|

// Matrix H (shape: W.rows * W.rows):
//  ___________________________________
// |       |      |         |           |
// |       |      |         |           |
// |_______|______|_________|___________| <--- i1
// |       |      |         |           |
// |_______|______|_________|___________| <--- i
// |       |      |         |           |
// |       |      |         |           |
// |_______|______|_________|___________| <--- i2
// |       |      |         |           |
// |       |      | H_part  |           |
// |_______|______|_________|___________|
//         i1     i         i2

void GPTQ::fasterquant(Tensor<int> &Q_int, Tensor<float> &Q_float, Tensor<float> &scale, Tensor<int> &zero,
        int blocksize, float percdamp, int groupsize, bool actorder) {

    quantizer->find_params(W, scale, zero);

    for (int i = 0; i < H.columns; ++i) {
        if (H.get(i, i) == 0) { H.get(i, i) = 1; }
    }
    if (actorder) {
        // TODO
    }
    float H_diag_mean = 0;
    for (int i = 0; i < H.columns; ++i) {
        H_diag_mean += H.get(i, i);
    }
    H_diag_mean /= H.columns;
    float damp = H_diag_mean * percdamp;
    for (int i = 0; i < H.columns; ++i)
        H.get(i, i) += damp;
    cholesky_decompose(H);
    cholesky_inverse(H);
    cholesky_decompose(H);

    Tensor<float> Losses(W.rows, W.columns, true);

    int i1, i2, count;
    for (i1 = 0; i1 < W.rows; i1 += blocksize) {
        i2 = std::min(i1 + blocksize, W.rows);
        count = i2 - i1;
        Tensor<float> Err1(count, W.columns, true);
        for (int i = 0; i < count; ++i) {
            Tensor<int> q_int(Q_int.data + (i1 + i) * Q_int.columns, Q_int.columns);
            Tensor<float> q_float(Q_float.data + (i1 + i) * Q_float.columns, Q_float.columns);
            Tensor<float> w(W.data + (i1 + i) * W.columns, W.columns);
            Tensor<float> losses(Losses.data + (i1 + i) * Losses.columns, Losses.columns);
            Tensor<float> err1(Err1.data + i * Err1.columns, Err1.columns);
            float d = H.get(i1 + i, i1 + i);

            if (groupsize != -1) {
                if ((i1 + i) % groupsize == 0) {
                    int row_start = i1 + i;
                    int row_end = std::min(i1 + i + groupsize, W.rows);
                    Tensor<float> W_part(std::move(W), row_start, row_end, 0, W.stride);
                    quantizer->find_params(W_part, scale, zero);
                }
            }

            quantize_to_int(q_int, w, scale, zero, quantizer->get_maxq());
            dequantize_to_float(q_float, q_int, scale, zero);

            minus_multiply(err1, w, q_float, 1 / d);
            square_multiply(losses, err1, 0.5);

            // Part of H. A column vector. Shape: (i2 - i1 - i) * 1
            Tensor<float> h(std::move(H), i1 + i, i2, i1 + i, i1 + i + 1);
            // Part of W. Shape: (i2 - i1 - i) * W.columns
            Tensor<float> W_part1(std::move(W), i1 + i, i2, 0, W.columns);
            matmul(false, false, -1.0, h, err1, 1.0, W_part1);
        }
        // Part of H. Shape: (i2 - i1) * (W.rows - i2)
        Tensor<float> H_part(std::move(H), i2, H.columns, i1, i2);
        // Part of W. Shape: (W.rows - i2) * W.columns
        Tensor<float> W_part2(std::move(W), i2, W.rows, 0, W.columns);
        matmul(false, false, -1.0, H_part, Err1, 1.0, W_part2);

        Losses.free();
    }
}