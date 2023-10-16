#include "utils.h"

void quantize_to_int(Tensor<int>& result, const Tensor<float>& input, const Tensor<float>& scale, const Tensor<int>& zero, int maxq) {
    for (int h = 0; h < input.rows; ++h) {
        for (int w = 0; w < input.columns; ++w) {
            int tmp = round(input.get(h, w) / scale[w]) + zero[w];
            tmp = tmp < 0 ? 0 : tmp;
            tmp = tmp > maxq ? maxq : tmp;
            result.get(h, w) = tmp; 
        }
    }
}

void dequantize_to_float(Tensor<float>& result, const Tensor<int>& input, const Tensor<float>& scale, const Tensor<int>& zero) {
    for (int h = 0; h < input.rows; ++h) {
        for (int w = 0; w < input.columns; ++w) {
            result.get(h, w) = scale[w] * (input.get(h, w) - zero[w]); 
        }
    }
}

void minus_multiply(Tensor<float>& result, const Tensor<float>& input1, const Tensor<float>& input2, float d) {
    for (int i = 0; i < input1.size; ++i) {
        result[i] = (input1[i] - input2[i]) * d;
    }
}

void square_multiply(Tensor<float>& result, const Tensor<float>& input, float d) {
    for (int i = 0; i < input.size; ++i) {
        result[i] = input[i] * input[i] * d;
    }
}

void self_multiply(Tensor<float>& input, float d) {
    for (int i = 0; i < input.size; ++i) {
        input[i] *= d;
    }
}

void self_minus(Tensor<float>& input1, const Tensor<float>& input2){
    for (int i = 0; i < input1.size; ++i) {
        input1[i] -= input2[i];
    }
}

void self_abs(Tensor<float>& input) {
    for (int i = 0; i < input.size; ++i) {
        input[i] = abs(input[i]);
    }
}

void self_pow(Tensor<float>& input, float d) {
    for (int i = 0; i < input.size; ++i) {
        input[i] = pow(input[i], d);
    }
}

void row_sum(Tensor<float>& output, Tensor<float>& input) {
    for (int h = 0; h < input.rows; ++h) {
        float tmp = 0;
        for (int w = 0; w < input.columns; ++w) {
            tmp += input.get(h, w);
        }
        output[h] = tmp;
    }
}

void divide_multiply(Tensor<int>& result, const Tensor<float>& input1, const Tensor<float>& input2, float d) {
    for (int i = 0; i < input1.size; ++i) {
        result[i] = round(d * input1[i] / input2[i]);
    }
}

void self_transpose(Tensor<float>& H) {
    for (int i = 1; i < H.rows; ++i) {
        for (int j = 0; j < i; ++j) {
            float tmp = H.get(i, j);
            H.get(i, j) = H.get(j, i);
            H.get(j, i) = tmp;
        }
    }
}

// C = alpha * A * B + beta * C
void matmul(bool trans_A, bool trans_B, float alpha, const Tensor<float>& A, const Tensor<float>& B, float beta, Tensor<float>& C) {
    int M = trans_A ? A.columns : A.rows;
    int K = trans_A ? A.rows : A.columns;
    int N = trans_B ? B.rows : B.columns;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float tmp = 0;
            for (int k = 0; k < K; ++k) {
                float value1 = trans_A ? A.get(k, m) : A.get(m, k);
                float value2 = trans_B ? B.get(n, k) : B.get(k, n);
                tmp += value1 * value2;
            }
            C.get(m, n) = alpha * tmp + beta * C.get(m, n);
        }
    }
}

// B = alpha * A * A^T + beta * B
void matmul_AAT(float alpha, const Tensor<float>& A, float beta, Tensor<float>& B) {
    for (int m = 0; m < A.rows; ++m) {
        for (int n = 0; n <= m; ++n) {
            float tmp = 0;
            for (int k = 0; k < A.columns; ++k) {
                tmp += A.get(m, k) * A.get(n, k);
            }
            B.get(m, n) = alpha * tmp + beta * B.get(m, n);
        }
    }
}

// H is a symmetric, positive definite matrix.
// If output_upper = false, H = P * P^T. P is a lower triangular matrix from cholesky decomposition. The lower part of H will be overwritten by P.
// If output_upper = true, H = P^T * P. P is a upper triangular matrix from cholesky decomposition. The upper part of H will be overwritten by P.
void cholesky_decompose(Tensor<float>& H, bool input_upper, bool output_upper) {
    int length = H.rows;
    for (int i = 0; i < length; ++i) {
        // H_ij = ( H_ij - sigma_{k=0~j-1} (H_ik * H_jk) ) / H_jj
        for (int j = 0; j < i; ++j) {
            float tmp = input_upper ? H.get(j, i) : H.get(i, j);
            for (int k = 0; k < j; ++k) {
                tmp -= H.get(i, k) * H.get(j, k); 
            }
            H.get(i, j) = tmp / H.get(j, j);
        }
        // H_ii = sqrt( H_ii - sigma_{k=0~i-1} (H_ik * H_ik) )
        float tmp = H.get(i, i);
        for (int k = 0; k < i; ++k) {
            tmp -= H.get(i, k) * H.get(i, k);
        }
        H.get(i, i) = sqrt(tmp);
    }
    if (output_upper) self_transpose(H);
}

// Input is a matrix from cholesky decomposition. We call it P and suppose that it is a upper triangle matrix.
// If input_upper = true, The upper triangle of H is P
// If input_upper = false, The lower triangle of H is P. We need to transpose to upper triangle.
// First calculate P's inverse matrix Pinv. The upper triangle of H will be overwritten by Pinv.
// Then, calculate Hinv = Pinv * Pinv^T. Since Hinv is a symmetric matrix, we only calculate its lower triangle and store in H. 
// Finally, copy the lower triangle to upper triangle.
void cholesky_inverse(Tensor<float>& H, bool input_upper) {
    int length = H.rows;
    // Make sure P is in the lower triangle of H.
    if (input_upper) self_transpose(H);
    // After inversion, the upper triangle of H is Pinv.
    for (int i = 0; i < length; ++i) {
        // H_ii = 1 / H_ii
        H.get(i, i) = 1 / H.get(i, i);
        // H_ij = - sigma_{k=i~j-1} (H_ik * H_jk) / H_jj
        for (int j = i + 1; j < length; ++j) {
            float tmp = 0;
            for (int k = i; k < j; ++k) {
                tmp -= H.get(i, k) * H.get(j, k);
            }
            H.get(i, j) = tmp / H.get(j, j);
        }
    }
    // Calculate Hinv = Pinv * Pinv^T. Since Hinv is a symmetric matrix, we only calculate its lower triangle and store in H.
    for (int i = 0; i < length; ++i) {
        for (int j = 0; j <= i; ++j) {
            float tmp = 0;
            for (int k = i; k < length; ++k) {
                tmp += H.get(i, k) * H.get(j, k);
            }
            H.get(i, j) = tmp;
        }
    }

    // Copy the lower triangle to upper triangle.
    for (int i = 1; i < length; ++i) {
        for (int j = 0; j < i; ++j) {
            H.get(j, i) = H.get(i, j);
        }
    }
}