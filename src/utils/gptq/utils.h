// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#pragma once

#include <iostream>
#include <math.h>

// TODO: Replace Tensor structure with xFT's Matrix
template <typename T>
class Tensor {
public:
    Tensor() {
        data = new T;
        rows = 1;
        columns = 1;
        stride = 1;
        size = 1;
    }

    Tensor(int size, bool zero = false) : rows(1), columns(size), stride(size), size(size) {
        data = new T[size];
        if (zero) set_zero();
    }

    Tensor(int rows, int columns, bool zero = false) : rows(rows), columns(columns), stride(columns) {
        size = rows * columns;
        data = new T[size];
        if (zero) set_zero();
    }

    Tensor(T *data, int size) : data(data), rows(1), columns(size), stride(size), size(size) {}

    Tensor(T *data, int rows, int columns, int stride) : data(data), rows(rows), columns(columns), stride(stride) {
        size = rows * columns;
    }

    Tensor(const Tensor<T> &other)
        : data(other.data), rows(other.rows), columns(other.columns), stride(other.stride), size(other.size) {}
    Tensor<T> &operator=(const Tensor<T> &other) {
        data = other.data;
        rows = other.rows;
        columns = other.columns;
        stride = other.stride;
        size = other.size;
    }

    Tensor(const Tensor<T> &other, int rows_start, int rows_end, int columns_start, int columns_end) {
        rows = rows_end - rows_start;
        columns = columns_end - columns_start;
        stride = other.stride;
        size = rows * columns;
        data = new T[size];
        for (int h = rows_start; h < rows_end; ++h) {
            for (int w = columns_start; w < columns_end; ++w) {
                data[(h - rows_start) * columns + (w - columns_start)] = other[h * stride + w];
            }
        }
    }

    Tensor(Tensor<T> &&other, int rows_start, int rows_end, int columns_start, int columns_end) {
        rows = rows_end - rows_start;
        columns = columns_end - columns_start;
        stride = other.stride;
        size = rows * columns;
        data = other.data + rows_start * other.stride + columns_start;
    }

    ~Tensor() {}

    T &operator[](int index) const { return data[index]; }

    T &get(int index) const { return data[index]; }

    T &get(int _row, int _column) const { return data[_row * stride + _column]; }

    void set(const Tensor<T> &other) {
        rows = other.rows;
        columns = other.columns;
        stride = other.stride;
        size = other.size;
        for (int i = 0; i < size; ++i)
            data[i] = other[i];
    }

    void set(Tensor<T> &&other) {
        data = other.data;
        rows = other.rows;
        columns = other.columns;
        stride = other.stride;
        size = other.size;
    }

    void set(int length, bool zero = false) {
        rows = 1;
        columns = length;
        stride = length;
        size = length;
        data = new T[size];
        if (zero) set_zero();
    }

    void set(int _rows, int _columns, bool zero = false) {
        rows = _rows;
        columns = _columns;
        stride = _columns;
        size = rows * columns;
        data = new T[size];
        if (zero) set_zero();
    }

    void set_zero() {
        for (int i = 0; i < size; ++i)
            data[i] = 0;
    }

    void free() { delete[] data; }

    void print() {
        std::cout << "Tensor shape is " << rows << "x" << columns << std::endl;
        std::cout << "[" << std::endl;
        for (int h = 0; h < rows; ++h) {
            std::cout << "[ ";
            for (int w = 0; w < columns; ++w) {
                if (w < columns - 1)
                    std::cout << get(h, w) << ", ";
                else
                    std::cout << get(h, w);
            }
            if (h < rows - 1)
                std::cout << "]," << std::endl;
            else
                std::cout << "]" << std::endl;
        }
        std::cout << "]" << std::endl;
    }

    T *data;
    int size;
    int rows;
    int columns;
    int stride;
};

void quantize_to_int(
        Tensor<int> &result, const Tensor<float> &input, const Tensor<float> &scale, const Tensor<int> &zero, int maxq);

void dequantize_to_float(
        Tensor<float> &result, const Tensor<int> &input, const Tensor<float> &scale, const Tensor<int> &zero);

void minus_multiply(Tensor<float> &result, const Tensor<float> &input1, const Tensor<float> &input2, float d);

void square_multiply(Tensor<float> &result, const Tensor<float> &input, float d);

void self_multiply(Tensor<float> &input, float d);

void self_minus(Tensor<float> &input1, const Tensor<float> &input2);

void self_abs(Tensor<float> &input);

void self_pow(Tensor<float> &input, float d);

void row_sum(Tensor<float> &output, Tensor<float> &input);

void divide_multiply(Tensor<int> &result, const Tensor<float> &input1, const Tensor<float> &input2, float d);

void self_transpose(Tensor<float> &H);

void matmul(bool trans_A, bool trans_B, float alpha, const Tensor<float> &A, const Tensor<float> &B, float beta,
        Tensor<float> &C);

void matmul_AAT(float alpha, const Tensor<float> &A, float beta, Tensor<float> &B);

void cholesky_decompose(Tensor<float> &H, bool input_upper = false, bool output_upper = false);

void cholesky_inverse(Tensor<float> &H, bool input_upper = false);
