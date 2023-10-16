#include <iostream>
#include "gptq.h"

int main() {
    float data[9] = {
            4.0, -12.0, -16.0,
            12.0, 37.0, -43.0,
            -16.0, -43.0, 98.0 };
    Tensor<float> A(&data[0], 3, 3, 3);
    A.print();
    cholesky_decompose(A);
    A.print();
    cholesky_inverse(A);
    A.print();
    return 0;
}