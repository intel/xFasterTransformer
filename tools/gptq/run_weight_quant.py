import os
import torch

from gptq import *

# TODO: move to tests/ut folder in xFT
if __name__ == '__main__':
    weight = torch.tensor([
        [0.123456, 1.234567, 3.456789],
        [4.567891, 5.678912, 6.789123],
        [7.891234, 8.912345, 9.123456]
    ]).float()

    llm_gptq = LLM_GPTQ(weight, 8, False)
    quantized_weight, scale, zero = llm_gptq.fasterquant()
    print("quantized weight is ")
    print(quantized_weight)
    print("scale is ")
    print(scale)
    print("zero is ")
    print(zero)
    