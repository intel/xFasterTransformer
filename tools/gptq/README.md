# GPTQ

A demo for gptq.

## How to use in model

```bash
python run_model_quant.py --input_model_path=/data/Llama-2-7b-cpu --output_model_path=/data/Llama-2-quantized-7b-cpu --model_type=llama2 --wbits=8

```

## How to use in only weight

```python
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
```