# xFT quantization with AutoAWQ + AutoGPTQ

A Guide for xFT quantization with AutoAWQ + AutoGPTQ on llama2-7b.

## description
xFT can use AutoAWQ and AutoGPTQ to quantize models, and these two quantization
techniques can be used individually or together.
Typically, AutoAWQ and AutoGPTQ only works on GPU. We have some hacks to make it
running on CPU.
AutoAWQ will perform a activation-aware quatization. It will search the scale and 
zero for weights. And set `export_compatible` to export the AWQ searched model.
After that this model can be quantized again by AutoGPTQ.

    ┌─────────┐ Float┌──────────┐Int ┌─────────────────┐
    │  Model  ├─────►│ AutoAWQ  ├───►│ quantized model │
    └─────────┘      └────┬─────┘    └─────────────────┘
                          │
                          │
                          │
                          │Float       Better accuracy
                     ┌────▼─────┐Int ┌─────────────────┐
                     │ AutoGPTQ ├───►│ quantized model │
                     └──────────┘    └─────────────────┘

In our test, AutoAWQ + AutoGPTQ will improve the accuracy of Llama2-7B on Lambada.

## AutoAWQ
install AutoAWQ==0.2.5.
```bash
pip install autoawq==0.2.5
```

### quantize model (llama2-7b as an example)
run awq quantization on llama2-7b and export model to `<path/to/Llama-2-7b-hf-awq-export>`
```python
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "<path/to/Llama-2-7b-hf>"
# Load model
model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM" }

model.quantize(
    tokenizer,
    quant_config=quant_config,
    export_compatible=True
)

export_path = "<path/to/Llama-2-7b-hf-awq-export>"
model.save_quantized(export_path) 
```

## AutoGPTQ
clone AutoGPTQ source code
```bash
cd 3rdparty
git clone https://github.com/PanQiWei/AutoGPTQ.git
cd AutoGPTQ
git reset --hard caf343b1826301c15f90e2e119cabd0347acfcdf
```
### AutoGPTQ on CPU
install AutoGPTQ from source
```bash
git apply ../tools/quantization/autogptq-cpu.patch
BUILD_CUDA_EXT=0 pip install -vvv --no-build-isolation -e .
```

### quantize model with awq searched model
In examples/quantization/basic_usage_wikitext2.py:
Set the input `pretrained_model_dir` to '<path/to/Llama-2-7b-hf-awq-export>' 
Set the output `quantized_model_dir` to '<path/to/Llama-2-7b-hf-awq-gptq-4bit-128g>' 
```bash
cd examples/quantization
python basic_usage_wikitext2.py
```
After that the GPTQ quantized model would be stored to `quantized_model_dir` according to this script.

## convert GPTQ quantized model into xFT format

### install optimum
```
pip install optimum==1.21.3
```

set `MODEL_PATH` to `quantized_model_dir` from AutoGPTQ output.
set `XFT_MODEL_PATH` to converted xFT model path.
```python
import xfastertransformer as xft

MODEL_PATH="/data/model/llama2-7b-int4"
XFT_MODEL_PATH="/data/model/llama2-7b-int-xft"
print(xft.LlamaConvert().convert(MODEL_PATH, XFT_MODEL_PATH, from_quantized_model="gptq"))
```
