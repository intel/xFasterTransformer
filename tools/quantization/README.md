# xFT quantization with AWQ + AutoGPTQ

A Guide for xFT quantization with AWQ + AutoGPTQ on llama2-7b.

## description
xFT can use AWQ and AutoGPTQ to quantize models, and these two quantization
techniques can be used individually or together.
Typically, AWQ and AutoGPTQ only works on GPU. We have some hacks to make it
running on CPU.
AWQ will perform a activation-aware quatization. It will search the scale and 
zero for weights. We have some modification to dump the AWQ searched model.
Then this model can be quantized again by AutoGPTQ.

    ┌─────────┐ Float┌──────────┐Int ┌─────────────────┐
    │  Model  ├─────►│   AWQ    ├───►│ quantized model │
    └─────────┘      └────┬─────┘    └─────────────────┘
                          │
                          │
                          │
                          │Float       Better accuracy
                     ┌────▼─────┐Int ┌─────────────────┐
                     │ AutoGPTQ ├───►│ quantized model │
                     └──────────┘    └─────────────────┘

In our test, AWQ + AutoGPTQ will improve the accuracy of Llama2-7B on Lambada.

## prepare AWQ
clone the llm-awq source code.
```bash
cd 3rdparty
git clone https://github.com/mit-han-lab/llm-awq
cd llm-awq
git reset --hard 398b9661415e6a1f89f65c393a13b7f7047b582a
```

## AWQ on CPU
The llm-awq is targeted for GPU. We have a patch to make it works on CPU.
```bash
git apply ../tools/awq/awq-cpu.patch
pip install -e .
```

## quantize model (llama2-7b as an example)
run awq search on llama2-7b to get scales, zeros and dump model to `awq_model`
```bash
python -m awq.entry --model_path <path/to/Llama-2-7b-hf>  --w_bit 8 --q_group_size 128  --run_awq --dump_awq awq_cache/llama2_7b_w8.pt --dump_model awq_model/
```

# AutoGPTQ
clone AutoGPTQ source code
```bash
cd 3rdparty
git clone https://github.com/PanQiWei/AutoGPTQ.git
cd AutoGPTQ
git reset --hard e4b2493733d69a6e60e22cebc64b619be39feb0e
```
## AutoGPTQ on CPU
install AutoGPTQ from source
```bash
git apply ../tools/awq/autogptq-cpu.patch
pip install -v .
```

## quantize model with awq searched model
change the `pretrained_model_dir` to `awq_model` in examples/quantization/basic_usage_wikitext2.py
```bash
cd examples/quantization
python basic_usage_wikitext2.py
```
After that the GPTQ quantized model would be stored to `quantized_model_dir` according to this script.

## convert quantized model into xFT IR
set `MODEL_PATH` to `quantized_model_dir` from AutoGPTQ output.
set `IR_PATH` to converted xFT IR path.
```bash
python llama_autogptq_convert.py -in_file=${MODEL_PATH} -saved_dir=${IR_PATH} -processes 8
```

## check the accuracy on lambada test set
set `TOKEN_PATH` to llama2-7b path.
set `IR_PATH` to converted xFT IR path.
```bash
cd tools/awq/
python llama2_acc_xft.py --dataset_path lambada_test.jsonl --token_path=${TOKEN_PATH} --model_path=${IR_PATH} --show_progress --dtype=int8 
```
