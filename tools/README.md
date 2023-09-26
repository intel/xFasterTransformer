# Convert tools
xFasterTransformer supports a different model format than huggingface, compatibe with NVIDIA FasterTransformer's format. The tools are used to dump Huggingface models parameters on every layer to binary for xFasterTransformer code on CPU.
## Step 1: Download the huggingface format model firstly.

## Step 2: Run convert script corresponding to the model.
After that, convert the model into xFasterTransformer format using the script. You will see many bin files in the output directory.
```bash
    python chatglm_convert.py -i ${HF_DATASET_DIR} -o  ${OUTPUT_DIR}

```