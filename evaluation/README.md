# Accuracy Evalution
xFasterTransformer supports a different model format than huggingface. This module are used to evaluate accuracy on the datasets for xFasterTransformer inference engine on CPU.
## Step 1: Compile xFT with enabling evaluation module.
cmake -DXFT_BUILD_EVALUATION=ON ..

## Step 2: Download the datasets to local (in case huggingface hub can not be connected)
wget https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl

## Step 3: Run testing script corresponding to the model.
After that, modify the config parameters in the scripts and run. You will see the accuracy report and dump files in the output directory.
| Params          | Use                                   |
| ----------------- | --------------------------------------------------------- |
| TOKEN_NAME        | config files and tokenizer models from huggingface        |
| MODEL_NAME        | xFT format model weights                                  |
| TRUST_REMOTE_CODE | True for chatglm,baichuan family models                   |
| TASKS             | dataset names like lambada_openai or boolq                |
| DATA_FILES        | Local path of json-type data files corresponding to TASKS |
| LIMIT             | ONLY FOR TESTING. REAL METRICS SHOULD Be set 0            |
| BATCH_SIZE        | batch size                                                |


```bash
    sh run.sh 1 48 run_model.sh
```
