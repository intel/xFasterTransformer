# llm_opt benchmark

This is a solution like NV's FasterTransformer for LLM. It only supply the benchmark reference code. Plese test the performance in the docker.

### Prepare 
- Prepare the models and convert the Huggingface model using the converter tools.
  ```bash
  python ../../tools/chatglm_convert.py -i /data/chatglm-6b -o /data/chatglm-6b/
 
  ```
### Benchmark
- An example of chatglm-6b
  ```bash
  cd chatglm-6b
  sh run-chatglm-6b.sh 
 
  ```
Notes:
- By default, you will get the performance on 1 socket  that "input token=32, output token=32, Beam_width=1, FP16".
- If more datatype and scenarios performance needed, please change the parameters in chatglm-6b.sh
- If system configuration needs modification, please change run-chatglm-6b.sh.
- If you want the custom input, please refer to the repo (example/pytorch/demo.py)


