# CHANGELOG

# [Version v1.6.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.6.0)
v1.6.0 - Llama3 and Qwen2 series models supported.

## Functionality
- Support Llama3 and Qwen2 series models.
- Add INT8 KV cache datatype, using `kv_cache_dtype` params to specify, including `int8`, `fp16`(default) and `fp32`.
- More models enable full BF16 pipline, includes Chatglm2/3 and yarn-llama.
- Add invokeMLPLLaMA FP16 API.
- Support logits output using `forward()` api.

## Dependency
- Bump `transformers` to `4.40.0` to support Llama3 models.

## Performance
- Update xDNN to release `v1.4.6`

## BUG fix
- Fix numeric overflow when calculate softmax in sampling.  
- fix assert bug when concat gate&up.

# [Version v1.5.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.5.0)
v1.5.0 - Gemma series models supported.

## Functionality
- Support Gemma series medels, including Gemma and CodeGemma, and DeepSeek model.
- Llama Converter support convert quantized huggingface model by params `from_quantized_model='gptq` into xFt format INT8/INT4 model files.
- Support loading INT4 data weights directly from local files.
- Optimize memory usage during QWen model conversion, particularly for QWen 72B.

## Dependency
- Bump `transformers` to `4.38.1` to support Gemma models.
- Add `protobuf` to support new behavier in `tokenzier`.

## Performance
- Update xDNN to release `v1.4.5`
- Add GPU kernel library gpuDNN v0.1 to support Intel Arc GPU series. 
- Optimize ROPE perfermance by reducing repeated sin and cos embedding table data.
- Accelerate KVCache copy by increasing parallelism in self attention.
- Accelerate addreduce operation in long sequence case by transposing KVCache and tuned comm.

## BUG fix
- Fix a incorrect computing which should be in float, but was in integer.
- Fix timeline is disordered.
- Fix runtime issue of Qwen when seq_length is bigger than 32768.

# [Version v1.4.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.4.0)
v1.4.0 - Fully BF16 support in Llama for better performance and serving framework support.

## Functionality
- Introduce pure BF16 support to Llama series models, now can use fully BF16 data type to to utilize AMX more effectively when deploying Llama models.
- Add MLServer serving framework support and demo in `serving` directory.
- GCC for compiling release binary files has been updated from GCC 8.5 to GCC 12.
- Introduce pipeline parallel feature for distributing deployment. Enabled by `cmake .. -DWITH_PIPELINE_PARALLEL=ON` in compilation and use `XFT_PIPELINE_STAGE` Marco to define pipeline parallel stages num.
- Deprecate convert tool scripts in `tools` directory and it recommended to using `Convert` in xfastertransformer python wheel.
- Support loading int8 data weights directly from local files.

## Performance
- Update xDNN to release `v1.4.4`.
- Accelerate model weights loading by optimizing cast operation after loading and gain up to 50% speed up.
- Optimize BF16 performance using AMX instruction when batchsize <= 8, and add `XFT_USE_AMX_M` to set threshold of M using AMX instead of AVX512, default `1`.

## Demo & Benchmark
- Update dependency `transformers` requirement from `4.30.0` to `4.36.0` for high risk CVE Vulnerabilities.
- Add distributed inference benchmark script which support deployment across platfrom.
- Add single node platform support in benchmark script.
- Add Yi model web demo.
- Enhance the command-line chat mode in pytorch demo.py, using `--chat true` to enable.

## BUG fix
- Fix calculation issue in Qwen models and enhance LogN support for long token sequence.
- Fix unsync issue in multi-rank model when `do_sample` is enabled.
- Fix Baichuan models calculation and convert issue.
- Fix repetition penalties not taking effect on other batches.

# [Version v1.3.1](https://github.com/intel/xFasterTransformer/releases/tag/v1.3.1)
v1.3.1
## BUG fix
- Fix oneCCL environment is still needed when running in single-rank mode.

# [Version v1.3.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.3.0)
v1.3.0 - Qwen model support enhancement and added support for the SecLLM (YaRN-Llama) model.
## Models
- Introduce SecLLM(YaRN-Llama) model support.
- Integrating the Qwen web demo, enhancing Qwen model support, and fix known issues in the Qwen convert tool.

## Functionality
- Introduce new generation configuration, `repetition_penalty` and `stop_words_ids`.
- Rotary embedding supports BF16 data type now.
- Introduce attention interfaces similar to page attention.
- Add a whitelist to gather timeline events based on filtered events.

## BUG fix
- Fix `libxft_comm_helper.so` can't be found issue in multi-ranks mode.
- Fix assert error in MLP when CAT_MLP opt is enabled.
- Fix a w8a8 crash issue due to buffer size isn't big enough.
- Correct GCC version for AVX512_BF16 instruction set.
- Fix int32 overflow issue for larger size.


# [Version v1.2.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.2.0)
v1.2.0 - Qwen models and much more data types supported.
## Models
- Introduced Qwen models support and added the convert tool for Qwen models.
- ChatGLM3 model is verfied and API supported.

## Performance Optimizations
- Update xDNN to version 1.4.2 to improve performance and support more data types.
- Accelerate first token's generation with BF16-gemm Multi-Head Attention.

## Functionality
- Introduce more data types supports, including `W8A8`, `INT4`, and `NF4`. The hybrid data types between these new data types are supported.
- Add accuracy evaluation script to assess the impact of different precisions on the text generation performance of the model.
- Introduce `XFT_VERBOSE` macro to help profile model performance of each gemm. Set `1` to enable information ouput and default is `0`.  
- Decouple oneCCL and MPI dependencies into a communication helper library. oneCCL environment is no longer needed when running in single-rank mode.


# [Version 1.1.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.1.0)
v1.1.0 - Baichuan models supported.

## Models
- Introduced Baichuan models support and added the convert tool for Baichuan models.

## Performance Optimizations
- Update xDNN to version 1.2.1 to improve performance of BF16 data type with AMX instruction on 4th generation Intel Xeon Scalable processors.
- Improved performance of BF16 data type inference by adding matMul bf16bf16bf16 primitives and optimizing kernel selection strategy.
- Improved performance of the model with unbalanced split allocation.

## Functionality
- Introduced prefix sharing feature.
- Add sample strategy for token search, support temperature, top k, and top P parameter.
- Introduce convert module to xfastertransformer python API.
- Introduced grouped-query attention support for Llama2.
- Auto-detect oneCCL environment and enter single-rank model if oneCCL does not exist.
- Auto-detect oneCCL environment in compilation. If not detected, oneCCL will be built from source.
- Add C++ exit function for multi-rank model. 
- Remove mklml 3rd party dependency.
- Export normalization and position embedding C++ API, including alibi embedding and rotary embedding.
- Introduced `XFT_DEBUG_DIR` environment value to specify the debug file directory.

## BUG fix
- Fix runtime issue of oneCCL shared memory model.
- Fix path concat issue in convert tools.


# [Version 1.0.0](https://github.com/intel/xFasterTransformer/releases/tag/v1.0.0)
This is the 1st official release of xFasterTransformer.🎇🎇🎇

## Support models
- ChatGLM-6B
- ChatGLM2-6B
- Llama 1, both 7B, 33B, and 65B
- Llama 2, both 7B, 13B, and 70B
- Opt larger than 1.3B

## Features
- Support Python and C++ API to integrate xFasterTransformer into the user's own solutions. Example codes are provided to demonstrate the usage.  
- Support hybrid data types such as BF16+FP16 and BF16+INT8 to accelerate the generation of the 1st token, in addition to supporting single data types like FP16, BF16, and INT8.
- Support multiple instances to accelerate model inference, both locally and through the network.
- Support Intel AMX instruction on 4th generation Intel Xeon Scalable processors.
- Support 4th generation Intel Xeon Scalable processors with HBM which has a higher memory bandwidth and shows a much better performance on LLM.
- Provide web demo scripts for users to show the performance of LLM models optimized by xFasterTransformer. 
- Support multiple distribution methods, both PyPI and docker images.