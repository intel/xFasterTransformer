# CHANGELOG

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