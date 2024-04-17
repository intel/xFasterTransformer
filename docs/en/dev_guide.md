# xFasterTransformer Development Guide  

This document describes how to add support for a new LLM model in xFasterTransformer, which is a highly optimized LLM inference framework on Xeon.  
Generally, to incorporate support for a new model, we need to:  

- Comprehend the structure of the model. For instance, we need to determine whether Attention and MLP are arranged in serial or in parallel, what type of positional embedding is used, and so on.
- Delve into some details for the implementation. This includes understanding what type of attention mask is used. While most models utilize a casual attention mask, some may differ. We also need to ascertain whether the residual input is the value before or after normalization.
- Select a proper data type to balance the accuracy and performance.

Below content give more details for how to add the support, which is organized into preparing, implementation and verification.

## 1. Preparing: Get familiar with xFT and the new LLM model

### 1.1 How xFT works

xFasterTransformer is an exceptionally optimized solution for large language models (LLM) on the X86 platform, which is similar to FasterTransformer on the GPU platform. xFasterTransformer is able to operate in distributed mode across multiple sockets and nodes to support inference on larger models. Here's a brief overview of how xFT works:

- Optimization for X86 Platforms: xFT is specifically optimized for X86 architectures. The optimization includes leveraging specific hardware features of Xeon processors to accelerate inference tasks.
- Support for Various Models and Data Types: xFT supports a range of popular LLMs and data types, ensuring broad applicability across different AI scenarios. The supported models and data types are detailed in the documentation, including FP16, BF16, INT8, and more.
- Model Conversion: xFT supports converting models from the Hugging Face format to a format compatible with xFT, facilitating the use of a wide range of pre-trained models.
- C++ and Python APIs: xFT provides both C++ and Python APIs, catering to different levels of integration needs. This flexibility allows users to easily adopt xFT in their existing projects or services.
- Integration with Transformers Library: For Python users, xFT offers compatibility with the Hugging Face Transformers library, allowing for easy integration and use of pre-trained models from the Transformers ecosystem.
- Distributed Inference: xFT can operate in distributed mode across multiple sockets and nodes, enabling the inference of larger models that may not fit into the memory of a single machine.
- Example Usage: The documentation includes examples demonstrating how to use xFT for both C++ and Python. These examples cover various scenarios, including single and multi-rank execution, and provide a practical guide to integrating xFT into applications.
- Web Demos and Benchmarking: xFT also provides web demos for popular LLM models and benchmarking tools/scripts to evaluate performance.

Here is the meaning of xFT each code directory:

- src: the source code directory. It consists pytorch/C++ API definitions, models, layers, kernels, searchers and other utilities.
- example and benchmark: references to examples and benchmarks suggest directories containing sample code and scripts to demonstrate the usage of xFasterTransformer and to measure its performance.
- evaluation: contains scripts and tools for evaluating the performance and accuracy of the models supported by xFT
- serving: hold resources related to deploying and serving the models in production environments.
- tests: This directory usually contains unit tests, integration tests, and other testing scripts designed to ensure the functionality and stability.
- 3rdparty: the dependencies required by xFasterTransformer, ensuring that all necessary libraries are available for the build process.
- docs: documentation files. This can range from API documentation, getting started guides, tutorials, and technical specifications of the xFT project.

### 1.2 Add new model steps

To add a new model to xFasterTransformer (xFT), follow these steps:

1. Contribution Guidelines: Follow the contribution guidelines provided in the CONTRIBUTING.md file. This includes ensuring your code adheres to the coding standards, writing meaningful commit messages, and submitting a pull request for review.
2. Model Implementation: Implement the new model in C++ within the xFT framework. In xFT, all LLMs are loaded by `AutoModel`. The parent class `Model` has a `generate()` interface to inference. In `Model::generate()`, there are 3 searcher: `GreedySearch`, `BeamSearch` and `SampleSearch`. Usually you do not need to modify seacher's logic. You only need to pay attention on the `Decoder` implementation in `Model`. This involves creating a new decoder class for your model that inherits from the `CommonDecoder` class provided by xFT, and implement the its forward pass, handling input and output tensors. 
3. Tokenizer Support: If your model requires a specific tokenizer, ensure support for this tokenizer is added. This might involve integrating with existing tokenization libraries or implementing a custom tokenizer.
4. Testing: Write unit tests for your model to ensure it performs as expected. This includes testing the model's inference capabilities and any specific features or configurations it supports.
5. Documentation: Document your model's API, including how to instantiate it, configure it, and run inference. Include any specific requirements or limitations.
6. Examples: Provide example scripts demonstrating how to use your model within the xFT framework. This helps users understand how to integrate the model into their applications.
7. Pull Request: Submit a pull request with your model implementation, tests, documentation, and examples. Ensure all continuous integration checks pass and address any feedback from the xFT maintainers.

### 1.3 New model investigation

To investigate a new Large Language Model (LLM) in the Hugging Face Transformers library, you can follow these steps:

1. Explore Hugging Face Model Hub:
Visit the Hugging Face Model Hub. Use the search bar to find the model by name or explore models by tags, languages, or tasks to discover new LLMs.
2. Read Model Documentation:
Once you've found a model, read its documentation on the model's page. This documentation often includes:
   - Model description and architecture.
   - Training data information.
   - Performance benchmarks.
   - Usage examples and limitations.
3. Check Implementation and Source Code:
For models integrated into the Transformers library, check the source code for implementation details. This can be found in the Transformers GitHub repository.
Look for the model's specific module (e.g., transformers.models.gpt2) to understand how it's implemented.
4. Write Implementation Script for Model Conversion:
xFT currently supports custom model IR format, consisting primarily of two components: config.ini and model weights. For specific custom implementation, refer to `src/xfastertransformer/tools/llama_convert.py`.
   - `config.ini`: This is a specialized model configuration file for xFT. When running a new model with xFT, this configuration file is used to initialize model parameters, analogous to the role of `config.json` in Hugging Face models.
   - Model Weights: The conversion of weight involves primarily renaming the model weight and converting the weight storage format.

## 2. Implementation: Add code in xFT for the new model

### 2.1 Code for Decoder Block

The `CommonDecoder` in this project serves as a base class for implementing various types of decoders for large langurage models, particularly focusing on decoder-based models. It inherits from the AbstractDecoder class and is designed to be highly modular and flexible, allowing for easy extension and customization for different model architectures.

Key characteristics and functionalities of CommonDecoder include:

- Template Parameters  
It uses template parameters to define the attention mechanism, the MLP layer, the data type of the cached keys/values, and a boolean indicating whether attention and MLP are in parallel.
- decoders  
The decoders is a protected member variable that is a vector of pointers to `DECODER` objects. Each `DECODER` object represents a layer within the large language model being implemented by CommonDecoder. The `DECODER` objects is parameterized by `ATTN_CLS` and `MLP_CLS`.
The decoders vector is used to store and manage the sequence of decoder layers in the model. Each layer is responsible for a portion of the computation involved in the model's forward pass, typically involving operations such as `Attention` and `MLP` (feed-forward).
- DecoderContext  
DecoderContext in CommonDecoder encapsulates various parameters and states that are necessary for the operation of the decoder during the model's inference. It includes information such as the batch size, sequence length, accumulated sequence length, hidden size, attention head numbers, and other model-specific parameters that are essential for the decoder's forward pass.
- KV Cache Management  
It includes mechanisms for managing key-value (KV) caches, which are crucial for efficient transformer model implementations, especially in incremental or streaming processing scenarios.

The CommonDecoder inference logic consists 4 stage:

- *Embedding*: the embedding layer processes the input IDs and stores the embeddings in `embBuf`.
- *Attention mask*: most of the LLMs are Autoregressive generative model, which predicts outputs based on past. Typically the Attention Mask is triangular mask.
- *Decoder*: the decoder layers will be invoked, and store the results to `outBuf`.
- *LastLayerNorm*: after all decoder layers finished, the result will pass to a final layernorm.
- *Predictor*: the final layernorm result will be processed by a predictor (typically a linear layer), and store in `finalOut`. If beamSize is more than 1, it will expand the result to make it cover multiple beams.

If you want to add a new model in xFT, you would focus on the Embedding and Decoder implementation of the new model.

#### 2.1.1 Attention

Attention class is the base layer for attention layer in `CommonDecoder`, all related weights and bias are set by the method `setWeights()`.   
Typically the LLM's interface is for PyTorch, thus the weights are already transposed(column-based) in files.  
The datatype of weights is defined by an Attention class's template parameter `WeiT`.  
The dataflow inside Attention looks like the following diagram:

```
  _________                _________                _________                _________                _________                
 |_________|------------->|_________|------------->|_________|------------->|_________|------------->|_________|
              layerNorm                QKV Linear                  MHA                   out Linear             
    input                   imBuffer                qkvMatMul                 imBuffer                  output
```

Some models would have different operations before MHA (multi-head attention), so Attention class has a member called qkpo (Query/Key post operation) parameterized by `QKPO_CLS`. Eg. Llama's `RotaryEmbedding`.
Attention class also has another template parameters `NORM_CLS`. For OPT, the `NORM_CLS` is layerNorm as the above diagram; but for Llama the `NORM_CLS` is `RmsNorm`.

#### 2.1.2 MLP (FFN)

MLP is the second important layer in Decoder. But it is relatively simple comparing to Attention layer.  
Typically it only consists two linears and a layernorm. But there are some different activation after first linear. Currently xFT supports `relu`, `gelu` and `silu`.  
Like Attention class, the MLP class also uses `setWeights()` to load the weights and bias, and uses `forward()` to compute the forward pass.  
MLP class has some template parameters: `WeiT` is the datatype of weights, `InT` and `OutT` is the datatype for MLP input and output. `INPUT_AS_RESID` indicates the input as residental or not, most models use input as residential. But there are exceptions like ChatGLM use values after layernorm as residential.

### 2.2 Code for inference logic before Decoder Block

Before Decoder Block, CommonDecoder prepares the input data(`ids`) and context(`DecoderContext`) for processing.  
Handles prefix sharing if enabled, adjusting input sequence lengths and IDs accordingly.  
Performs embedding operations on the input IDs to generate embeddings.  

### 2.3 Coding for inference logic after Decoder Block

After Decoder Block, typically there is a final layernorm. It supports inplace computing for this input data to save memory footprint. Then pass this result to Predictor.  
The predictor generates the final output, which is then expanded to cover multiple beams if necessary.

## 3. Verification: Debugging Accuracy and Performance

### 3.1 Correctness Checking

- Unit Testing (UT):
For testing individual functionalities, such as new function implementations or new operator implementations, it is recommended to integrate this portion of functionality code into UT testing. The compilation option for UT is `-DXFT_BUILD_TESTS=ON`, for example: `cmake .. -DXFT_BUILD_TESTS=ON.`. UT testing is convenient, avoiding the complexities of debugging in end-to-end long chains.

- Comparing xFT Output with Open Source Versions:
By enabling `DEBUG=true` in cmake, compare the output of each layer with the PyTorch implementation. For end-to-end model testing, consider enabling `-DDEBUG=true` in cmake. After enabling this macro definition, xFT will output the operator results of each layer of the model in the current directory. You can compare the numerical results of each layer's computation on PyTorch. Refer to the following code snippet to implement output hooks in PyTorch:
```python
...
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=dtype,
).eval()

def hook_fn(module, inputs, outputs):
    print(f"\n >>> xFT test module({str(module.xft_module_name)})")
    for _input in inputs:
        print(f"   -- input({_input.shape if type(_input) is torch.Tensor else type(_input)}):{_input}")
    for _output in outputs:
        print(f"   -- output({_output.shape if type(_output) is torch.Tensor else type(_output)}):{_output}")

def hook_modules(module, prefix=""):
    for name, child in module.named_children():
        if isinstance(child, nn.Module):
            module_name = prefix + "." + name + "." + child.__class__.__name__
            child.xft_module_name = module_name
            child.register_forward_hook(hook_fn)
            hook_modules(child, module_name)

hook_modules(model)
...
```

- Utilizing OpenCompass for xFT Dataset Accuracy Testing:
Validation of xFT accuracy can be achieved through OpenCompass. OpenCompass is an LLM evaluation platform that supports a wide range of models (InternLM2, GPT-4, LLaMa2, Qwen, GLM, Claude, etc.) over 100+ datasets. By integrating xFT's patch code, you can easily obtain accuracy test results for multiple datasets and data types using OpenCompass.

### 3.2 Performance Debugging
- XFT_VERBOSE=1:
During runtime, consider enabling the XFT_VERBOSE=1 environment variable to view the percentage of time spent on specific gemm computations. This can roughly identify the current performance bottleneck.

- Enabling Timeline in CMake:
By enabling the option WITH_TIMELINE in cmake, you can enable timeline support with `cmake -DWITH_TIMELINE=ON ...`, This generates the timeline.json file, which can be opened in edge://tracing/ from a web browser to view a more detailed model execution process.