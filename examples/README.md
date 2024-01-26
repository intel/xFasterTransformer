# Examples
xFasterTransformer provides C++, Python(Pytorch) examples to help users learn the API usage. Web demos of some models based on [Gradio](https://www.gradio.app/) are provided. All of the examples and web demo support multi-rank.

## [C++ example](cpp/README.md)
C++ example support automatic identification model and tokenizer which is implemented by [SentencePiece](https://github.com/google/sentencepiece), excluding Opt model which tokenizer is a hard code.

## [Python (PyTorch) example](pytorch/README.md)
Python(PyTorch) example achieves end-to-end inference of the model with streaming output combining the transformer's tokenizer.

## [Web Demo](web_demo/README.md)
A web demo based on [Gradio](https://www.gradio.app/) is provided in repo.  
Support list:
- ChatGLM
- ChatGLM2
- ChatGLM3
- Llama2-chat
- Baichuan2
- Qwen