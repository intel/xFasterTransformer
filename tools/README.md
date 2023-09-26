The tools are used to dump Huggingface models parameters on every layer to binary for C/C++ code on CPU.

Take opt-13b model as an example
```
python opt_convert.py       -i opt-13b/       -o  opt-13b/c-model/ 
```

