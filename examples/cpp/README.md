# Build
```
    # cd into root directory
    # please make sure torch is installed
    mkdir build
    cd build && cmake ..
    make
```

# Run
## C++ interface examples(supporting automatic identification model)
```
    ./example ${DATASET} ${DATATYPE}
```
- DATASET: support Opt, Llama and chatGLM now.
- DATATYPE: support int8 and fp16 now, default and `0` for fp16, `1` for int8. 

Notes: Please manually modify input token ids to match the requirements of the model which may cause error!!!

Default use Opt's token list to decode output.