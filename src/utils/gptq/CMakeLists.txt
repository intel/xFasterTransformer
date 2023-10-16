cmake_minimum_required(VERSION 3.15.1)

project(llm-gptq)

add_executable(run_cholesky run_cholesky.cc utils.h utils.cc quantizer.h quantizer.cc gptq.h gptq.cc)
add_executable(run_weight_quant run_weight_quant.cc utils.h utils.cc quantizer.h quantizer.cc gptq.h gptq.cc)