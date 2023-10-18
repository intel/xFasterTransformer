import os
import shutil
import argparse

import struct
import configparser
import torch

from gptq import *

def get_model_configs(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    first_section = config.sections()[0]
    model_configs = {}
    model_configs['head_num'] = int(config[first_section]['head_num'])
    model_configs['size_per_head'] = int(config[first_section]['size_per_head'])
    model_configs['inter_size'] = int(config[first_section]['inter_size'])
    model_configs['num_layer'] = int(config[first_section]['num_layer'])
    model_configs['weight_data_type'] = config[first_section]['weight_data_type']
    return model_configs

def read_bin(bin_file_path, rows, columns):
    weight = []
    bin_f = open(bin_file_path, "rb")
    for h in range(rows):
        weight.append([])
        for w in range(columns):
            binary_value = bin_f.read(4)
            fp32_value = struct.unpack("f", binary_value)[0]
            weight[-1].append(fp32_value)
    weight = torch.tensor(weight)
    return weight

def write_bin(bin_file_path, weight, wbits):
    if os.path.exists(bin_file_path):
        os.remove(bin_file_path)
    bin_f = open(bin_file_path, "ab+")
    weight = weight.flatten()
    if wbits == 8:
        for i in range(weight.shape[0]):
            int8_value = int(weight[i])
            binary_value = int8_value.to_bytes(1, 'big')
            bin_f.write(binary_value)
    elif wbits == 4:
        pos = 0
        int8_value = 0
        for i in range(weight.shape[0]):
            if pos == 0:
                int8_value = int(weight[i])
                pos = 1
            else:
                int8_value += 16 * int(weight[i])
                binary_value = int8_value.to_bytes(1, 'big')
                bin_f.write(binary_value)
                pos = 0
        if pos == 1:
            binary_value = int8_value.to_bytes(1, 'big')

            bin_f.write(binary_value)

    bin_f.close()

def quantize_weight(input_model_path, output_model_path, bin_file, rows, columns, wbits, sym):
    print("Start reading " + os.path.join(input_model_path, bin_file))
    weight = read_bin(os.path.join(input_model_path, bin_file), rows, columns)
    llm_gptq = LLM_GPTQ(weight, wbits, sym)
    quantized_weight, scale, zero = llm_gptq.fasterquant()

    output_bin_prefix = bin_file[:-5]
    write_bin(os.path.join(output_model_path, output_bin_prefix + "quantized.0.bin"), quantized_weight, wbits)
    write_bin(os.path.join(output_model_path, output_bin_prefix + "scale.0.bin"), scale, wbits)
    write_bin(os.path.join(output_model_path, output_bin_prefix + "zero.0.bin"), zero, wbits)
    print("Finish quantization for " + os.path.join(input_model_path, bin_file))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_type', type=str, default="llama", 
        choices=["llama", "llama2", "chatglm", "chatglm2"]
    )
    parser.add_argument(
        '--input_model_path', type=str, default="/data/llama-13b-cpu"
    )
    parser.add_argument(
        '--output_model_path', type=str, default="/data/llama-quantized-13b-cpu"
    )
    parser.add_argument(
        '--wbits', type=int, default=8, choices=[4, 8]
    )
    args = parser.parse_args()

    # TODO: read from config.ini in xFT
    model_type = args.model_type
    input_model_path = args.input_model_path
    output_model_path = args.output_model_path
    model_configs = get_model_configs(os.path.join(input_model_path, "config.ini"))
    wbits = args.wbits # Support 4 bits or 8 bits
    sym = False

    attHeadNum = model_configs["head_num"]
    kvHeadNum = model_configs["head_num"]
    size_per_head = model_configs["size_per_head"]
    imSize = model_configs["inter_size"]
    layers = model_configs["num_layer"]
    hiddenSize = attHeadNum * size_per_head
    qSize = hiddenSize
    attHeadSize = int(hiddenSize / attHeadNum)
    kvSize = attHeadSize * kvHeadNum
    qkvSize = qSize + kvSize + kvSize

    attention_qkv_weight_rows = hiddenSize
    attention_qkv_weight_columns = qkvSize
    attention_dense_weight_rows = hiddenSize
    attention_dense_weight_columns = hiddenSize
    # For llama and llama2
    mlp_down_weight_rows = imSize
    mlp_down_weight_columns = hiddenSize
    mlp_up_weight_rows = hiddenSize
    mlp_up_weight_columns = imSize
    mlp_gate_weight_rows = hiddenSize
    mlp_gate_weight_columns = imSize
    # For chatglm and chatglm2
    mlp_dense_h_to_4h_rows = hiddenSize
    mlp_dense_h_to_4h_columns = imSize
    mlp_dense_4h_to_h_rows = imSize
    mlp_dense_4h_to_h_columns = hiddenSize

    prefix = "model.layers"
    suffix = "0.bin" 
    for layer_index in range(layers):
        quantize_weight(input_model_path, output_model_path, "model.layers.{}.attention.query_key_value.weight.0.bin".format(layer_index), attention_qkv_weight_rows, attention_qkv_weight_columns, wbits, sym)
        quantize_weight(input_model_path, output_model_path, "model.layers.{}.attention.dense.weight.0.bin".format(layer_index), attention_dense_weight_rows, attention_dense_weight_columns, wbits, sym)
        if "llama" in model_type: 
            quantize_weight(input_model_path, output_model_path, "model.layers.{}.mlp.down_proj.weight.0.bin".format(layer_index), mlp_down_weight_rows, mlp_down_weight_columns, wbits, sym)
            quantize_weight(input_model_path, output_model_path, "model.layers.{}.mlp.up_proj.weight.0.bin".format(layer_index), mlp_up_weight_rows, mlp_up_weight_columns, wbits, sym)
            quantize_weight(input_model_path, output_model_path, "model.layers.{}.mlp.gate_proj.weight.0.bin".format(layer_index), mlp_gate_weight_rows, mlp_gate_weight_columns, wbits, sym)
        elif "chatglm" in model_type:
            quantize_weight(input_model_path, output_model_path, "model.layers.{}.mlp.dense_h_to_4h.weight.0.bin".format(layer_index), mlp_dense_h_to_4h_rows, mlp_dense_h_to_4h_columns, wbits, sym)
            quantize_weight(input_model_path, output_model_path, "model.layers.{}.mlp.dense_4h_to_h.weight.0.bin".format(layer_index), mlp_dense_4h_to_h_rows, mlp_dense_4h_to_h_columns, wbits, sym)

    quantized_bin_files = ["attention.query_key_value.weight", "attention.dense.weight", "mlp.down_proj.weight", "mlp.up_proj.weight", "mlp.gate_proj.weight", "mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight"]
    for bin_file in os.listdir(input_model_path):
        quantized = False
        for quantized_bin_file in quantized_bin_files:
            if quantized_bin_file in bin_file:
                quantized = True
                break
        if quantized:
            quantized = False
            continue
        else:
            shutil.copyfile(os.path.join(input_model_path, bin_file), os.path.join(output_model_path, bin_file))
