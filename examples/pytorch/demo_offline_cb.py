# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import sys
from typing import Tuple, List

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer

import argparse

all_results = {}
args = None

DTYPE_LIST = [
    "fp16",
    "bf16",
    "int8",
    "w8a8",
    "int4",
    "nf4",
    "bf16_fp16",
    "bf16_int8",
    "bf16_w8a8",
    "bf16_int4",
    "bf16_nf4",
    "w8a8_int8",
    "w8a8_int4",
    "w8a8_nf4",
]

KVCACHE_DTYPE_LIST = ["fp16", "int8"]

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/llama-2-7b-chat-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/llama-2-7b-chat-cpu", help="Path to model file")
parser.add_argument("-i", "--input_file", type=str, help="Path to input file, if not provided, use builtin prompts.")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="bf16", help="Data type")
parser.add_argument("--kv_cache_dtype", type=str, choices=KVCACHE_DTYPE_LIST, default="fp16", help="KV cache dtype")
parser.add_argument("-s", "--sort_results", action="store_true", help="Sort the results")

sys.path.append("../../src")
import xfastertransformer

class FileReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = open(filename, 'r')

    def getNextInput(self):
        line = self.file.readline().strip()
        if line:
            return line
        else:
            self.file.close()
            return None
        
class TextReader:
    def __init__(self):
        self.prompts = [
            "What is the meaning of life?",
            "Tell me a joke.",
            "How do you spell supercalifragilisticexpialidocious?",
            "What is the capital of France?",
            "Can you recommend a good book?",
            "What is the square root of 144?",
            "How do you say 'hello' in Spanish?",
            "What is the largest planet in our solar system?",
            "What is the color of the sky?",
            "What is the distance between Earth and the Moon?",
            "Can you play the piano?",
            "What is the temperature today?",
            "How many continents are there?",
            "What is the national animal of Australia?",
            "What is the speed of light?",
            "What is the chemical symbol for gold?",
            "Who painted the Mona Lisa?",
            "What is the largest ocean in the world?",
            "What is the capital of Japan?",
            "Can you swim?",
            "What is the time in New York right now?",
            "How many legs does a spider have?",
            "What is the largest country in the world?",
            "What is the freezing point of water?",
            "Who wrote the play 'Romeo and Juliet'?",
            "What is the currency of Germany?",
            "Can you speak French?",
            "What is the population of India?",
            "What is the boiling point of water?",
            "Who discovered gravity?",
            "What is the national flower of England?",
            "What is the diameter of the Earth?",
            "Can you dance?",
            "What is the time in London right now?",
            "How many players are there in a basketball team?",
            "What is the tallest mountain in the world?",
            "What is the capital of Brazil?",
            "Can you cook?",
            "What is the population of China?",
            "What is the melting point of ice?",
            "Who wrote the novel 'Pride and Prejudice'?",
            "What is the currency of Japan?",
            "Can you play chess?",
            "What is the population of the United States?",
            "What is the height of the Eiffel Tower?",
            "Can you sing?",
            "What is the time in Tokyo right now?",
            "How many players are there in a soccer team?",
            "What is the deepest ocean in the world?",
            "What is the capital of Australia?",
            "Can you paint?",
            "What is the population of Russia?",
            "What is the boiling point of nitrogen?",
            "Who invented the telephone?",
            "What is the national animal of Canada?",
            "What is the circumference of the Earth?",
            "Can you code?",
            "What is the time in Paris right now?",
            "How many players are there in a baseball team?",
            "What is the highest mountain in Africa?",
            "What is the capital of China?",
            "Can you write?",
            "What is the population of Brazil?",
            "What is the melting point of gold?",
            "Who wrote the novel 'To Kill a Mockingbird'?",
            "What is the currency of France?",
            "Can you play guitar?",
            "What is the population of Japan?",
            "What is the boiling point of mercury?",
            "Who discovered penicillin?",
            "What is the national animal of India?",
            "What is the radius of the Earth?",
            "What is the time in Berlin right now?",
            "How many players are there in a hockey team?",
            "What is the highest mountain in South America?",
            "What is the capital of Germany?",
            "Can you draw?",
            "What is the population of the United Kingdom?",
            "What is the melting point of iron?",
            "Who wrote the novel '1984'?",
            "What is the currency of Italy?",
            "What is the population of France?",
            "What is the boiling point of water?",
            "Who discovered electricity?",
            "What is the national animal of China?",
            "What is the diameter of the Moon?",
            "What is the time in Sydney right now?",
            "How many players are there in a cricket team?",
            "What is the highest mountain in Europe?",
            "What is the population of Germany?",
            "What is the melting point of copper?",
            "Who wrote the novel 'The Great Gatsby'?",
            "What is the currency of Spain?",
            "What is the population of Italy?",
            "Who discovered the theory of relativity?",
            "What is the national animal of Russia?",
            "What is the circumference of the Moon?",
            "What is the time in Moscow right now?",
            "How many players are there in a rugby team?",
            "What is the highest mountain in North America?",
            "What is the capital of Russia?",
            "Can you write?",
            "What is the population of Italy?",
            "What is the melting point of silver?",
        ]
        self.index = 0
    
    def getNextInput(self):
        if self.index < len(self.prompts):
            self.index += 1
            return self.prompts[self.index - 1]
        else:
            return None

def outputReady(id, prompt, generated):
    if args.sort_results:
        # Record the result and sort them when finished
        all_results[id.item()] = (prompt, generated)
    else:
        # Directly print the result
        print('Prompt: ', prompt)
        print('Generated: ', generated)
        print('\n')

def genNextToken(model, active_seqs, max_seqlen, end_ids, output_func):
    seq_ids = []
    input_ids = []
    
    for seq_id, (_, _, gen_ids) in active_seqs.items():
        seq_ids.append(seq_id)
        input_ids.append(gen_ids[-1])

    infer_seq_ids = torch.stack(seq_ids)
    input_ids = torch.stack(input_ids).unsqueeze(1)

    # Generate next token
    model.set_input_cb(input_ids, infer_seq_ids, max_seqlen)
    ret_tensor = model.forward_cb()
    _, max_index = torch.max(ret_tensor, dim=-1)
    next_ids = max_index
    
    # Check if some sequences finished
    idx = 0
    for seq_id in seq_ids:
        prompt, input_ids, gen_ids = active_seqs[seq_id]

        # Check if encountered end id
        if next_ids[idx] in end_ids:
            output_func(seq_id, prompt, tokenizer.batch_decode([gen_ids], skip_special_tokens=True))
            model.free_seqs(torch.stack([seq_id]))
            del active_seqs[seq_id]
        
        else:
            # Extend generated IDs
            gen_ids = torch.cat((gen_ids, next_ids[idx].unsqueeze(0)))
            active_seqs[seq_id] = (prompt, input_ids, gen_ids)

            # Check if the sequence reaches max length
            if len(input_ids[0]) + len(gen_ids) >= max_seqlen - 1:
                output_func(seq_id, prompt, tokenizer.batch_decode([gen_ids], skip_special_tokens=True))
                model.free_seqs(torch.stack([seq_id]))
                del active_seqs[seq_id]
        
        idx += 1

def generate(model, input_func, output_func):
    # Modify below parameters according to the model
    end_ids = [2] # LLama2 model
    max_seqlen = 256
    active_seqs = {}

    while True:
        prompt = input_func()
        if prompt is None:
            break

        # Tokenize the input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # First token generation for the new input
        seq_id = model.set_input_cb(input_ids, None, max_seqlen)
        ret_tensor = model.forward_cb()

        _, max_index = torch.max(ret_tensor, dim=-1)
        gen_ids = max_index.view(-1)
        active_seqs[seq_id[0]] = (prompt, input_ids, gen_ids)

        # Batching together for next tokens
        while len(active_seqs) >= 64:
            genNextToken(model, active_seqs, max_seqlen, end_ids, output_func)
        
    # Finish all the active sequences
    while len(active_seqs) > 0:
        genNextToken(model, active_seqs, max_seqlen, end_ids, output_func)

if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
    )

    model = xfastertransformer.AutoModel.from_pretrained(
        args.model_path, dtype=args.dtype, kv_cache_dtype=args.kv_cache_dtype
    )

    start_time = time.perf_counter()

    # Call generate to generate tokens for all prompts
    input_func = None
    if args.input_file is None:
        input_func = TextReader().getNextInput
    else:
        input_func = FileReader(args.input_file).getNextInput
    generate(model, input_func, outputReady)

    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Execution time:\t{execution_time:.2f} s")

    # Sort and print all values in all_results
    if args.sort_results:
        keys = list(all_results.keys())
        keys.sort()
        sorted_result = {i: all_results[i] for i in keys}
        for (prompt, generated) in sorted_result.values():
            print('Prompt: ', prompt)
            print('Generated: ', generated)
            print('\n')
