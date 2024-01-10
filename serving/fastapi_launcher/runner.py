from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
import uvicorn
import xfastertransformer
from typing import List
from transformers import AutoTokenizer
from pydantic import BaseModel
import os

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


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

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/chatglm-6b-hf", help="Path to token file")
parser.add_argument("-m", "--model_path", type=str, default="/data/chatglm-6b-cpu", help="Path to model file")
parser.add_argument("-d", "--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--chat", help="Enable chat mode, Default to False.", type=boolean_string, default=False)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.", type=boolean_string, default=False)
parser.add_argument("--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument("--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument("--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)
parser.add_argument("--rep_penalty", help="param for repetition penalty. 1.0 means no penalty", type=float, default=1.0)
parser.add_argument("--port", help="serve port, default 18996.", type=int, default=8096)

args = parser.parse_args()

model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)

if model.rank != 0:
    while True:
        model.generate()

tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=False, padding_side="left", trust_remote_code=True)
if "llama" in args.token_path.lower():
    tokenizer.pad_token_id = tokenizer.eos_token_id


class InputType(BaseModel):
    query: List[str]


class ResponseType(BaseModel):
    response: List[str]


app = FastAPI()


@app.post("/xft/predict/")
async def predict(data: InputType) -> ResponseType:
    input_ids = tokenizer(data.query, return_tensors="pt", padding=True).input_ids
    generated_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[-1] + args.output_len,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
    )
    ret = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {"response": ret}


@app.post("/xft/stream/")
async def predict(data: InputType):
    input_ids = tokenizer(data.query, return_tensors="pt", padding=True).input_ids

    model.config(
        input_ids.shape[-1] + args.output_len,
        num_beams=args.num_beams,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.rep_penalty,
    )
    model.input(input_ids)

    async def async_generator():
        while not model.is_done():
            next_token_id = model.forward()
            next_token_id = next_token_id.view(-1).tolist()[0]
            yield tokenizer.decode(next_token_id, skip_special_tokens=True)

    return EventSourceResponse(async_generator())


uvicorn.run(app, host="127.0.0.1", port=args.port)
