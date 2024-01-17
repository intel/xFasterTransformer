import asyncio
import time

import grpc
import xft_pb2
import xft_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc

from transformers import AutoTokenizer
import torch

import argparse


def boolean_string(string):
    low_string = string.lower()
    if low_string not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return low_string == "true"


parser = argparse.ArgumentParser()
parser.add_argument("-t", "--token_path", type=str, default="/data/llama-2-7b-chat-hf", help="Path to token file")
parser.add_argument("--port", help="serve port, default 50051.", type=int, default=50051)

args = parser.parse_args()


def health_check_call(stub: health_pb2_grpc.HealthStub):
    start_time = time.time()
    request = health_pb2.HealthCheckRequest(service="xft.Runner")
    while True:
        try:
            resp = stub.Check(request)
            if resp.status == health_pb2.HealthCheckResponse.SERVING:
                return True
        except grpc._channel._InactiveRpcError as e:
            pass

        elapsed_time = time.time() - start_time
        if elapsed_time >= 30:
            print("Health check timed out.")
            return False

        time.sleep(1)


tokenizer = AutoTokenizer.from_pretrained(args.token_path, use_fast=False, padding_side="left", trust_remote_code=True)
if "llama" in args.token_path.lower():
    tokenizer.pad_token_id = tokenizer.eos_token_id

Queries = [
    "Once upon a time, there existed a little girl who liked to have adventures.",
    "Intel Corporation (commonly known as Intel) is an American multinational corporation and technology company.",
]

TokenIds = tokenizer(Queries, return_tensors="pt", padding=True).input_ids

with grpc.insecure_channel(f"localhost:{args.port}") as channel:
    stub = xft_pb2_grpc.XFTServerStub(channel)
    health_stub = health_pb2_grpc.HealthStub(channel)
    if not health_check_call(health_stub):
        print(f"[ERROR] XFT server is not ready on localhost:{args.port}")

    response = stub.predict(
        xft_pb2.QueryIds(Ids=TokenIds.view(-1).tolist(), batch_size=TokenIds.shape[0], seq_len=TokenIds.shape[-1])
    )

    response_ids = torch.Tensor(response.Ids).view(response.batch_size, response.seq_len)
    ret = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    for snt in ret:
        print(snt)
