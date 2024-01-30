import requests
import json
import uuid
from mlserver.types import InferenceResponse
from mlserver.codecs.string import StringRequestCodec
from pprint import PrettyPrinter
import traceback

pp = PrettyPrinter(indent=1)

PAYLOAD_TEMPLATE = "Hello! How are you today?"
request_id = uuid.uuid4()
inputs_string = json.dumps({"prompt": PAYLOAD_TEMPLATE, "request_id": str(request_id)})

inference_request = {
    "inputs": [
        {
            "name": "xft-llama-2-model",
            "shape": [len(inputs_string)],
            "datatype": "BYTES",
            "data": [inputs_string],
        }
    ]
}
endpoint = "http://127.0.0.1:8080/v2/models/xft-llama-2-model/streams"
response = requests.post(endpoint, json=inference_request, stream=True)

for line in response.iter_lines(decode_unicode=True):
    if line.decode('utf-8').strip().startswith("data:"):
        try:
            out = line.decode('utf-8').strip().split("data:", 1)[1]
            print(out, end="", flush=True)
        except Exception as e:
            print("[error] run atom_detect failed!: {}".format(traceback.format_exc()))
            pass