import requests
import json

data = {"query": ["Once upon a time, there existed a little girl who liked to have adventures."]}


response = requests.post("http://127.0.0.1:8096/xft/predict/", json=data)
print(response.json())

response = requests.post("http://127.0.0.1:8096/xft/stream/", json=data, stream=True)
for line in response.iter_lines(decode_unicode=True,chunk_size=1):
    if line.strip().startswith("data:"):
        print(line.replace("data: ", ""), end="", flush=True)