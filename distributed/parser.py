import os
import pandas as pd
import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type=str, default="./logs", help="log file path")
parser.add_argument("--model_config_path", type=str, default="../examples/model_config", help="model config file path")
parser.add_argument("--token_in", "-i", type=int, help="Input Token Len")
parser.add_argument("--token_out", "-o", type=int, help="Output Token Len, MaxLen=IN+OUT")
parser.add_argument("--percentile", "-p", type=int, default=90, help="percentile P90/P99")
args = parser.parse_args()

data_types = ["bf16_fp16", "bf16_int8", "bf16_int4", "bf16", "fp16", "int8"]

model_names = [file for file in os.listdir(args.model_config_path)]

sheet = pd.DataFrame(
    columns=[
        "device",
        "socket",
        "instance",
        "model",
        "dtype",
        "num_threads",
        "loop",
        "input_lens",
        "output_lens",
        "bs",
        f"MIN infer_latency(ms)",
        f"Avg infer_latency(ms)",
        f"P{args.percentile} infer_latency(ms)",
        f"MAX infer_latency(ms)",
        f"MIN first_comm(ms)",
        f"Avg first_comm(ms)",
        f"P{args.percentile} first_comm(ms)",
        f"MAX first_comm(ms)",
        f"MIN second_comm(ms)",
        f"Avg second_comm(ms)",
        f"P{args.percentile} second_comm(ms)",
        f"MAX second_comm(ms)",
        f"MIN 1st_token_latency(ms)",
        f"Avg 1st_token_latency(ms)",
        f"P{args.percentile} 1st_token_latency(ms)",
        f"MAX 1st_token_latency(ms)",
        f"MIN 2nd_token_latency(ms)",
        f"Avg 2nd_token_latency(ms)",
        f"P{args.percentile} 2nd_token_latency(ms)",
        f"MAX 2nd_token_latency(ms)",
        f"MIN throughput(token/s)",
        f"Avg throughput(token/s)",
        f"P{args.percentile} throughput(token/s)",
        f"MAX throughput(token/s)",
    ]
)

if not os.path.exists(args.log_path):
    print(f"[Error] The file '{args.log_path}' not exists.")
    sys.exit(1)

LOG_PATH = os.path.abspath(args.log_path)
print("[Info] Parse log files at ", LOG_PATH)


def parse_file_name(file_name):
    row_data = []
    file_name = file_name[len("test_run_") :]
    row_data.append(file_name[0:1])
    file_name = file_name[(1 + len("device_")) :]

    row_data.append(file_name[0:1])
    file_name = file_name[(1 + len("s_")) :]

    row_data.append(file_name[0:1])
    file_name = file_name[(1 + len("ins_")) :]

    model_name = [name for name in model_names if file_name.startswith(name)][0]
    row_data.append(model_name)
    file_name = file_name[len(model_name) + 1 :]

    dtype = [name for name in data_types if file_name.startswith(name)][0]
    row_data.append(dtype)

    row_data += file_name[len(dtype) + 1 : -4].split("_")
    return row_data


def parse_file_content(file_name):
    file_path = os.path.join(LOG_PATH, file_name)
    rtn_map = {
        "MINinferlatency": float(-1),
        "Avginferlatency": float(-1),
        "Pinferlatency": float(-1),
        "MAXinferlatency": float(-1),
        "MINfirst_comm": float(-1),
        "Avgfirst_comm": float(-1),
        "Pfirst_comm": float(-1),
        "MAXfirst_comm": float(-1),
        "MINsecond_comm": float(-1),
        "Avgsecond_comm": float(-1),
        "Psecond_comm": float(-1),
        "MAXsecond_comm": float(-1),
        "MIN1st_token": float(-1),
        "Avg1st_token": float(-1),
        "P1st_token": float(-1),
        "MAX1st_token": float(-1),
        "MIN2nd_token": float(-1),
        "Avg2nd_token": float(-1),
        "P2nd_token": float(-1),
        "MAX2nd_token": float(-1),
        "MINthroughput": float(-1),
        "Avgthroughput": float(-1),
        "Pthroughput": float(-1),
        "MAXthroughput": float(-1),
    }

    first_tokens = []
    second_tokens = []
    inferlatency = []
    first_comm_tokens = []
    second_comm_tokens = []
    comm_times = {}

    with open(file_path, "r") as file:
        for line in file:
            try:
                if "[INFO] First token time" in line:
                    # 以空格分隔并将最后一个数字保存到first_tokens数组
                    tokens = line.split()
                    first_tokens.append(float(tokens[-2]))
                elif "[INFO] Second token time" in line:
                    # 以空格分隔并将最后一个数字保存到second_tokens数组
                    tokens = line.split()
                    second_tokens.append(float(tokens[-2]))
                elif "[INFO] inference latency time" in line:
                    # 以空格分隔并将最后一个数字保存到second_tokens数组
                    tokens = line.split()
                    inferlatency.append(float(tokens[-2]))
                elif "FP32 count " in line:
                    # 以空格分隔并将最后一个数字保存到first_comm_tokens数组
                    tokens = line.split()
                    comm_size = float(tokens[-4])
                    _comm_time_array = comm_times.get(comm_size, [])
                    _comm_time_array.append(float(tokens[-2]))
                    comm_times[comm_size] = _comm_time_array
            except Exception as e:
                print(f"warning: encountered Exception: {e}")

        sorted_keys = sorted(comm_times.keys(), reverse=True)
        if len(sorted_keys) == 2:
            first_comm_tokens = comm_times[sorted_keys[0]]
            second_comm_tokens = comm_times[sorted_keys[1]]

        rtn_map["MINinferlatency"] = np.min(inferlatency[1:]) if len(inferlatency) > 1 else rtn_map["MINinferlatency"]
        rtn_map["Avginferlatency"] = np.mean(inferlatency[1:]) if len(inferlatency) > 1 else rtn_map["Avginferlatency"]
        rtn_map["Pinferlatency"] = (
            np.percentile(inferlatency[1:], args.percentile) if len(inferlatency) > 1 else rtn_map["Pinferlatency"]
        )
        rtn_map["MAXinferlatency"] = np.max(inferlatency[1:]) if len(inferlatency) > 1 else rtn_map["MAXinferlatency"]
        rtn_map["MINfirst_comm"] = np.min(first_comm_tokens) if len(first_comm_tokens) > 1 else rtn_map["MINfirst_comm"]
        rtn_map["Avgfirst_comm"] = (
            np.mean(first_comm_tokens) if len(first_comm_tokens) > 1 else rtn_map["Avgfirst_comm"]
        )
        rtn_map["Pfirst_comm"] = (
            np.percentile(first_comm_tokens, args.percentile) if len(first_comm_tokens) > 1 else rtn_map["Pfirst_comm"]
        )
        rtn_map["MAXfirst_comm"] = np.max(first_comm_tokens) if len(first_comm_tokens) > 1 else rtn_map["MAXfirst_comm"]
        rtn_map["MINsecond_comm"] = (
            np.min(second_comm_tokens) if len(second_comm_tokens) > 1 else rtn_map["MINsecond_comm"]
        )
        rtn_map["Avgsecond_comm"] = (
            np.mean(second_comm_tokens) if len(second_comm_tokens) > 1 else rtn_map["Avgsecond_comm"]
        )
        rtn_map["Psecond_comm"] = (
            np.percentile(second_comm_tokens, args.percentile)
            if len(second_comm_tokens) > 1
            else rtn_map["Psecond_comm"]
        )
        rtn_map["MAXsecond_comm"] = (
            np.max(second_comm_tokens) if len(second_comm_tokens) > 1 else rtn_map["MAXsecond_comm"]
        )
        rtn_map["MIN1st_token"] = np.min(first_tokens[1:]) if len(first_tokens) > 1 else rtn_map["MIN1st_token"]
        rtn_map["Avg1st_token"] = np.mean(first_tokens[1:]) if len(first_tokens) > 1 else rtn_map["Avg1st_token"]
        rtn_map["P1st_token"] = (
            np.percentile(first_tokens[1:], args.percentile) if len(first_tokens) > 1 else rtn_map["P1st_token"]
        )
        rtn_map["MAX1st_token"] = np.max(first_tokens[1:]) if len(first_tokens) > 1 else rtn_map["MAX1st_token"]
        rtn_map["MIN2nd_token"] = np.min(second_tokens) if len(second_tokens) > 1 else rtn_map["MIN2nd_token"]
        rtn_map["Avg2nd_token"] = np.mean(second_tokens) if len(second_tokens) > 1 else rtn_map["Avg2nd_token"]
        rtn_map["P2nd_token"] = (
            np.percentile(second_tokens, args.percentile) if len(second_tokens) > 1 else rtn_map["P2nd_token"]
        )
        rtn_map["MAX2nd_token"] = np.max(second_tokens) if len(second_tokens) > 1 else rtn_map["MAX2nd_token"]
        rtn_map["MINthroughput"] = (
            (args.token_out / rtn_map["MINinferlatency"] * 1000)
            if rtn_map["MINinferlatency"] != -1
            else rtn_map["MINthroughput"]
        )
        rtn_map["Avgthroughput"] = (
            (args.token_out / rtn_map["Avginferlatency"] * 1000)
            if rtn_map["Avginferlatency"] != -1
            else rtn_map["Avgthroughput"]
        )
        rtn_map["Pthroughput"] = (
            (args.token_out / rtn_map["Pinferlatency"] * 1000)
            if rtn_map["Pinferlatency"] != -1
            else rtn_map["Pthroughput"]
        )
        rtn_map["MAXthroughput"] = (
            (args.token_out / rtn_map["MAXinferlatency"] * 1000)
            if rtn_map["MAXinferlatency"] != -1
            else rtn_map["MAXthroughput"]
        )
    return [*rtn_map.values()]


def process_log_files():
    log_files = [file for file in os.listdir(LOG_PATH) if file.endswith(".log") and file.startswith("test_run")]
    for index, file_name in enumerate(log_files):
        print(f"parse file -- {file_name}")
        name_params = parse_file_name(file_name)
        # file_params = parse_file_content(os.path.join(LOG_PATH, file_name))
        sheet.loc[index] = parse_file_name(file_name) + parse_file_content(file_name)

    print(sheet)
    sheet.to_excel(os.path.join(LOG_PATH, f"xft_perfs_data_{os.path.basename(LOG_PATH)}.xlsx"), index=0)


if __name__ == "__main__":
    process_log_files()
