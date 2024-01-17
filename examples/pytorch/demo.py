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
from typing import Tuple, List

# Ignore Tensor-RT warning from huggingface
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import torch
import time
from transformers import AutoTokenizer, TextStreamer

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
parser.add_argument("--padding", help="Enable padding, Default to False.", type=boolean_string, default=False)
parser.add_argument("--streaming", help="Streaming output, Default to True.", type=boolean_string, default=True)
parser.add_argument("--num_beams", help="Num of beams, default to 1 which is greedy search.", type=int, default=1)
parser.add_argument("--output_len", help="max tokens can generate excluded input.", type=int, default=100)
parser.add_argument("--chat", help="Enable chat mode, Default to False.", type=boolean_string, default=False)
parser.add_argument("--do_sample", help="Enable sampling search, Default to False.", type=boolean_string, default=False)
parser.add_argument("--temperature", help="value used to modulate next token probabilities.", type=float, default=1.0)
parser.add_argument("--top_p", help="retain minimal tokens above topP threshold.", type=float, default=1.0)
parser.add_argument("--top_k", help="num of highest probability tokens to keep for generation", type=int, default=50)


def build_inputs_chatglm(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
    prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
    inputs = tokenizer(prompt, return_tensors="pt", padding=padding).input_ids
    return inputs


def build_inputs_baichuan(tokenizer, query: str, padding, history: List[Tuple[str, str]] = []):
    inputs = tokenizer(query, return_tensors="pt", padding=padding).input_ids
    suffix = torch.tensor([[196]])
    prefix = torch.tensor([[195]])
    inputs = torch.cat((prefix, inputs, suffix), dim=1)
    return inputs

def build_inputs_qwen(tokenizer, query: str, padding, history: List[Tuple[str, str]] = [],
                           system: str = "You are a helpful assistant.",
                           max_window_size: int = 6144, chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return torch.tensor([context_tokens])


import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("../../src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer

DEFAULT_PROMPT = "Once upon a time, there existed a little girl who liked to have adventures."

# DEFAULT_PROMPT = """你是一个文档问答专家。我将提供文档内容和问题，你将提供答案。

# 文档内容是：
# “”“
# 同时，在每个小车的平移两端都装 有减速限位、停止限位开关，防止小车撞到两边 的梁上而发生故障。\n\n当小车碰到减速限位时，无 论现在手柄在几挡，小车只能以10%速度运行 当碰到停止限位则停机。\n\n图2小车机构控制流程图 小车机构设有K01急停回路和K02急停回路， 当 K01急停回路发生故障时，整个系统跳闸，所 有机构均停止动作，如果想再次开机，则需要 3mini 之后，这是为了等待 AFE 充分放电，保证安 全；当K02急停回路中发生故障时，则只有小车 机构跳闸，其他机构可正常工作。\n\n2.3.\n\n大车机构的设计大车机构操作手柄共有4挡，其控制程序流 程图如图3 所示。\n\n在正常情况下，速度的设定分 别为10%、20%、50%和 80%。\n\n同时，在大车的 后端分别装有减速限位、停止限位和极限限位开 关，在大车的前端分别装有防撞减速限位和防 停止限位，以防止2台车发生相撞。\n\n当大车碰到 减速限位时，无论当前手柄在儿挡，大车只能以 10%速度运行，当碰到停止限位则停机，如果碰 到停止限位还没有停下来则会碰到极限限位，此 时机构紧急停车，整个系统跳闸停机。
# 司机通 过大车主令手柄操作框架大车时，大车海侧和陆侧逆变器的初始 速度给定是相等的，在绝对值编码器检测的框架大车海陆侧位置 偏差达到30mm时，自动纠偏功能激活，偏差值被送到PLC的PID 控制器中，经计算后输出速度附加给定值，速度主给定和附加给 定共同作用在海侧逆变器的速度控制器上，对海侧和陆侧电机进 行相对速度调整，实现框架大车的同步控制。\n\n当起重机海陆侧运 行行程偏差达到59.5mm～119mm时，框架大车降速纠偏，并发出 相应的警示，司机室框架大车同步报警指示灯闪亮；当起重机海 陆侧运行行程偏差大于119mm时，框架大车自动停车，并在司机 室发出报警信号，司机室框架大车同步报警指示灯常亮。\n\n偏差发 生后，只能通过本地操作箱手动操作海侧或陆侧运行机构进行手 动纠偏，消除偏差并查处误差原因后恢复框架大车运行。
# 框架大车机构海陆侧各有11台22kW电机，500V时电流为 37A，过载系数为1.5，根据机械提出的额定总静负载功率计 算总静负载电流（100%负载）：I=37×11=407A，允许过载 时的负载电流，按照总静负载电流的1.5倍考虑：Ig=407×1.5 98 博览AUTOMATION PAI 框架式起重机大车同步及纠偏控制框架式起重机的框架大车海陆侧是刚性连接在一起的，但 由于跨距为59.5m，在实际使用过程中受到各种因素的影响, 如海陆侧运行阻力不同，机械制造时走轮直径偏差，电磁干扰 等。\n\n这些因素会造成海陆侧腿运行中速度快慢不一，造成位置 偏差，偏差过大时甚至会扭坏框架式起重机的机械结构。\n\n这种 情况下，框架式大车海陆侧运行同步及安全的偏差检测控制显 得尤为重要。\n\n同步控制原理如图4所示： 图4框架大车同步控制 海侧1#电机和陆侧1#电机安装有增量式脉冲编码器实时检测 海陆侧的行走速度，通过编码器模块SMC30连接到各自的逆变器 做速度闭环控制。\n\n如果同侧电机实际反馈速度与给定速度偏差过 大或海陆侧电机实际反馈速度偏差过大，需要停止大车运行进行 保护，然后检查故障原因。
# 所有起升机构的高度参考点都是一样的，一般以地面水平面 作为参考0m位置，通过高精度激光测距仪对起升绝对值编码器位 置进行校准，三个小车机构的水平0m位置参考点也是一致的，然 后通过高精度激光测距仪对小车绝对值编码器位置进行校准，这 样起升和小车机构均有了标准的参考坐标，便于用户实时查看相 应机构位置信息，而且在此基础上可以进行多机构同步控制。\n\n速 度和位置同步原理如图6所示： 图6多机构同步控制图6多机构同步控制 选择一个机构作为主轴，其他从轴机构的速度给定跟随主 轴，同步控制激活时分别记录各从轴与主轴的初始相对位置偏 差作为控制基准，同步运行时实时检测各从轴与主轴的实际相 对位置，如果实际相对位置超过初始相对位置100mm，则激活 相对位置控制器，偏差值作为位置控制器调节输入量去调节从 轴速度直到主从轴的实际相对位置回到允许范围内。\n\n为了保证 司步运行的安全性，任意轴的软硬件保护及安全回路均同时对 同步运行中的其他轴起作用，而且轴与轴之间也有必要的同步 速度和位置偏差检测连锁，最大程度上保证了同步运行的稳定 可靠。
# 表1各主机构电机参数 表1各主机构电机参数          机构 极数和功率 转速 数量 电压 主起升I: 308kW8极 741/1725r/min 1台 500V 主起升II: 308kW8极741/1725r/min 1台 500V 1#小车: 11kW4极1440r/min 4台 500V 2#小车： 11kW4极1440r/min 4台 500V 2x300t大车海侧 15kW4极1460r/min 4台 500V 2x300t大车陆侧 15kW4极1460/min 4台 500V 主起升IⅢI 1#: 308kW8极741/1725r/min 1台 500V 主起升IⅢI 2#: 308kW8极741/1725r/min 1台 500V 副起升: 45kW8极741/1725r/min 1台 500V 3#小车： 11kW4极1440r/min 8台 500V 框架大车海侧: 22kW4极1465r/min 11台 500V 框架大车陆侧: 22kW4极1465r/min 11台 500V           面分别以起开、小车、大车机构为例，校验计算驱动容量。
# 大车机构设有 K01 和K02急停回路，当 K01 急停回路发生故障时，整个系统跳闸，所有机构 均停止动作，如果想再次开机，则同样需要3min 之后；当K02 急停回路中有故障发生时，则只有 大车机构跳闸，其他机构可正常工作 图3大车机构控制流程图 结束语利用 PLC控制的变频调速技术，可以集中控 制各机构的运行状态，铸造起重机系统的各挡速 度、加速时间和制动减速时间可根据现场情况由 变频器设置，调整方便。\n\n负载变化时，各挡速度 基本不变，调速性能好。\n\n通过现场应用，该控制 系统应用准确、可靠、良好。\n\n参考文献[1」马兵.\n\nPLC在铸造起重机安全制动器上的应用[J]，起 重运输机械，2006(6)：33-35.\n\n[2]\n\n徐丽娟.PLC 控制的变频调速在桥式起重机拖动系统 中的应用［OL］．\n\n中国工控网.\n\n[3]韩泓：铸造起重机电气系统介绍[J]．\n\n电气传动自动 化,2003, 25 (6):52 -54.\n\n[4］\n\n王慕文，王晓瑜，盘式制动器的性能分析及应用[J], 治金设备，2003，2(1)：56-58.
# 为了防止增量型脉冲编码器与绝对值编码器的测量误差累 积，我们在大车轨道侧间隔固定距离安装基准磁块对编码器校 准，以提高测量精度，根据现场起重机轨道情况，此间隔距离定 为35m，基准磁块排布如图5所示： 起升同步和主从控制，小车同步及防撞框架式起重机有三台小车，1#小车和2#小车位于移动梁，额 定载荷300吨，机械房内起升机构卷筒各自由一台315kW电机拖 动；3#小车位于固定梁，额定载荷600吨，其中起升机构卷筒由 两台315kW起升电机拖动，两台电机要做主从控制以合理分配力 矩。\n\n框架式起重机吊载较大吨位分段时需要三个起升吊钩联动， 而且三个小车机构也要同步运行，移动梁上的两个小车之间还需 要考虑防撞保护。\n\n（1）起升或小车的同步控制需要综合考虑以下内容：建立 -个标准的参考坐标系；速度和位置同步；各机构之间的安全联 锁。
# 系统启动时， 卷筒的液压制动器首先开闸，起升时电机上的制 动器松闸，同时启动起升电机；起升机构按照事 先设定的速度根据手柄的挡位开始运行；起升机 构停止起吊重物时先切断起升电机，随即电机的 制动器全部制动抱闸，此时起升机构完成了1个 工作循环。\n\n系统停机时，卷筒的液压制动器卸压 上闸，因此在系统正常工作状态中，无论是超速开天或是时间继电器都不起作用，都不会指令液 压制动器闭合上闸[3,]。\n\n当传动轴系统发生破坏性故障，如断轴时， 钢水包迅速下坠，卷筒超速旋转，超速开关（装 在起升电机后面）\n\n在超速状态下关断电磁阀使液 压制动器卸压上闸，卷筒停止旋转。\n\n此时切断驱 动电机电源，起升电机全部制动器上闸，吊钩或起 吊的重物停止下坠，避免因坠包而酿成事故。\n\n当起吊重物超载时，起升机构不再继续上升， 只能下降，司机室报警器报警，直至超载消除 小车机构的设计小车机构共有主小车机构和副小车机构，用 于实现主起升和副起升水平方向的移动，其操作 手柄共有4 挡，其控制程序流程图如图2 所示。\n\n在 正常情况下，速度的设定分别为10%、20%、 50%和60%。为保证起重机正常、安全地工作，框架大车运行系统采用 两套相互独立的编码器双重保护纠偏装置。\n\n海侧和陆侧各自通过 一个检测轮安装绝对值编码器实时检测海陆侧的位置。\n\n如果只依 靠绝对值编码器检测位置，当检测轮在轨道面打滑或堵转时会造 成位置检测不准确。\n\n我们将编码器模块SMC30采集的增量式脉 冲编码器信号传输到逆变器控制单元CU320-2\n\nDP，根据大车减速 箱变比和大车轮径计算得到大车的实际位置，这个位置用于和绝 对值编码器检测的位置进行比较。\n\n如果同侧两个编码器位置偏差 过大，说明其中一路检测肯定出现问题，这时候需要停止大车运行，检查排除故障；如果海陆侧编码器位置偏差过大，同样需要 进行停车保护。\n\n由于框架大车海陆侧的刚性连接，同步纠偏时的控制量要保 持比较小的范围进行微动纠偏，以免造成机械结构损伤。（2）600t起升卷筒是由两台315kW电机拖动，机械同轴刚性 连接。\n\n为了实现同轴相连的两台电机在运行过程中保持负荷的均 匀分配，两台逆变器需要采用主-从控制。\n\n主-从控制可分为两种 方式：速度跟随，转矩限幅；速度控制，转矩跟随。\n\n第一种方式 图5大车基准磁块排布 在起重机和地面建立以35m为刻度的绝对坐标系，校准的原 理是让两个坐标系重合。\n\n200m为轨道中间位置，向左磁块计数个 数依次递增，向右磁块计数个数依次递减，当大车行走到磁铁同 步区域时，磁感应同步限位触发，将大车海侧和陆侧的编码器位 置分别记录；到下一次磁铁同步区域感应前海侧和陆侧行走的相 对位移作为纠偏PID控制器的输人量，去调节框架大车海陆侧电 机的速度。\n\n如果在磁铁同步区域中磁感应限位未检测到，则大车 减速运行，直到下一个磁铁同步区域；如果连续2个同步区域均未 检测到磁块，则大车停止运行。\n\n实践证明，这种基准磁块校准方 法可以有效地将编码器累计误差控制在一个检测区间以内，提高 了同步纠偏控制的精确度和稳定性。
# “”“

# 我的问题是：
# “”“
# 大车行走速度是多少
# “”“
# """


if __name__ == "__main__":
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.token_path, use_fast=False, padding_side="left", trust_remote_code=True
    )

    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)
    streamer = None
    if model.rank == 0 and args.streaming and args.num_beams == 1:
        streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=False)

    if model.rank == 0:
        # Master
        while True:
            input_prompt = input("\nPlease enter the prompt: ")
            if input_prompt == "":
                input_prompt = DEFAULT_PROMPT
                print("[Use default prompt]:" + input_prompt)

            if args.chat and "chatglm" in args.model_path.lower():
                input_ids = build_inputs_chatglm(tokenizer, input_prompt, args.padding)
            elif "baichuan" in args.model_path.lower():
                input_ids = build_inputs_baichuan(tokenizer, input_prompt, args.padding)
            elif "qwen" in args.model_path.lower() and "chat" in args.model_path.lower():
                input_ids = build_inputs_qwen(tokenizer, input_prompt, args.padding)
            else:
                input_ids = tokenizer(input_prompt, return_tensors="pt", padding=args.padding).input_ids
            print("=" * 50)

            start_time = time.perf_counter()
            generated_ids = model.generate(
                input_ids,
                max_length=input_ids.shape[-1] + args.output_len,
                streamer=streamer,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=151643,
                pad_token_id=151643,
            )
            end_time = time.perf_counter()

            if streamer is None:
                ret = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                for snt in ret:
                    print(snt)
            print("=" * 20 + "Performance" + "=" * 20)
            execution_time = end_time - start_time
            print(f"Execution time:\t{execution_time:.2f} s")
            input_token_nums = torch.numel(input_ids)
            output_token_nums = torch.numel(generated_ids) - input_token_nums
            latency = execution_time * 1000 / output_token_nums
            througput = output_token_nums / execution_time
            print(f"Latency:\t{latency:.2f} ms/token")
            print(f"Througput:\t{througput:.2f} tokens/s")
    else:
        # Slave
        while True:
            model.generate()
