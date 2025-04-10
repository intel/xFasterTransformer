# Copyright (c) 2023-2024 Intel Corporation
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

from .llama_convert import LlamaConvert
from .llama_convert import LlamaConvert as DeepSeekConvert
from .llama_convert import LlamaConvert as YiConvert
from .llama_convert import LlamaConvert as GemmaConvert
from .chatglm_convert import ChatGLMConvert
from .chatglm2_convert import ChatGLM2Convert
from .chatglm3_convert import ChatGLM3Convert
from .chatglm4_convert import ChatGLM4Convert
from .opt_convert import OPTConvert
from .baichuan_convert import BaichuanConvert
from .baichuan2_convert import Baichuan2Convert
from .qwen_convert import QwenConvert
from .qwen2_convert import Qwen2Convert
from .yarn_llama_convert import YaRNLlamaConvert
from .telechat_convert import TelechatConvert
from .mixtral_convert import MixtralConvert
from .deepseek_moe_convert import DeepSeekV2Convert
from .deepseek_moe_convert import DeepSeekV2Convert as DeepSeekV3Convert
from .deepseek_moe_convert import DeepSeekV2Convert as DeepSeekR1Convert
from .qwen3_convert import Qwen3Convert
