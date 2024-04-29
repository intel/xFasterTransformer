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
import math
import torch
import torch.nn.functional as F
import os
from collections import OrderedDict
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

import transformers
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
)
from transformers import AutoTokenizer

from lm_eval.models.huggingface import HFLM
from lm_eval.api.model import CacheHook
torch.classes.load_library(os.path.dirname(os.path.abspath(__file__)) + '/../build/libevaluation.so')


MODUEL_FOR_PRETRAINING_MAPPING = OrderedDict([
    ("gpt", "CausalDecoderLM"),
    ("llama", "CausalDecoderLM"),
    ("baichuan", "CausalDecoderLM"),
    ("chatglm", "XGLMDecoderLM"),
    ("chatglm2", "XGLMDecoderLM"),
    ("chatglm3", "XGLMDecoderLM"),
])

class CausalDecoderLM(HFLM):
    """Causal language modeling for evaluation.
    """
    def __init__(
        self,
        weights: Optional[str] = None,
        pretrained: Optional[str] = None,
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[str] = None,
        truncation: Optional[bool] = False,
        max_length: Optional[int] = None,
        device: Optional[str] = "cpu",
        dtype: Optional[Union[str, torch.dtype]] = "bf16",
        kvtype: Optional[Union[str, torch.dtype]] = "fp16",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
    ) -> None:

        # use xft's parallel method
        self._rank = 0
        self._world_size = 1
        self.cache_hook = CacheHook(None)

        assert isinstance(weights, str)
        assert isinstance(pretrained, str)
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        revision = revision + ("/" + subfolder if subfolder is not None else "")

        self._device = device

        self.AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

        self._model = torch.classes.evaluation.EvalAutoDecoder(weights, dtype, kvtype)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, use_fast=False, padding_side='left', trust_remote_code=trust_remote_code)

        self.truncation = truncation

        self.vocab_size = self.tokenizer.vocab_size
        self._max_length = max_length

        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = max_batch_size
        self.batch_size_per_gpu = int(batch_size)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        print("Not Implemented yet")
        pass

    @property
    def max_length(self):
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length

    def _model_call(
        self, inputs, attn_mask=None, labels=None):
        ret = self.model.forward_logits_all(inputs)
        return ret


class XGLMDecoderLM(CausalDecoderLM):
    """Prefix language modeling for evaluation.
    """

    #def __init__(
    #    self,
    #    *args,
    #    **kwargs
    #) -> None:
 
    def tok_encode(
        self, string: str, left_truncate_len=None, add_special_tokens=None
    ) -> List[int]:
        """ """
        if add_special_tokens is None:
            add_special_tokens = True

        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)

        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]

        return encoding

    def _encode_pair(
        self, context: str, continuation: str
    ) -> Tuple[List[int], List[int]]:
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        context_enc = self.tok_encode(context, add_special_tokens=True)
        if hasattr(self.tokenizer, "get_prefix_tokens"):
            # prefix for chatglm2/3
            whole_enc = self.tok_encode(context + continuation, add_special_tokens=True)
            context_enc_len = len(context_enc)
            continuation_enc = whole_enc[context_enc_len:]
        else:
            # suffix for chatglm
            continuation_enc = self.tok_encode(continuation, add_special_tokens=False)

        return context_enc, continuation_enc


    #def _model_call(
    #    self, inputs, attn_mask=None, labels=None):

    #def _select_cont_toks(self, logits, contlen=None, inplen=None):

    #def _loglikelihood_tokens(
    #    self,
    #    requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
    #    disable_tqdm: bool = False,
    #    override_bs: int = None,
    #) -> List[Tuple[float, bool]]:
