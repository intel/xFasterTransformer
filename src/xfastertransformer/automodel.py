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
import torch
from typing import Union, Literal


class AutoModel:
    def __init__(self, path, dtype: str = "fp16"):
        if dtype in ["fp16", "bf16", "int8", "w8a8", "int4", "nf4", "bf16_fp16", "bf16_int8", "bf16_w8a8",
                     "bf16_int4", "bf16_nf4", "w8a8_int8", "w8a8_int4", "w8a8_nf4"]:
            self.model = torch.classes.xfastertransformer.AutoModel(path, dtype)
        else:
            raise Exception(f"{self.__class__.__name__} don't support {dtype}.")

    @classmethod
    def from_pretrained(cls, path, dtype: str = "fp16"):
        return cls(path, dtype)

    @property
    def rank(self):
        return self.model.get_rank()

    def finalize(self):
        return self.model.finalize()

    def is_done(self):
        return self.model.is_done()

    def config(
        self,
        max_length=20,
        num_beams=1,
        num_return_sequences=1,
        length_penalty=1.0,
        early_stopping=False,
        eos_token_id=-1,
        pad_token_id=-1,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
    ):
        self.model.config(
            max_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            early_stopping,
            eos_token_id,
            pad_token_id,
            do_sample,
            temperature,
            top_k,
            top_p,
        )

    def input(self, input_ids=None):
        self.model.input(input_ids)

    def forward(self):
        return self.model.generate()

    def prefix_sharing(self, input_ids=None, truncate_tail=0):
        if input_ids is not None and truncate_tail > 0:
            input_ids = input_ids[:, :-truncate_tail]

        self.model.set_prefix(input_ids)

    def disable_prefix_sharing(self):
        self.model.unset_prefix()

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        max_length=20,
        num_beams=1,
        num_return_sequences=1,
        length_penalty=1.0,
        early_stopping=False,
        eos_token_id=-1,
        pad_token_id=-1,
        do_sample=False,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        streamer=None,
    ):
        #  streamer: Optional["BaseStreamer"] = None):
        if streamer is not None:
            if num_beams > 1:
                raise ValueError(
                    "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
                )
            if input_ids is not None and input_ids.size(0) != 1:
                raise ValueError("`streamer` cannot be used with batch size != 1 (yet!).")
            if input_ids is not None:
                streamer.put(input_ids.cpu())

        self.config(
            max_length,
            num_beams,
            num_return_sequences,
            length_penalty,
            early_stopping,
            eos_token_id,
            pad_token_id,
            do_sample,
            temperature,
            top_k,
            top_p,
        )
        self.input(input_ids)

        while not self.is_done():
            next_tokens = self.forward()
            if streamer is not None and input_ids is not None:
                streamer.put(next_tokens.cpu())

        if streamer is not None and input_ids is not None:
            streamer.end()

        return self.finalize()
