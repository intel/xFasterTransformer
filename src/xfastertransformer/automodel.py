import torch
from typing import Union, Literal


class AutoModel:
    def __init__(self, path, dtype: str = "fp16"):
        if dtype in ["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"]:
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
    ):
        self.model.config(
            max_length, num_beams, num_return_sequences, length_penalty, early_stopping, eos_token_id, pad_token_id
        )

    def input(self, input_ids=None):
        self.model.input(input_ids)

    def forward(self):
        return self.model.generate()

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
            max_length, num_beams, num_return_sequences, length_penalty, early_stopping, eos_token_id, pad_token_id
        )
        self.input(input_ids)

        while not self.is_done():
            next_tokens = self.forward()
            if streamer is not None and input_ids is not None:
                streamer.put(next_tokens.cpu())

        if streamer is not None and input_ids is not None:
            streamer.end()

        return self.finalize()
