import json
import tqdm
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union, Dict
import argparse
import time

import importlib.util

xft_spec = importlib.util.find_spec("xfastertransformer")

if xft_spec is None:
    import sys

    sys.path.append("/root/xFasterTransformer/src")
    print("[INFO] xfastertransformer is not installed in pip, using source code.")
else:
    print("[INFO] xfastertransformer is installed, using pip installed package.")

import xfastertransformer

class LambadaDataset(torch.utils.data.Dataset):
    """ LAMBADA dataset class. """

    def __init__(self,
                 path: str,
                 tokenizer: transformers.PreTrainedTokenizerBase):
        self.tokenizer = tokenizer
        with open(path, 'r') as f:
            inputs, targets = zip(*[
                json.loads(line)["text"] .strip('\n').rsplit(' ', 1)
                for line in f.readlines()])
            # This whitespace preprocessing (additional space to the target)
            # is required.
            #targets = [' ' + tgt for tgt in targets]
            self.encodings = self.tokenizer(list(inputs),
                                            targets,
                                            padding=True,
                                            add_special_tokens=False,
                                            return_token_type_ids=True,
                                            return_tensors='pt')

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return dict(
            input_ids=self.encodings['input_ids'][idx],
            attention_mask=self.encodings['attention_mask'][idx],
            token_type_ids=self.encodings['token_type_ids'][idx]
        )


class Timer:

    def __init__(self):
        self._start_times = {}
        self._total_elapsed_times = {}

    def start(self, tag='__default'):
        self._start_times[tag] = time.time()

    def stop(self, tag='__default'):
        elapsed_time = time.time() - self._start_times[tag]
        if tag not in self._total_elapsed_times:
            self._total_elapsed_times[tag] = 0
        self._total_elapsed_times[tag] += elapsed_time
        return elapsed_time

    def elapsed_time_in_sec(self, tag='__default'):
        if tag not in self._total_elapsed_times:
            return None
        return self._total_elapsed_times[tag]

    def reset(self):
        self._start_times.clear()
        self._total_elapsed_times.clear()

def split_inputs_and_targets(entries: Dict[str, torch.LongTensor],
                             pad_token_id: int,
                             pad_to_left=False):
    input_ids = entries['input_ids']
    attn_mask = entries['attention_mask']
    token_type_ids = entries['token_type_ids']

    # Split inputs and labels by token_type_ids.
    input_token_ids = [
        ids[(mask == 1) & (type_ids == 0)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    # FT allows int32 tensors.
    input_lengths = torch.tensor(
        [len(input_tokens) for input_tokens in input_token_ids]).int()
    max_length = input_lengths.max()
    input_token_ids = torch.stack([
        torch.nn.functional.pad(
            token_ids,
            pad=[max_length - len(token_ids), 0]
                if pad_to_left else [0, max_length - len(token_ids)],
            mode='constant',
            value=pad_token_id
        ) for token_ids in input_token_ids])
    target_token_ids = [
        ids[(mask == 1) & (type_ids == 1)]
        for ids, mask, type_ids in zip(input_ids, attn_mask, token_type_ids)]
    return input_token_ids, input_lengths, target_token_ids

def get_args():
    DTYPE_LIST = ["fp16", "bf16", "int8", "w8a8", "bf16_fp16", "bf16_int8"]
    
    parser = argparse.ArgumentParser(
        'Evaluation: LAMBADA Task',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('LAMBADA Task Parameters')
    group.add_argument(
        '--dataset_path', type=str, metavar='PATH', required=True,
        help="A file path to LAMBADA task dataset.")
    group.add_argument(
        "--token_path", type=str, metavar='DIR_OR_PATH', default=None,
        help='A file path of a pretrained tokenizer or a checkpoint directory '
             'of HF pretrained model.')
    group.add_argument("--model_path", type=str, default="/data/chatglm-6b-cpu", help="Path to model file")
    group.add_argument("--dtype", type=str, choices=DTYPE_LIST, default="fp16", help="Data type")
    group.add_argument("--batch", default=1, type=int)
    group.add_argument("--beam", default=1, type=int)
    group.add_argument(
        '--show_progress', action='store_true',
        help='Show evaluation progress')
    args = parser.parse_args()

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f' - {k.ljust(25, ".")}: {v}')
    print('=================================================')

    return args

def main():
    args = get_args()

    model = xfastertransformer.AutoModel.from_pretrained(args.model_path, dtype=args.dtype)

    if model.rank == 0:
        # Master
        tokenizer = AutoTokenizer.from_pretrained(args.token_path, padding_side="left")

        dataset = LambadaDataset(args.dataset_path, tokenizer=tokenizer)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch)

        num_requests = 0
        num_corrects = 0

        timer = Timer()
        if args.show_progress:
            data_loader = tqdm.tqdm(data_loader)

        for entries in data_loader:
            input_token_ids, input_lengths, target_token_ids = \
                split_inputs_and_targets(entries, tokenizer.pad_token_id, True)

            batch_size = input_token_ids.shape[0]
            output_length = max([len(target) for target in target_token_ids])

            timer.start()
            outputs = model.generate(input_token_ids, max_length=input_lengths+output_length, num_beams=args.beam)
            timer.stop()
            output_token_ids = outputs[:, input_token_ids.shape[1]:]
            output_token_ids = [
                    out[:len(tgt)].cpu()
                    for out, tgt in zip(output_token_ids, target_token_ids)]

            output_texts = tokenizer.batch_decode(output_token_ids)
            target_texts = tokenizer.batch_decode(target_token_ids)
            print('\n', output_texts, target_texts, flush=True)

            for i in range(batch_size):
                out = output_token_ids[i]
                tgt = target_token_ids[i]
                is_correct = (tgt == out).all()
                num_corrects += int(is_correct)

            num_requests += batch_size

        accuracy = num_corrects * 100 / num_requests
        print(f'Accuracy: {accuracy:0.4f}% ({num_corrects}/{num_requests}) '
              f'(elapsed time: {timer.elapsed_time_in_sec():.4f} sec)')
    else:
        # Slave
        while True:
            model.generate()


if __name__ == "__main__":
    main()

