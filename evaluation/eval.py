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
import argparse
import json
import logging
import os

import eval_models
import lm_eval
from lm_eval import tasks, evaluator, utils, models
from lm_eval.tasks import initialize_tasks, include_path

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=utils.MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument("--max_batch_size", type=int, default=None,
                        help="Maximal batch size to try with --batch_size auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=float, default=None,
                        help="Limit the number of examples per task. "
                             "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_files", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    initialize_tasks("INFO")
    args = parse_args()

    limit = None
    if args.limit and args.limit > 0:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
        limit = args.limit

    if args.tasks is None:
        print("Please Select Tasks")
        exit(1)
    else:
        task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    if (args.data_files is not None):
        dataset_kwargs = dict()
        dataset_kwargs["data_files"] = args.data_files
        task_dict = tasks.get_task_dict(task_names, dataset_path="json", test_split="train", dataset_kwargs=dataset_kwargs)
    else:
        task_dict = tasks.get_task_dict(task_names)

    if args.check_integrity:
        run_task_tests(task_list=task_names)

    model_type = args.model
    model_mapping = eval_models.MODUEL_FOR_PRETRAINING_MAPPING

    if model_type in model_mapping:
        model_name = model_mapping[model_type]
        model = getattr(eval_models, model_name)
        lm = model.create_from_arg_string(
            args.model_args, {"batch_size": args.batch_size, "max_batch_size": args.max_batch_size, "device": args.device}
        )
    else:
        lm = lm_eval.api.registry.get_model(model_type).create_from_arg_string(
            args.model_args, {"batch_size": args.batch_size, "max_batch_size": args.max_batch_size, "device": args.device}
        )


    results = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=100000,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        write_out=args.write_out
    )

    # add info about the model and few shot config
    if (model_type not in model_mapping or lm.model.get_rank() == 0):
        model_name = model_type
        results["config"] = {
            "model": model_name,
            "model_args": args.model_args,
            "num_fewshot": args.num_fewshot,
            "batch_size": args.batch_size,
            "batch_sizes": list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else [],
            "device": args.device,
            "no_cache": args.no_cache,
            "limit": args.limit,
            "bootstrap_iters": 100000,
        }

        dumped = json.dumps(results, indent=2)
        # print(dumped)

        if args.output_path:
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            with open(args.output_path, "w") as f:
                f.write(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
            f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(evaluator.make_table(results))
    else:
        return


if __name__ == "__main__":
    main()
