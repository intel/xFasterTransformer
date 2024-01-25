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
import json
import transformers

def compatibility_check(file_path):
    config_path = os.path.join(file_path, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as _config_file:
        _config = json.load(_config_file)
        expect_version = _config.get('transformers_version')
        if not expect_version:
            return
        if expect_version != transformers.__version__:
            print("[Warning] transformers version mismatch, "
                  f"current transformers version is {transformers.__version__}, "
                  f"expect as {expect_version}.\n If you encounter compatibility issues, please try"
                  f"Run `pip install --force-reinstall transformers=={expect_version}`.")
