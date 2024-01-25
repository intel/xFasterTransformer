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
