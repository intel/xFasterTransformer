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
import torch
import os
import sys
from types import ModuleType
from typing import Any
from typing import TYPE_CHECKING
from ctypes import *


def with_mpirun():
    return any(os.getenv(env) for env in ["MPI_LOCALRANKID", "MPI_LOCALNRANKS", "PMI_RANK", "PMI_SIZE", "PMIX_RANK"])


if os.getenv("SINGLE_INSTANCE", "0") == "0" and with_mpirun():
    cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__)) + "/libxft_comm_helper.so")

torch.classes.load_library(os.path.dirname(os.path.abspath(__file__)) + "/libxfastertransformer_pt.so")

_import_structure = {
    "automodel": ["AutoModel"],
    "tools": [
        "LlamaConvert",
        "YiConvert",
        "GemmaConvert",
        "ChatGLMConvert",
        "ChatGLM2Convert",
        "ChatGLM3Convert",
        "OPTConvert",
        "BaichuanConvert",
        "Baichuan2Convert",
        "QwenConvert",
        "Qwen2Convert",
        "YaRNLlamaConvert",
        "DeepseekConvert",
    ],
    "env": ["get_env"],
}

if TYPE_CHECKING:
    from .tools import LlamaConvert
    from .tools import DeepseekConvert
    from .tools import LlamaConvert as YiConvert
    from .tools import LlamaConvert as GemmaConvert
    from .tools import ChatGLMConvert
    from .tools import ChatGLM2Convert
    from .tools import ChatGLM3Convert
    from .tools import OPTConvert
    from .tools import BaichuanConvert
    from .tools import Baichuan2Convert
    from .tools import QwenConvert
    from .tools import Qwen2Convert
    from .tools import YaRNLlamaConvert
    from .env import get_env
else:
    # This LazyImportModule is refer to optuna.integration._IntegrationModule
    # Source code url https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    class _LazyImportModule(ModuleType):
        """
        This class applies lazy import under `xfastertransformer`, where submodules are imported when they
        are actually accessed. Otherwise, `import xfastertransformer` will import some unnecessary dependencise.
        """

        __file__ = globals()["__file__"]
        __path__ = [os.path.dirname(__file__)]

        _modules = set(_import_structure.keys())
        _class_to_module = {}
        for key, values in _import_structure.items():
            for value in values:
                _class_to_module[value] = key

        def __getattr__(self, name: str) -> Any:
            if name in self._modules:
                value = self._get_module(name)
            elif name in self._class_to_module.keys():
                module = self._get_module(self._class_to_module[name])
                value = getattr(module, name)
            else:
                raise AttributeError("module {} has no attribute {}".format(self.__name__, name))

            setattr(self, name, value)
            return value

        def _get_module(self, module_name: str) -> ModuleType:
            import importlib

            try:
                return importlib.import_module("." + module_name, self.__name__)
            except ModuleNotFoundError:
                raise ModuleNotFoundError(f"Fail to import module {module_name}.")

    sys.modules[__name__] = _LazyImportModule(__name__)
