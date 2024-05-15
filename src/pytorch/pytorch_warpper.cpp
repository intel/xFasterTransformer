// Copyright (c) 2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
#include "auto_model.h"

// Referred to https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html
TORCH_LIBRARY(xfastertransformer, m) {
    m.class_<TorchAutoModel>("AutoModel")
            .def(torch::init<std::string, std::string, std::string>())
            .def("get_rank", &TorchAutoModel::getRank)
            .def("input", &TorchAutoModel::input)
            .def("config", &TorchAutoModel::config)
            .def("set_input_cb", &TorchAutoModel::setInputCB)
            .def("is_done", &TorchAutoModel::isDone)
            .def("forward", &TorchAutoModel::forward)
            .def("forward_cb", &TorchAutoModel::forwardCB)
            .def("generate", &TorchAutoModel::generate)
            .def("finalize", &TorchAutoModel::finalize)
            .def("free_seqs", &TorchAutoModel::freeSeqs)
            .def("set_prefix", &TorchAutoModel::setPrefix)
            .def("unset_prefix", &TorchAutoModel::unsetPrefix);
}
