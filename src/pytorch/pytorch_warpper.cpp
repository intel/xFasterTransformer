#include "auto_model.h"

// Referred to https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html
TORCH_LIBRARY(xfastertransformer, m) {
    m.class_<TorchAutoModel>("AutoModel")
            .def(torch::init<std::string, std::string>())
            .def("get_rank", &TorchAutoModel::getRank)
            .def("input", &TorchAutoModel::input)
            .def("config", &TorchAutoModel::config)
            .def("is_done", &TorchAutoModel::isDone)
            .def("generate", &TorchAutoModel::generate)
            .def("finalize", &TorchAutoModel::finalize);
}
