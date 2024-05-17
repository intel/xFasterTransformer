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
#pragma once

#include <torch/custom_class.h>
#include <torch/script.h>

#include <string>
#include <vector>

#include "xfastertransformer.h"

struct TorchAutoModel : torch::CustomClassHolder {
public:
    TorchAutoModel(std::string modelPath, std::string dtype, std::string KVCacheDtype) {
        xft::DataType dataType;
        if (dtype == "fp16") {
            dataType = xft::DataType::fp16;
        } else if (dtype == "bf16") {
            dataType = xft::DataType::bf16;
        } else if (dtype == "int8") {
            dataType = xft::DataType::int8;
        } else if (dtype == "int4") {
            dataType = xft::DataType::int4;
        } else if (dtype == "w8a8") {
            dataType = xft::DataType::w8a8;
        } else if (dtype == "nf4") {
            dataType = xft::DataType::nf4;
        } else if (dtype == "bf16_fp16") {
            dataType = xft::DataType::bf16_fp16;
        } else if (dtype == "bf16_int8") {
            dataType = xft::DataType::bf16_int8;
        } else if (dtype == "bf16_w8a8") {
            dataType = xft::DataType::bf16_w8a8;
        } else if (dtype == "bf16_int4") {
            dataType = xft::DataType::bf16_int4;
        } else if (dtype == "bf16_nf4") {
            dataType = xft::DataType::bf16_nf4;
        } else if (dtype == "w8a8_int8") {
            dataType = xft::DataType::w8a8_int8;
        } else if (dtype == "w8a8_int4") {
            dataType = xft::DataType::w8a8_int4;
        } else if (dtype == "w8a8_nf4") {
            dataType = xft::DataType::w8a8_nf4;
        } else {
            throw std::invalid_argument("Invalid data type.");
        }
        xft::DataType KVCacheDataType;
        if (KVCacheDtype == "fp16") {
            KVCacheDataType = xft::DataType::fp16;
        } else if (KVCacheDtype == "int8") {
            KVCacheDataType = xft::DataType::int8;
        } else {
            throw std::invalid_argument("Invalid KV cache data type.");
        }
        model = new xft::AutoModel(modelPath, dataType, KVCacheDataType);
    };

    ~TorchAutoModel() {
        if (model != nullptr) { delete model; }
    }

    int64_t getRank() { return static_cast<int64_t>(model->getRank()); }

    bool isDone() { return model->isDone(); }

    void input(torch::optional<torch::Tensor> inputIds) {
        int batchSize = 0;
        if (model->getRank() == 0) {
            TORCH_CHECK(inputIds.has_value(), "Make sure master's input is not None.")

            batchSize = inputIds.value().size(0);
            int seqLen = inputIds.value().size(1);

            tokenIds.resize(batchSize * seqLen);
            int64_t *p = inputIds.value().data_ptr<int64_t>();
            if (model->getRank() == 0) {
                for (int i = 0; i < batchSize * seqLen; ++i) {
                    tokenIds[i] = static_cast<int>(*p);
                    p += 1;
                }
            }
        }
        model->input(tokenIds, batchSize);
    }

    void config(torch::optional<int64_t> maxLength, torch::optional<int64_t> numBeamsOpt,
            torch::optional<int64_t> numReturnSequencesOpt, torch::optional<double> lenPenaltyOpt,
            torch::optional<bool> earlyStoppingOpt, torch::optional<int64_t> eosTokenIdOpt,
            torch::optional<int64_t> padTokenIdOpt, torch::optional<bool> doSampleOpt,
            torch::optional<double> temperaturOpt, torch::optional<int64_t> topKOpt, torch::optional<double> topPOpt,
            torch::optional<double> repetitionPenaltyOpt,
            torch::optional<std::vector<std::vector<int64_t>>> stopWordsListOpt) {
        TORCH_CHECK(maxLength.has_value(), "Make sure master's maxLen is not None.")
        int maxLen = static_cast<int>(maxLength.value());
        int numBeams = numBeamsOpt.has_value() ? (int)numBeamsOpt.value() : 1;
        int numBeamHypsToKeep = numReturnSequencesOpt.has_value() ? (int)numReturnSequencesOpt.value() : 1;
        float lenPenalty = lenPenaltyOpt.has_value() ? static_cast<float>(lenPenaltyOpt.value()) : 1.0;
        bool doEarlyStopping = earlyStoppingOpt.has_value() ? (bool)earlyStoppingOpt.value() : false;
        int eosTokenId = eosTokenIdOpt.has_value() ? static_cast<int>(eosTokenIdOpt.value()) : -1;
        int padTokenId = padTokenIdOpt.has_value() ? static_cast<int>(padTokenIdOpt.value()) : -1;
        bool doSample = doSampleOpt.has_value() ? (bool)doSampleOpt.value() : false;
        float temperature = temperaturOpt.has_value() ? static_cast<float>(temperaturOpt.value()) : 1.0;
        int topK = topKOpt.has_value() ? (int)topKOpt.value() : 50;
        float topP = topPOpt.has_value() ? static_cast<float>(topPOpt.value()) : 1.0;
        float repetitionPenalty
                = repetitionPenaltyOpt.has_value() ? static_cast<float>(repetitionPenaltyOpt.value()) : 1.0;

        std::vector<std::vector<int>> stopWordsList_int32;
        if (stopWordsListOpt.has_value()) {
            std::vector<std::vector<int64_t>> &stopWordsList = stopWordsListOpt.value();
            stopWordsList_int32.reserve(stopWordsList.size());
            for (const auto &inner_vector : stopWordsList) {
                std::vector<int> converted_vector;
                converted_vector.reserve(inner_vector.size());

                std::transform(inner_vector.begin(), inner_vector.end(), std::back_inserter(converted_vector),
                        [](int64_t value) { return static_cast<int>(value); });

                stopWordsList_int32.emplace_back(converted_vector);
            }
        }

        model->config(maxLen, numBeams, numBeamHypsToKeep, lenPenalty, doEarlyStopping, eosTokenId, padTokenId,
                doSample, temperature, topK, topP, repetitionPenalty, stopWordsList_int32);
    }

    torch::Tensor forward(torch::Tensor &inputIds) {
        int batchSize = inputIds.size(0);
        int seqLen = inputIds.size(1);
        int vocabSize = model->getVocabSize();
        int logitsN = batchSize * seqLen * vocabSize;

        if (model->getRank() == 0) { input(inputIds); }

        std::tuple<float *, int, int> result = model->forward();
        float *outBuf = std::get<0>(result);
        int sampleOffset = std::get<1>(result);
        int sampleSize = std::get<2>(result);

        // Create a torch::Tensor from the C array
        int64_t tdims[3] = {batchSize, seqLen, vocabSize};
        torch::Tensor ret = torch::from_blob(outBuf, tdims, torch::kFloat32);
        return ret;
    }

    torch::Tensor generate() {
        auto nextTokens = model->generate();

        int batchSize = model->getBatchSize();
        int numBeams = model->getConfig().numBeams;

        torch::Tensor ret = torch::empty({batchSize, numBeams}, torch::kInt64);
        int64_t *p = ret.data_ptr<int64_t>();
        for (int i = 0; i < nextTokens.size(); ++i) {
            p[i] = nextTokens[i];
        }
        return ret;
    }

    torch::Tensor finalize() {
        auto outputs = model->finalize();

        int batchSize = model->getBatchSize();
        int numBeamHypsToKeep = model->getConfig().numBeamHypsToKeep;
        int outputLen = outputs.size() / (batchSize * numBeamHypsToKeep);

        torch::Tensor ret = torch::empty({batchSize * numBeamHypsToKeep, outputLen}, torch::kInt64);
        int64_t *p = ret.data_ptr<int64_t>();
        for (int i = 0; i < outputs.size(); ++i) {
            p[i] = outputs[i];
        }
        return ret;
    }

    void setPrefix(torch::optional<torch::Tensor> inputIds) {
        std::vector<int> prefixIds;
        if (model->getRank() == 0) {
            TORCH_CHECK(inputIds.has_value(), "Make sure master's prefix input is not None.")
            TORCH_CHECK(inputIds.value().dim() <= 2, "Prefix sharing input expected dim <= 2 but tensor has ",
                    inputIds.value().dim());
            inputIds.value().squeeze();
            TORCH_CHECK(inputIds.value().dim() == 2, "Prefix sharing only support 1 prompt but input has ",
                    inputIds.value().size(0));

            int seqLen = inputIds.value().size(-1);

            prefixIds.resize(seqLen);
            int64_t *p = inputIds.value().data_ptr<int64_t>();
            for (int i = 0; i < seqLen; ++i) {
                prefixIds[i] = static_cast<int>(*p);
                p += 1;
            }
        }

        model->setPrefix(prefixIds);
    };

    void unsetPrefix() { model->unsetPrefix(); };

    torch::Tensor forwardCB() {
        // Forward for continuous batching
        int batchSize = model->getBatchSize();
        int vocabSize = model->getVocabSize();

        std::tuple<float *, int, int> result = model->forward(false);
        float *outBuf = std::get<0>(result);
        int sampleOffset = std::get<1>(result);
        int sampleSize = std::get<2>(result);

        // Create a torch::Tensor from the C array
        int64_t tdims[2] = {batchSize, vocabSize};
        torch::Tensor ret = torch::from_blob(outBuf, tdims, torch::kFloat32);
        return ret;
    }

    torch::Tensor setInputCB(torch::optional<torch::Tensor> inputIds_, torch::optional<torch::Tensor> seqLens_,
            torch::optional<torch::Tensor> seqIDs_, torch::optional<torch::Tensor> maxLens_) {
        int batchSize = 0;
        std::vector<int> seqLens;
        std::vector<int> seqIDs;
        std::vector<int> maxLens;
        if (model->getRank() == 0) {
            TORCH_CHECK(inputIds_.has_value(), "Make sure master's input is not None.")
            TORCH_CHECK(inputIds_.value().dim() == 2 || inputIds_.value().dim() == 1,
                    "Make sure master's input is 1-D or 2-D.")

            int totalSeqLen;

            if (inputIds_.value().dim() == 2) {
                batchSize = inputIds_.value().size(0);
                totalSeqLen = inputIds_.value().size(0) * inputIds_.value().size(1);
                seqLens.assign(batchSize, inputIds_.value().size(1));
            } else {
                TORCH_CHECK(seqLens_.has_value(), "Make sure master's seqLens_ is not None when input is 1-D.")
                TORCH_CHECK(seqLens_.value().dim() == 1, "Make sure master's seqLens_ is 1-D.")
                TORCH_CHECK(!seqIDs_.has_value(), "Make sure master's seqIDs_ is None when input is 1-D.")
                batchSize = seqLens_.value().size(0);
                totalSeqLen = inputIds_.value().size(0);
                torch::Tensor seqLensTensor = seqLens_.value().to(torch::kInt32);
                seqLens.resize(batchSize);
                memcpy(seqLens.data(), seqLensTensor.data_ptr<int>(), batchSize * sizeof(int));
            }

            torch::Tensor inputTensor = inputIds_.value().to(torch::kInt32);

            tokenIds.resize(totalSeqLen);
            memcpy(tokenIds.data(), inputTensor.data_ptr<int>(), totalSeqLen * sizeof(int));

            if (seqIDs_.has_value()) {
                torch::Tensor seqIDsTensor = seqIDs_.value().to(torch::kInt32);
                TORCH_CHECK(batchSize == seqIDsTensor.size(0), "seqIDs'shape must equal to batchSize.")
                seqIDs.resize(batchSize);
                memcpy(seqIDs.data(), seqIDsTensor.data_ptr<int>(), batchSize * sizeof(int));
            }

            if (maxLens_.has_value()) {
                torch::Tensor maxLensTensor = maxLens_.value().to(torch::kInt32);
                TORCH_CHECK(maxLensTensor.size(-1) == batchSize || maxLensTensor.size(-1) == 1,
                        "maxLens size must equal to inputIds size[0] or 1.")
                maxLens.resize(batchSize);
                memcpy(maxLens.data(), maxLensTensor.data_ptr<int>(), batchSize * sizeof(int));
            }
        }

        seqIDs = model->set_input(tokenIds, seqLens, seqIDs, maxLens);
        torch::Tensor ret = torch::from_blob(seqIDs.data(), {batchSize}, torch::kInt32).to(torch::kInt64);
        return ret;
    }

    bool freeSeqs(torch::optional<torch::Tensor> seqIDs_) {
        std::vector<int> seqIDs;
        if (model->getRank() == 0) {
            TORCH_CHECK(seqIDs_.has_value(), "Make sure master's input is not None.")
            torch::Tensor seqIDsTensor = seqIDs_.value().to(torch::kInt32);
            seqIDs.resize(seqIDsTensor.size(0));
            memcpy(seqIDs.data(), seqIDsTensor.data_ptr<int>(), seqIDsTensor.size(0) * sizeof(int));
        }
        return model->freeSeqs(seqIDs);
    }

private:
    xft::Model *model;
    std::vector<int> tokenIds;
};
