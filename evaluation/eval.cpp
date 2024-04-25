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
#include "eval.h"
#include "baichuan.h"
#include "chatglm.h"
#include "chatglm2.h"
#include "chatglm3.h"
#include "common_decoder.h"
#include "datatypes.h"
#include "gemma.h"
#include "hybrid_model.h"
#include "llama.h"
#include "opt_decoder.h"
#include "qwen.h"
#include "qwen2.h"
#include "yarn_llama.h"

EvalAutoDecoder::EvalAutoDecoder(std::string modelPath, std::string dtype, std::string kvCacheType = "fp16") {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    if (reader.ParseError() < 0) {
        printf("Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();
    vocabSize = reader.GetInteger(modeltype, "vocab_size");

    pdecoder = DecoderFactory::Create(modeltype + "-" + getTypeName(dtype) + "-" + getTypeName(kvCacheType), modelPath);
};

std::string EvalAutoDecoder::getTypeName(std::string dtype) {
    static const std::unordered_map<std::string, xft::DataType> dtypeMap = {
            {"fp32", xft::DataType::fp32},
            {"bf16", xft::DataType::bf16},
            {"fp16", xft::DataType::fp16},
            {"int8", xft::DataType::int8},
            {"w8a8", xft::DataType::w8a8},
            {"int4", xft::DataType::int4},
            {"nf4", xft::DataType::nf4},
            {"bf16_fp16", xft::DataType::bf16_fp16},
            {"bf16_w8a8", xft::DataType::bf16_w8a8},
            {"bf16_int8", xft::DataType::bf16_int8},
            {"bf16_int4", xft::DataType::bf16_int4},
            {"bf16_nf4", xft::DataType::bf16_nf4},
            {"w8a8_int8", xft::DataType::w8a8_int8},
            {"w8a8_int4", xft::DataType::w8a8_int4},
            {"w8a8_nf4", xft::DataType::w8a8_nf4},
    };

    xft::DataType xftType;
    auto it = dtypeMap.find(dtype);
    if (it != dtypeMap.end()) {
        xftType = it->second;
    } else {
        xftType = xft::DataType::unknown;
    }
    return xft::getTypeIdName(xftType);
}

torch::Tensor EvalAutoDecoder::forward(torch::Tensor &inputIds) {

    int batchSize = inputIds.size(0);
    int seqLen = inputIds.size(1);

    int logitsN = batchSize * seqLen * vocabSize;
    float *decBuf = (float *)SimpleMemPool::instance().getBuffer("evalDecoderBuf", 2 * logitsN * sizeof(float));
    int *tokenIds = (int *)SimpleMemPool::instance().getBuffer("evalTokenIds", batchSize * seqLen * sizeof(int));

    float *recvBuf = decBuf;
    float *logits = decBuf + logitsN;

    // Prepare input token IDs
    for (int i = 0; i < batchSize; ++i) {
        for (int j = 0; j < seqLen; j++)
            tokenIds[i * seqLen + j] = (int)(inputIds[i][j].item<int64_t>());
    }

    int64_t dims[3] = {batchSize, 1, seqLen};

    std::tuple<float *, int, int> result = pdecoder->forward(tokenIds, dims, 0, true);

    float *outBuf = std::get<0>(result);
    int sampleOffset = std::get<1>(result);
    int sampleSize = std::get<2>(result);

    Messenger &messenger = pdecoder->getMessenger();
    int rank = messenger.getRank();
    int worldSize = messenger.getSize();

    if (worldSize > 1) {
        int rCount[worldSize];
        std::vector<long unsigned int> recvCount(worldSize, 1);

        messenger.allgatherv((float *)(&sampleSize), 1, (float *)rCount, recvCount);

        for (int i = 0; i < worldSize; ++i) {
            recvCount[i] = batchSize * seqLen * rCount[i];
        }
        messenger.allgatherv(outBuf, recvCount[rank], recvBuf, recvCount);

        for (int m = 0; m < batchSize; ++m) {
            for (int i = 0; i < seqLen; ++i) {
                int off = 0;
                for (int n = 0; n < worldSize; ++n) {
                    for (int j = 0; j < rCount[n]; ++j) {
                        logits[m * seqLen * vocabSize + i * vocabSize + j + off]
                                = recvBuf[off * batchSize * seqLen + m * seqLen * rCount[n] + i * rCount[n] + j];
                    }
                    off += rCount[n];
                }
            }
        }
    } else {
        memcpy(logits, outBuf, logitsN * sizeof(float));
    }

    // Create a torch::Tensor from the C array
    int64_t tdims[3] = {batchSize, seqLen, vocabSize};
    torch::Tensor ret = torch::from_blob(logits, tdims, torch::kFloat32);

    return ret;
}

int64_t EvalAutoDecoder::getRank() {
    return static_cast<int64_t>(pdecoder->getRank());
}
