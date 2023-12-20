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
#include "hybrid_model.h"
#include "llama.h"
#include "opt_decoder.h"

EvalAutoDecoder::EvalAutoDecoder(std::string modelPath, std::string dtype) {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    if (reader.ParseError() < 0) {
        printf("Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modelType = *reader.Sections().begin();
    vocabSize = reader.GetInteger(modelType, "vocab_size");

    if (modelType == "gpt") {
        if (dtype == "fp16") {
            pdecoder = new OptDecoder<float16_t>(modelPath);
        } else if (dtype == "int8") {
            pdecoder = new OptDecoder<int8_t>(modelPath);
        } else if (dtype == "int4") {
            pdecoder = new OptDecoder<uint4x2_t>(modelPath);
        } else if (dtype == "bf16") {
            pdecoder = new OptDecoder<bfloat16_t>(modelPath);
        } else if (dtype == "bf16_fp16") {
            pdecoder = new HybridModel<OptDecoder, bfloat16_t, float16_t>(modelPath);
        } else if (dtype == "bf16_int8") {
            pdecoder = new HybridModel<OptDecoder, bfloat16_t, int8_t>(modelPath);
        } else if (dtype == "bf16_int4") {
            pdecoder = new HybridModel<OptDecoder, bfloat16_t, uint4x2_t>(modelPath);
        } else {
            throw std::invalid_argument("Invalid DataType");
        }
    } else if (modelType == "llama") {
        if (dtype == "fp16") {
            pdecoder = new LlamaLLM<float16_t>(modelPath);
        } else if (dtype == "int8") {
            pdecoder = new LlamaLLM<int8_t>(modelPath);
        } else if (dtype == "int4") {
            pdecoder = new LlamaLLM<uint4x2_t>(modelPath);
        } else if (dtype == "bf16") {
            pdecoder = new LlamaLLM<bfloat16_t>(modelPath);
        } else if (dtype == "nf4") {
            pdecoder = new LlamaLLM<nf4x2_t>(modelPath);
        } else if (dtype == "w8a8") {
            pdecoder = new LlamaLLM<w8a8_t>(modelPath);
        } else if (dtype == "bf16_fp16") {
            pdecoder = new HybridModel<LlamaLLM, bfloat16_t, float16_t>(modelPath);
        } else if (dtype == "bf16_int8") {
            pdecoder = new HybridModel<LlamaLLM, bfloat16_t, int8_t>(modelPath);
        } else if (dtype == "bf16_int4") {
            pdecoder = new HybridModel<LlamaLLM, bfloat16_t, uint4x2_t>(modelPath);
        } else if (dtype == "bf16_nf4") {
            pdecoder = new HybridModel<LlamaLLM, bfloat16_t, nf4x2_t>(modelPath);
        } else if (dtype == "bf16_w8a8") {
            pdecoder = new HybridModel<LlamaLLM, bfloat16_t, w8a8_t>(modelPath);
        }else {
            throw std::invalid_argument("Invalid DataType");
        }
    } else if (modelType == "baichuan") {
        if (dtype == "fp16") {
            pdecoder = new Baichuan<float16_t>(modelPath);
        } else if (dtype == "int8") {
            pdecoder = new Baichuan<int8_t>(modelPath);
        } else if (dtype == "int4") {
            pdecoder = new Baichuan<uint4x2_t>(modelPath);
        } else if (dtype == "bf16") {
            pdecoder = new Baichuan<bfloat16_t>(modelPath);
        } else if (dtype == "nf4") {
            pdecoder = new Baichuan<nf4x2_t>(modelPath);
        } else if (dtype == "w8a8") {
            pdecoder = new Baichuan<w8a8_t>(modelPath);
        } else if (dtype == "bf16_fp16") {
            pdecoder = new HybridModel<Baichuan, bfloat16_t, float16_t>(modelPath);
        } else if (dtype == "bf16_int8") {
            pdecoder = new HybridModel<Baichuan, bfloat16_t, int8_t>(modelPath);
        } else if (dtype == "bf16_int4") {
            pdecoder = new HybridModel<Baichuan, bfloat16_t, uint4x2_t>(modelPath);
        } else if (dtype == "bf16_nf4") {
            pdecoder = new HybridModel<Baichuan, bfloat16_t, nf4x2_t>(modelPath);
        } else if (dtype == "bf16_w8a8") {
            pdecoder = new HybridModel<Baichuan, bfloat16_t, w8a8_t>(modelPath);
        } else {
            throw std::invalid_argument("Invalid DataType");
        }
    } else if (modelType == "chatglm") {
        if (dtype == "fp16") {
            pdecoder = new ChatGLM<float16_t>(modelPath);
        } else if (dtype == "int8") {
            pdecoder = new ChatGLM<int8_t>(modelPath);
        } else if (dtype == "int4") {
            pdecoder = new ChatGLM<uint4x2_t>(modelPath);
        } else if (dtype == "bf16") {
            pdecoder = new ChatGLM<bfloat16_t>(modelPath);
        } else if (dtype == "nf4") {
            pdecoder = new ChatGLM<nf4x2_t>(modelPath);
        } else if (dtype == "w8a8") {
            pdecoder = new ChatGLM<w8a8_t>(modelPath);
        } else if (dtype == "bf16_fp16") {
            pdecoder = new HybridModel<ChatGLM, bfloat16_t, float16_t>(modelPath);
        } else if (dtype == "bf16_int8") {
            pdecoder = new HybridModel<ChatGLM, bfloat16_t, int8_t>(modelPath);
        } else if (dtype == "bf16_int4") {
            pdecoder = new HybridModel<ChatGLM, bfloat16_t, uint4x2_t>(modelPath);
        } else if (dtype == "bf16_nf4") {
            pdecoder = new HybridModel<ChatGLM, bfloat16_t, nf4x2_t>(modelPath);
        } else if (dtype == "bf16_w8a8") {
            pdecoder = new HybridModel<ChatGLM, bfloat16_t, w8a8_t>(modelPath);
        } else {
            throw std::invalid_argument("Invalid DataType");
        }
    } else if (modelType == "chatglm2") {
        if (dtype == "fp16") {
            pdecoder = new ChatGLM2<float16_t>(modelPath);
        } else if (dtype == "int8") {
            pdecoder = new ChatGLM2<int8_t>(modelPath);
        } else if (dtype == "int4") {
            pdecoder = new ChatGLM2<uint4x2_t>(modelPath);
        } else if (dtype == "bf16") {
            pdecoder = new ChatGLM2<bfloat16_t>(modelPath);
        } else if (dtype == "nf4") {
            pdecoder = new ChatGLM2<nf4x2_t>(modelPath);
        } else if (dtype == "w8a8") {
            pdecoder = new ChatGLM2<w8a8_t>(modelPath);
        } else if (dtype == "bf16_fp16") {
            pdecoder = new HybridModel<ChatGLM2, bfloat16_t, float16_t>(modelPath);
        } else if (dtype == "bf16_int8") {
            pdecoder = new HybridModel<ChatGLM2, bfloat16_t, int8_t>(modelPath);
        } else if (dtype == "bf16_int4") {
            pdecoder = new HybridModel<ChatGLM2, bfloat16_t, uint4x2_t>(modelPath);
        } else if (dtype == "bf16_nf4") {
            pdecoder = new HybridModel<ChatGLM2, bfloat16_t, nf4x2_t>(modelPath);
        } else if (dtype == "bf16_w8a8") {
            pdecoder = new HybridModel<ChatGLM2, bfloat16_t, w8a8_t>(modelPath);
        } else {
            throw std::invalid_argument("Invalid DataType");
        }
    } else if (modelType == "chatglm3") {
        if (dtype == "fp16") {
            pdecoder = new ChatGLM3<float16_t>(modelPath);
        } else if (dtype == "int8") {
            pdecoder = new ChatGLM3<int8_t>(modelPath);
        } else if (dtype == "int4") {
            pdecoder = new ChatGLM3<uint4x2_t>(modelPath);
        } else if (dtype == "bf16") {
            pdecoder = new ChatGLM3<bfloat16_t>(modelPath);
        } else if (dtype == "nf4") {
            pdecoder = new ChatGLM3<nf4x2_t>(modelPath);
        } else if (dtype == "w8a8") {
            pdecoder = new ChatGLM3<w8a8_t>(modelPath);
        } else if (dtype == "bf16_fp16") {
            pdecoder = new HybridModel<ChatGLM3, bfloat16_t, float16_t>(modelPath);
        } else if (dtype == "bf16_int8") {
            pdecoder = new HybridModel<ChatGLM3, bfloat16_t, int8_t>(modelPath);
        } else if (dtype == "bf16_int4") {
            pdecoder = new HybridModel<ChatGLM3, bfloat16_t, uint4x2_t>(modelPath);
        } else if (dtype == "bf16_nf4") {
            pdecoder = new HybridModel<ChatGLM3, bfloat16_t, nf4x2_t>(modelPath);
        } else if (dtype == "bf16_w8a8") {
            pdecoder = new HybridModel<ChatGLM3, bfloat16_t, w8a8_t>(modelPath);
        } else {
            throw std::invalid_argument("Invalid DataType");
        }
    } else {
        throw std::invalid_argument("Invalid ModelType");
    }
};

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

        messenger.allgatherv(&sampleSize, 1, rCount, recvCount);

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
