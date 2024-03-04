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
#include "models.h"

#include <string.h>

#include <stdexcept>

#include "INIReader.h"
#include "baichuan.h"
#include "chatglm.h"
#include "chatglm2.h"
#include "chatglm3.h"
#include "hybrid_model.h"
#include "llama.h"
#include "opt_decoder.h"
#include "qwen.h"
#include "searcher.h"
#include "timeline.h"
#include "yarn_llama.h"

namespace xft {
enum class GenerationMode { GREEDY_SEARCH, BEAM_SEARCH, SAMPLE };

GenerationMode getGenerationMode(SearcherConfig &config_) {
    if (config_.numBeams == 1) {
        if (config_.doSample) {
            return GenerationMode::SAMPLE;
        } else {
            return GenerationMode::GREEDY_SEARCH;
        }
    } else if (config_.numBeams > 1) {
        return GenerationMode::BEAM_SEARCH;
    } else {
        printf("numBeams should greater than or equal to 1.\n");
        exit(-1);
    }
}

Model::Model() : decoder(nullptr), searcher(nullptr), isNewInput(true) {
    Env::initVerbose();
    Env::initPipelineStage();
    Env::initEngineKindIndex();
    TimeLine::init();
}

Model::~Model() {
    exitSlaves();
    if (decoder != nullptr) { delete decoder; }
    if (searcher != nullptr) { delete searcher; }
}

void Model::exitSlaves() {
    if (decoder->getRank() == 0) {
        configuration.numBeams = 0;
        Messenger &messenger = decoder->getMessenger();
        messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));
    }
}

void Model::input(std::vector<int32_t> &inputIds_, int batchSize_) {
    isNewInput = true;
    Messenger &messenger = decoder->getMessenger();
    int dims[2];
    if (decoder->getRank() == 0) {
        dims[0] = batchSize_;
        dims[1] = inputIds_.size();
    }
    messenger.broadcast(dims, 2);
    batchSize = dims[0];
    seqLen = dims[1] / batchSize;

    inputIds.resize(dims[1]);
    if (decoder->getRank() == 0) { inputIds = inputIds_; }
    messenger.broadcast(inputIds.data(), dims[1]);
}

void Model::config(int maxLen_, int numBeams_, int numBeamHypsToKeep_, float lenPenalty_, bool doEarlyStopping_,
        int eosTokenId_, int padTokenId_, bool doSample_, float temperature_, int topK_, float topP_,
        float repetitionPenalty_, const std::vector<std::vector<int>> &stopWordsList_) {
    configuration.maxLen = maxLen_;
    configuration.numBeams = numBeams_;
    configuration.numBeamHypsToKeep = numBeamHypsToKeep_;
    configuration.lenPenalty = lenPenalty_;
    configuration.doEarlyStopping = doEarlyStopping_;
    configuration.eosTokenId = eosTokenId_;
    configuration.padTokenId = padTokenId_;
    configuration.doSample = doSample_;
    configuration.temperature = temperature_;
    configuration.topK = topK_;
    configuration.topP = topP_;
    configuration.repetitionPenalty = repetitionPenalty_;

    this->config(configuration, stopWordsList_);
}

void Model::config(SearcherConfig &config_, const std::vector<std::vector<int>> &stopWordsList_) {
    isNewInput = true;
    if (decoder->getRank() == 0) { configuration = config_; }
    Messenger &messenger = decoder->getMessenger();
    messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));

    // Slaves get exit flags and exit directly
    if (decoder->getRank() > 0 && configuration.numBeams == 0) { exit(0); }

    createSearcher(configuration);
    setStopWords(stopWordsList_);
}

bool Model::isDone() {
    if (searcher == nullptr || inputIds.empty()) {
        printf("Please set input and config first.\n");
        exit(-1);
    }
    return !isNewInput && searcher->isDone();
}

std::vector<int32_t> Model::generate() {
    if (inputIds.empty()) {
        printf("Please set input tokens by model.input().\n");
        exit(-1);
    }
    if (searcher == nullptr) {
        printf("Please set generation config by model.config().\n");
        exit(-1);
    }

    if (isNewInput) {
        isNewInput = false;
        return searcher->getNextToken(inputIds.data(), batchSize, inputIds.size() / batchSize);
    } else {
        return searcher->getNextToken();
    }
}

void Model::createSearcher(SearcherConfig &config_) {
    if (searcher != nullptr) { delete searcher; }

    GenerationMode genMode = getGenerationMode(config_);
    if (genMode == GenerationMode::GREEDY_SEARCH) {
        searcher = new GreedySearch(*decoder, config_);
    } else if (genMode == GenerationMode::BEAM_SEARCH) {
        searcher = new BeamSearch(*decoder, config_);
    } else if (genMode == GenerationMode::SAMPLE) {
        searcher = new SampleSearch(*decoder, config_);
    }
}

int Model::getRank() {
    return decoder->getRank();
}

void Model::setDecoder(AbstractDecoder *dec) {
    decoder = dec;
}

void Model::setPrefix(std::vector<int32_t> &prefixIDs_) {
    Messenger &messenger = decoder->getMessenger();
    int prefixSeqLen = prefixIDs_.size();

    messenger.broadcast(&prefixSeqLen, 1);

    std::vector<int32_t> perfixIDs;
    perfixIDs.resize(prefixSeqLen);
    if (decoder->getRank() == 0) { perfixIDs = prefixIDs_; }
    messenger.broadcast(perfixIDs.data(), prefixSeqLen);

    decoder->setPrefix(perfixIDs.data(), prefixSeqLen);
}

void Model::unsetPrefix() {
    decoder->unsetPrefix();
}

bool Model::setStopWords(std::vector<std::vector<int>> stopWordsList) {
    if (searcher == nullptr) {
        printf("[Warning] Fails to set stop words. Please config model first.");
        return false;
    }
    Messenger &messenger = decoder->getMessenger();

    // Remove empty words and words containing non-positive elements.
    if (decoder->getRank() == 0) {
        for (auto it = stopWordsList.rbegin(); it != stopWordsList.rend(); ++it) {
            if ((*it).empty()) {
                stopWordsList.erase(std::next(it).base());
                continue;
            }
            for (auto x : *it) {
                if (x <= 0) { stopWordsList.erase(std::next(it).base()); }
            }
        }
    }

    int listSize = stopWordsList.size();
    messenger.broadcast(&listSize, 1);
    // If stopWordsList is empty, stop broadcasting and return.
    if (listSize == 0) { return false; }

    vector<int> wordsSize(listSize);
    if (decoder->getRank() == 0) {
        for (int i = 0; i < listSize; i++) {
            wordsSize[i] = stopWordsList[i].size();
        }
    }
    messenger.broadcast(wordsSize.data(), listSize);

    int wordsDataLen = 0;
    for (auto x : wordsSize) {
        wordsDataLen += x;
    }

    // flatten to 1-D vector
    vector<int> wordsData(wordsDataLen);
    if (decoder->getRank() == 0) {
        int currentIndex = 0;
        for (const auto &words : stopWordsList) {
            std::copy(words.begin(), words.end(), wordsData.begin() + currentIndex);
            currentIndex += words.size();
        }
    }
    messenger.broadcast(wordsData.data(), wordsDataLen);

    if (decoder->getRank() == 0) {
        return searcher->setStopWords(stopWordsList);
    } else {
        // restore stop words list to 2-D vector
        std::vector<std::vector<int>> restoredList;
        int currentIndex = 0;
        for (int i = 0; i < wordsSize.size(); ++i) {
            int size = wordsSize[i];
            std::vector<int> subVector(wordsData.begin() + currentIndex, wordsData.begin() + currentIndex + size);
            currentIndex += size;
            restoredList.emplace_back(subVector);
        }

        return searcher->setStopWords(restoredList);
    }
}

AutoModel::AutoModel(std::string modelPath, xft::DataType datatype) : Model() {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);

    if (reader.ParseError() < 0) {
        printf("Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();

    if (modeltype == "gpt") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new OptDecoder<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new OptDecoder<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new OptDecoder<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new OptDecoder<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new OptDecoder<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new OptDecoder<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, int8_t>(modelPath));
                break;
            case xft::DataType::bf16_w8a8:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, w8a8_t>(modelPath));
                break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, nf4x2_t>(modelPath));
                break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<OptDecoder, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<OptDecoder, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<OptDecoder, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "llama") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new LlamaLLM<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new LlamaLLM<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new LlamaLLM<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new LlamaLLM<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new LlamaLLM<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new LlamaLLM<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<LlamaLLM, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<LlamaLLM, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<LlamaLLM, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<LlamaLLM, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<LlamaLLM, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "yarn_llama") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new YaRNLlama<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new YaRNLlama<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new YaRNLlama<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new YaRNLlama<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new YaRNLlama<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new YaRNLlama<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<YaRNLlama, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<YaRNLlama, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<YaRNLlama, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<YaRNLlama, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<YaRNLlama, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<YaRNLlama, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<YaRNLlama, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<YaRNLlama, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "baichuan") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new Baichuan<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new Baichuan<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new Baichuan<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new Baichuan<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new Baichuan<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new Baichuan<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<Baichuan, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<Baichuan, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<Baichuan, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<Baichuan, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<Baichuan, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<Baichuan, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<Baichuan, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<Baichuan, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new ChatGLM<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new ChatGLM<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new ChatGLM<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<ChatGLM, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<ChatGLM, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<ChatGLM, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<ChatGLM, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<ChatGLM, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<ChatGLM, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm2") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM2<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM2<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM2<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new ChatGLM2<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new ChatGLM2<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new ChatGLM2<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM2, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<ChatGLM2, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<ChatGLM2, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<ChatGLM2, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<ChatGLM2, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm3") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM3<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM3<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM3<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new ChatGLM3<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new ChatGLM3<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new ChatGLM3<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM3, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM3, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<ChatGLM3, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4:
                setDecoder(new HybridModel<ChatGLM3, bfloat16_t, uint4x2_t>(modelPath));
                break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<ChatGLM3, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<ChatGLM3, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<ChatGLM3, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<ChatGLM3, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "qwen") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new Qwen<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new Qwen<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new Qwen<int8_t>(modelPath)); break;
            case xft::DataType::w8a8: setDecoder(new Qwen<w8a8_t>(modelPath)); break;
            case xft::DataType::int4: setDecoder(new Qwen<uint4x2_t>(modelPath)); break;
            case xft::DataType::nf4: setDecoder(new Qwen<nf4x2_t>(modelPath)); break;
            case xft::DataType::bf16_fp16: setDecoder(new HybridModel<Qwen, bfloat16_t, float16_t>(modelPath)); break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<Qwen, bfloat16_t, int8_t>(modelPath)); break;
            case xft::DataType::bf16_w8a8: setDecoder(new HybridModel<Qwen, bfloat16_t, w8a8_t>(modelPath)); break;
            case xft::DataType::bf16_int4: setDecoder(new HybridModel<Qwen, bfloat16_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::bf16_nf4: setDecoder(new HybridModel<Qwen, bfloat16_t, nf4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_int8: setDecoder(new HybridModel<Qwen, w8a8_t, int8_t>(modelPath)); break;
            case xft::DataType::w8a8_int4: setDecoder(new HybridModel<Qwen, w8a8_t, uint4x2_t>(modelPath)); break;
            case xft::DataType::w8a8_nf4: setDecoder(new HybridModel<Qwen, w8a8_t, nf4x2_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else {
        printf("Unsupported data type.\n");
        exit(-1);
    }
}
} // namespace xft
