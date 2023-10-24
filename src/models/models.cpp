#include "models.h"

#include <string.h>

#include <stdexcept>

#include "INIReader.h"
#include "chatglm.h"
#include "chatglm2.h"
#include "hybrid_model.h"
#include "llama.h"
#include "opt_decoder.h"
#include "searcher.h"

namespace xft {
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
        int eosTokenId_, int padTokenId_) {
    isNewInput = true;
    if (decoder->getRank() == 0) {
        configuration.maxLen = maxLen_;
        configuration.numBeams = numBeams_;
        configuration.numBeamHypsToKeep = numBeamHypsToKeep_;
        configuration.lenPenalty = lenPenalty_;
        configuration.doEarlyStopping = doEarlyStopping_;
        configuration.eosTokenId = eosTokenId_;
        configuration.padTokenId = padTokenId_;
    }
    Messenger &messenger = decoder->getMessenger();
    messenger.broadcast((int *)&configuration, sizeof(SearcherConfig) / sizeof(int));

    // Slaves get exit flags and exit directly
    if (decoder->getRank() > 0 && configuration.numBeams == 0) {
        exit(0);
    }

    createSearcher(configuration);
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
    if (config_.numBeams == 1) {
        searcher = new GreedySearch(*decoder, config_);
    } else if (config_.numBeams > 1) {
        searcher = new BeamSearch(*decoder, config_);
    }
}

int Model::getRank() {
    return decoder->getRank();
}

void Model::setDecoder(AbstractDecoder *dec) {
    decoder = dec;
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
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8:
                setDecoder(new HybridModel<OptDecoder, bfloat16_t, int8_t>(modelPath));
                break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "llama") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new LlamaLLM<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new LlamaLLM<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new LlamaLLM<int8_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<LlamaLLM, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<LlamaLLM, bfloat16_t, int8_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM<float16_t>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM<bfloat16_t>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM<int8_t>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM, bfloat16_t, int8_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else if (modeltype == "chatglm2") {
        switch (datatype) {
            case xft::DataType::fp16: setDecoder(new ChatGLM2<float16_t, RmsNorm>(modelPath)); break;
            case xft::DataType::bf16: setDecoder(new ChatGLM2<bfloat16_t, RmsNorm>(modelPath)); break;
            case xft::DataType::int8: setDecoder(new ChatGLM2<int8_t, RmsNorm>(modelPath)); break;
            case xft::DataType::bf16_fp16:
                setDecoder(new HybridModel<ChatGLM2, bfloat16_t, float16_t>(modelPath));
                break;
            case xft::DataType::bf16_int8: setDecoder(new HybridModel<ChatGLM2, bfloat16_t, int8_t>(modelPath)); break;
            default: printf("Unsupported data type.\n"); exit(-1);
        }
    } else {
        printf("Unsupported data type.\n");
        exit(-1);
    }
}
} // namespace xft
