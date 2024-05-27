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
#include <filesystem>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <type_traits>

#include "INIReader.h"
#include "cmdline.h"
#include "sentencepiece_processor.h"
#include "timer.h"
#include "xfastertransformer.h"

extern const char *vocab_opt[];
extern const char *vocab_qwen[];

class TokenizerBase {
public:
    TokenizerBase() {}
    TokenizerBase(std::string &tokenPath) {
        std::filesystem::path filePath(tokenPath);

        if (!(std::filesystem::exists(filePath) && std::filesystem::is_regular_file(filePath))) {
            std::cout << "[ERROR] " << filePath << " isn't a file or not existed." << std::endl;
            exit(-1);
        }

        const auto status = processor.Load(tokenPath);
        if (!status.ok()) {
            std::cout << status.ToString() << std::endl;
            std::cout << "[ERROR] Fail to load tokenizer file." << std::endl;
            exit(-1);
        }
        vocabSize = processor.GetPieceSize();
    };

    virtual std::vector<int> encode(std::string &input) {
        std::vector<int> output;
        processor.Encode(input, &output);
        addSpecialTokenIds(output);
        return output;
    }

    void addSpecialTokenIds(std::vector<int> &input) {
        input.insert(input.begin(), prefixTokenIds.begin(), prefixTokenIds.end());
        input.insert(input.end(), suffixTokenIds.begin(), suffixTokenIds.end());
    }

    virtual std::string decode(std::vector<int> &ids) {
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }
    virtual std::string decode(int id) { return processor.IdToPiece(id); }

    void printResult(std::vector<int> &ids, int batchSize, int numBeams) {
        if (batchSize * numBeams > 2) {
            printf("[%d]%s [%d]%s ... [%d]%s\n", ids[0], decode(ids[0]).c_str(), ids[1], decode(ids[1]).c_str(),
                    ids[batchSize * numBeams - 1], decode(ids[batchSize * numBeams - 1]).c_str());
        } else if (batchSize * numBeams > 1) {
            printf("[%d]%s [%d]%s\n", ids[0], decode(ids[0]).c_str(), ids[batchSize * numBeams - 1],
                    decode(ids[batchSize * numBeams - 1]).c_str());
        } else {
            printf("[%d]%s ", ids[0], decode(ids[0]).c_str());
        }
    }

    std::vector<std::string> batchDecode(std::vector<int> &input, int batchSize) {
        int seqLen = input.size() / batchSize;
        std::vector<std::string> ret;
        for (int i = 0; i < batchSize; ++i) {
            std::vector<int> tokens(input.begin() + i * seqLen, input.begin() + (i + 1) * seqLen);
            ret.emplace_back(decode(tokens));
        }
        return ret;
    }

protected:
    std::vector<std::string> prefixTokens;
    std::vector<std::string> suffixTokens;
    std::vector<int> prefixTokenIds;
    std::vector<int> suffixTokenIds;

    sentencepiece::SentencePieceProcessor processor;
    int vocabSize;
};

class ChatGLMTokenizer : public TokenizerBase {
public:
    ChatGLMTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        suffixTokens = {"[gMASK]", "<sop>"};
        suffixTokenIds = {processor.PieceToId("[gMASK]"), processor.PieceToId("<sop>")};
    }
};

class ChatGLM2Tokenizer : public TokenizerBase {
public:
    ChatGLM2Tokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        // ChatGLM2's special tokens is not included in sentencepiece. ["[MASK]", "[gMASK]", "[sMASK]", "sop", "eop"]
        prefixTokens = {"[gMASK]", "sop"};
        prefixTokenIds = {vocabSize + 1, vocabSize + 3};
    }
    std::string decode(std::vector<int> &ids) override {
        ids.erase(std::remove_if(ids.begin(), ids.end(), [this](int value) { return value >= vocabSize; }), ids.end());
        std::string text;
        processor.Decode(ids, &text);
        return text;
    }

    std::string decode(int id) override {
        if (id > vocabSize) {
            return "";
        } else {
            return processor.IdToPiece(id);
        }
    }
};

class LlamaTokenizer : public TokenizerBase {
public:
    LlamaTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) { processor.SetEncodeExtraOptions("bos"); }
};

class BaichuanTokenizer : public TokenizerBase {
public:
    BaichuanTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        // 195: user_id 196: assistant_id
        prefixTokenIds = {195};
        suffixTokenIds = {196};
    }
};

class YaRNLlamaTokenizer : public TokenizerBase {
public:
    YaRNLlamaTokenizer(std::string &tokenPath) { vocabSize = 106963; }

    // TODO: Need to achieve actual encode function
    std::vector<int> encode(std::string &input) override {
        // only for Test
        return std::vector<int>(
                {7454, 2402, 257, 640, 11, 612, 11196, 257, 1310, 2576, 508, 8288, 284, 423, 17545, 13});
    }

    std::string decode(std::vector<int> &ids) override {
        if (ids.size() == 1) { return decode(ids[0]); }

        std::string text("");
        text.reserve(ids.size());

        for (int id : ids) {
            if (id < vocabSize) {
                if (vocab_list == nullptr)
                    text += "[" + std::to_string(id) + "] ";
                else
                    text += vocab_list[id];
            } else {
                text += "(null) ";
            }
        }

        return text;
    }
    std::string decode(int id) override {
        if (id < vocabSize) {
            return vocab_list[id];
        } else {
            return "(null)";
        }
    }

private:
    const char **vocab_list = nullptr;
};

class OptTokenizer : public TokenizerBase {
public:
    OptTokenizer(std::string &tokenPath) { vocabSize = 50265; }

    std::vector<int> encode(std::string &input) override {
        return std::vector<int>({2, 11475, 2115, 10, 86, 6, 89, 13412, 10, 410, 1816, 54, 6640, 7, 33, 18848, 4});
    }

    std::string decode(std::vector<int> &ids) override {
        if (ids.size() == 1) { return decode(ids[0]); }
        std::string text("");
        for (int id : ids) {
            if (id < vocabSize) {
                text += vocab_list[id];
                text += " ";
            } else {
                text += "(null) ";
            }
        }
        return text;
    }
    std::string decode(int id) override {
        if (id < vocabSize) {
            return vocab_list[id];
        } else {
            return "(null)";
        }
    }

private:
    const char **vocab_list = vocab_opt;
};

class QwenTokenizer : public TokenizerBase {
public:
    QwenTokenizer(std::string &tokenPath) { vocabSize = 151851; }

    // TODO: Need to achieve actual encode function
    std::vector<int> encode(std::string &input) override {
        // only for Test
        return std::vector<int>(
                {12522, 5193, 264, 882, 11, 1052, 24295, 264, 2632, 3743, 879, 14915, 311, 614, 30978, 13});
    }

    std::string decode(std::vector<int> &ids) override {
        if (ids.size() == 1) { return decode(ids[0]); }

        std::string text("");
        text.reserve(ids.size());

        for (int id : ids) {
            if (id < vocabSize) {
                text += vocab_list[id];
            } else {
                text += "(null) ";
            }
        }

        return text;
    }
    std::string decode(int id) override {
        if (id < vocabSize) {
            return vocab_list[id];
        } else {
            return "(null)";
        }
    }

private:
    const char **vocab_list = vocab_qwen;
};

class GemmaTokenizer : public TokenizerBase {
public:
    GemmaTokenizer(std::string &tokenPath) : TokenizerBase(tokenPath) {
        vocabSize = 256000;
        prefixTokenIds = {2, 2, 106, 1645, 108};
        suffixTokenIds = {107, 108, 106, 2516, 108};
    }
};

TokenizerBase *getTokenizer(std::string &modeltype, std::string &tokenPath) {
    if (modeltype == "gpt") {
        return new OptTokenizer(tokenPath);
    } else if (modeltype == "llama") {
        return new LlamaTokenizer(tokenPath);
    } else if (modeltype == "yarn_llama") {
        return new YaRNLlamaTokenizer(tokenPath);
    } else if (modeltype == "baichuan") {
        return new BaichuanTokenizer(tokenPath);
    } else if (modeltype == "chatglm") {
        return new ChatGLMTokenizer(tokenPath);
    } else if (modeltype == "chatglm2" or modeltype == "chatglm3") {
        return new ChatGLM2Tokenizer(tokenPath);
    } else if (modeltype == "qwen") {
        return new QwenTokenizer(tokenPath);
    } else if (modeltype == "gemma") {
        return new GemmaTokenizer(tokenPath);
    } else {
        std::cout << "[Error] Token list of loaded model is unsupported yet.\n" << std::endl;
        exit(-1);
    }
}

std::map<std::string, xft::DataType> dataTypeMap = {{"fp16", xft::DataType::fp16}, {"bf16", xft::DataType::bf16},
        {"int8", xft::DataType::int8}, {"w8a8", xft::DataType::w8a8}, {"int4", xft::DataType::int4},
        {"nf4", xft::DataType::nf4}, {"bf16_fp16", xft::DataType::bf16_fp16}, {"bf16_int8", xft::DataType::bf16_int8},
        {"bf16_w8a8", xft::DataType::bf16_w8a8}, {"bf16_int4", xft::DataType::bf16_int4},
        {"bf16_nf4", xft::DataType::bf16_nf4}, {"w8a8_int8", xft::DataType::w8a8_int8},
        {"w8a8_int4", xft::DataType::w8a8_int4}, {"w8a8_nf4", xft::DataType::w8a8_nf4}};

std::map<std::string, xft::DataType> kvCacheDataTypeMap
        = {{"fp16", xft::DataType::fp16}, {"int8", xft::DataType::int8}};

std::string getModelType(std::string &modelPath) {
    std::string configPath = modelPath + "/config.ini";
    INIReader reader = INIReader(configPath);
    if (reader.ParseError() < 0) {
        printf("[Error] Could not load model config.ini.\n");
        exit(-1);
    }
    std::string modeltype = *reader.Sections().begin();
    return modeltype;
}

int main(int argc, char **argv) {
    cmdline::parser args;

    args.add<std::string>("model", 'm', "path of xft format model", true);
    args.add<std::string>("token", 't', "path of tokenizer", true);
    args.add<std::string>("input", 'i', "input prompt.", false,
            "Once upon a time, there existed a little girl who liked to have adventures.");
    args.add<std::string>("dtype", 'd', "weight data type", false, "fp16");
    args.add<std::string>("kv_cache_dtype", '\0', "kv cache data type", false, "fp16");

    args.parse_check(argc, argv);

    std::string modelPath = args.get<std::string>("model");
    std::string tokenPath = args.get<std::string>("token");

    std::string dtype_name = args.get<std::string>("dtype");
    xft::DataType dtype = xft::DataType::fp16;
    std::string kv_cache_dtype_name = args.get<std::string>("kv_cache_dtype");
    xft::DataType kvCacheDataType = xft::DataType::fp16;

    // Check data type
    auto it = dataTypeMap.find(dtype_name);
    if (it != dataTypeMap.end()) {
        dtype = it->second;
    } else {
        std::cout << "[Error] Unsupport dtype index: " << dtype_name << std::endl;
        return 0;
    }

    it = kvCacheDataTypeMap.find(kv_cache_dtype_name);
    if (it != kvCacheDataTypeMap.end()) {
        kvCacheDataType = it->second;
    } else {
        std::cout << "[Error] Unsupport KV cache dtype index: " << kv_cache_dtype_name << std::endl;
        return 0;
    }

    std::string modeltype = getModelType(modelPath);

    auto *tokenizer = getTokenizer(modeltype, tokenPath);
    std::string inputPrompt = args.get<std::string>("input");
    std::vector<int> input = tokenizer->encode(inputPrompt);

    xft::AutoModel model(modelPath, dtype, kvCacheDataType);
    bool isMaster = model.isMaster();

    if (isMaster) {
        std::cout << "[INFO] Model path is " << modelPath << std::endl;
        std::cout << "[INFO] Token path is " << tokenPath << std::endl;
        std::cout << "[INFO] Data type is " << dtype_name << std::endl;
        std::cout << "[INFO] KV cache data type is " << kv_cache_dtype_name << std::endl;
        std::cout << "[INFO] Input prompt: " << inputPrompt << std::endl;
        std::cout << "[INFO] Input Token Ids: ";
        for (auto x : input) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    SearcherConfig config;
    config.maxLen = 128;
    std::vector<std::vector<int>> generatedTokens(3);
    int seqIDs[3];
    std::vector<std::vector<int>> inputIDs;
    std::vector<int> seqs;

    // 1st sequence: generate some tokens
    inputIDs = {input};
    auto ret = model.set_input(inputIDs, seqs, config);
    seqIDs[0] = ret[0];
    ret = model.generate(); // 1st token
    for (auto id : ret) {
        generatedTokens[0].emplace_back(id);
    }

    for (int i = 0; i < 2; ++i) { // some next tokens
        inputIDs = {{generatedTokens[0].at(generatedTokens[0].size() - 1)}};
        model.set_input(inputIDs, {seqIDs[0]}, config);
        auto ret = model.generate();
        for (auto id : ret) {
            generatedTokens[0].emplace_back(id);
        }
    }

    // 2nd sequence: first token generation
    inputIDs = {input};
    seqs.clear();
    ret = model.set_input(inputIDs, seqs, config);
    seqIDs[1] = ret[0];
    ret = model.generate();
    for (auto id : ret) {
        generatedTokens[1].emplace_back(id);
    }

    // Batching together to generate some tokens for both sequences
    for (int i = 0; i < 2; ++i) {
        inputIDs = {{generatedTokens[0].at(generatedTokens[0].size() - 1)},
                {generatedTokens[1].at(generatedTokens[1].size() - 1)}};
        model.set_input(inputIDs, {seqIDs[0], seqIDs[1]}, config);
        auto ret = model.generate();
        assert(ret.size() == 2);
        for (int j = 0; j < 2; ++j) {
            generatedTokens[j].emplace_back(ret[j]);
        }
    }

    // 3rd sequence: first token generation
    inputIDs = {input};
    seqs.clear();
    ret = model.set_input(inputIDs, seqs, config);
    seqIDs[2] = ret[0];
    ret = model.generate();
    for (auto id : ret) {
        generatedTokens[2].emplace_back(id);
    }

    // Batching together to generate some tokens for 3 sequences
    for (int i = 0; i < 2; ++i) {
        inputIDs = {{generatedTokens[0].at(generatedTokens[0].size() - 1)},
                {generatedTokens[1].at(generatedTokens[1].size() - 1)},
                {generatedTokens[2].at(generatedTokens[2].size() - 1)}};
        model.set_input(inputIDs, {seqIDs[0], seqIDs[1], seqIDs[2]}, config);
        auto ret = model.generate();
        assert(ret.size() == 3);
        for (int j = 0; j < 3; ++j) {
            generatedTokens[j].emplace_back(ret[j]);
        }
    }

    // Suppose sequence 0 finished
    for (int i = 0; i < 2; ++i) {
        inputIDs = {{generatedTokens[1].at(generatedTokens[1].size() - 1)},
                {generatedTokens[2].at(generatedTokens[2].size() - 1)}};
        model.set_input(inputIDs, {seqIDs[1], seqIDs[2]}, config);
        auto ret = model.generate();
        assert(ret.size() == 2);
        for (int j = 0; j < 2; ++j) {
            generatedTokens[j + 1].emplace_back(ret[j]);
        }
    }

    // Print out values inside generatedTokens
    for (int i = 0; i < 3; ++i) {
        std::cout << "Generated Tokens [" << i << "]: ";
        for (auto id : generatedTokens[i]) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::vector<std::string> strs = tokenizer->batchDecode(generatedTokens[i], 1);
        std::cout << strs[0] << std::endl;
    }

    return 0;
}
