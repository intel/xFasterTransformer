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

TokenizerBase *getTokenizer(std::string &modeltype, std::string &tokenPath) {
    if (modeltype == "gpt") {
        return new OptTokenizer(tokenPath);
    } else if (modeltype == "llama") {
        return new LlamaTokenizer(tokenPath);
    } else if (modeltype == "chatglm") {
        return new ChatGLMTokenizer(tokenPath);
    } else if (modeltype == "chatglm2") {
        return new ChatGLM2Tokenizer(tokenPath);
    } else {
        std::cout << "[Error] Token list of loaded model is unsupported yet.\n" << std::endl;
        exit(-1);
    }
}

std::map<std::string, xft::DataType> dataTypeMap
        = {{"fp16", xft::DataType::fp16}, {"bf16", xft::DataType::bf16}, {"int8", xft::DataType::int8},
                {"bf16_fp16", xft::DataType::bf16_fp16}, {"bf16_int8", xft::DataType::bf16_int8}};

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
    args.add<std::string>("input", 'i', "input prompt, invalid for Opt model.", false,
            "Once upon a time, there existed a little girl who liked to have adventures.");
    args.add<std::string>("dtype", 'd', "weight data type", false, "fp16",
            cmdline::oneof<std::string>("fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"));
    args.add<int>("input_len", 'l', "input token size", false, -1);
    args.add<int>("output_len", '\0', "max tokens can generate excluded input.", false, 100, cmdline::range(1, 4096));
    args.add<int>("num_beams", 'n', "number of beam size.", false, 1, cmdline::range(1, 32));
    args.add<int>("batch_size", 'b', "batch size.", false, 1, cmdline::range(1, 32));
    args.add<int>("loop", '\0', "number of loop.", false, 10);
    args.add<int>("topK", '\0', "number of highest probability tokens to keep for top-k-filtering.", false, 50);
    args.add<float>("temperature", '\0', "value used to modulate the next token probabilities.", false, 1.0);
    args.add<float>("topP", '\0', "retain minimal tokens above topP threshold.", false, 1.0);
    args.add("no_stream", '\0', "disable streaming output");
    args.add("do_sample", '\0', "use sampling");
    args.parse_check(argc, argv);

    std::string modelPath = args.get<std::string>("model");
    std::string tokenPath = args.get<std::string>("token");

    bool streamingOutput = !args.exist("no_stream");
    bool doSample = args.exist("do_sample");

    std::string dtype_name = args.get<std::string>("dtype");
    xft::DataType dtype = xft::DataType::fp16;

    auto it = dataTypeMap.find(dtype_name);
    if (it != dataTypeMap.end()) {
        dtype = it->second;
    } else {
        std::cout << "[Error] Unsupport dtype index: " << dtype_name << std::endl;
        return 0;
    }

    int inputSize = args.get<int>("input_len");
    int outputLen = args.get<int>("output_len");
    int numBeams = args.get<int>("num_beams");
    int batchSize = args.get<int>("batch_size");
    int loop = args.get<int>("loop");
    int topK = args.get<int>("topK");
    float temperature = args.get<float>("temperature");
    float topP = args.get<float>("topP");

    std::string modeltype = getModelType(modelPath);

    auto *tokenizer = getTokenizer(modeltype, tokenPath);
    // std::string inputPrompt("Once upon a time, there existed a little girl who liked to have adventures.");
    std::string inputPrompt = args.get<std::string>("input");
    std::vector<int> input = tokenizer->encode(inputPrompt);

    xft::AutoModel model(modelPath, dtype);
    bool isMaster = (model.getRank() == 0);

    // Need longer prompt
    if (inputSize > 0 && inputSize > input.size()) {
        input.reserve(inputSize);
        std::vector<int> fakeTokens(inputSize - input.size(), input[2]);
        input.insert(input.begin() + 2, fakeTokens.begin(), fakeTokens.end());
    } else if (inputSize > 0) {
        printf("[Warning] Do not support token size of %d, use %ld instead.\n", inputSize, input.size());
    }
    int maxLen = input.size() + outputLen;

    if (batchSize > 1) {
        int len = input.size();
        input.resize(len * batchSize);
        for (int i = 1; i < batchSize; i++) {
            std::copy(input.begin(), input.begin() + len, input.begin() + i * len);
        }
    }

    if (isMaster) {
        std::cout << "[INFO] Model path is " << modelPath << std::endl;
        std::cout << "[INFO] Token path is " << tokenPath << std::endl;
        std::cout << "[INFO] Data type is " << dtype_name << std::endl;
        std::cout << "[INFO] inputSize is " << inputSize << std::endl;
        std::cout << "[INFO] outputLen is " << outputLen << std::endl;
        std::cout << "[INFO] num_beams is " << numBeams << std::endl;
        std::cout << "[INFO] do_samlpe is " << std::boolalpha << doSample << std::endl;
        std::cout << "[INFO] temperature is " << temperature << std::endl;
        std::cout << "[INFO] topK is " << topK << std::endl;
        std::cout << "[INFO] topP is " << topP << std::endl;
        std::cout << "[INFO] batch_size is " << batchSize << std::endl;
        std::cout << "[INFO] loop is " << loop << std::endl;
        std::cout << "[INFO] Input prompt is :" << inputPrompt << std::endl;
        std::cout << "[INFO] Input Token Ids is :";
        for (auto x : input) {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    for (int i = 0; i < loop; ++i) {
        // model.config(maxLen, numBeams);
        model.config(/*maxLen*/ maxLen, /*numBeams*/ numBeams, /*numBeamHypsToKeep*/ 1, /*lenPenalty*/ 1.0,
                /*doEarlyStopping*/ false, /*eosTokenId*/ -1, /*padTokenId*/ -1,
                /*doSample*/ doSample, /*temperature*/ temperature,
                /*topK*/ topK, /*topP*/ topP);
        model.input(input, batchSize);

        std::vector<int> firstIds;
        std::vector<int> seconedIds;

        if (!model.isDone()) {
            Timer t(isMaster, "[INFO] Fisrt token");
            firstIds = model.generate();
        }

        if (!model.isDone()) {
            Timer t(isMaster, "[INFO] Second token");
            seconedIds = model.generate();
        }

        if (isMaster && streamingOutput) {
            if (!firstIds.empty()) {
                tokenizer->printResult(firstIds, batchSize, numBeams);
                if (!seconedIds.empty()) { tokenizer->printResult(seconedIds, batchSize, numBeams); }
            }
        }

        while (!model.isDone()) {
            auto nextIds = model.generate();
            if (isMaster && streamingOutput) { tokenizer->printResult(nextIds, batchSize, numBeams); }
        }
        auto result = model.finalize();

        if (isMaster) {
            std::cout << "\n[INFO] Finalzie output is:" << std::endl;
            std::vector<std::string> sent = tokenizer->batchDecode(result, batchSize);
            for (auto str : sent) {
                std::cout << "==============================================" << std::endl;
                std::cout << str << std::endl;
            }
        }
    }

    return 0;
}
