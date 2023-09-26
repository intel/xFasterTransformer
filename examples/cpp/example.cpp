#include <iostream>
#include <string>
#include <vector>

#include "timer.h"
#include "xfastertransformer.h"

// For OPT model
extern const char *vocab[];
const int max_size = 50265;
const char **vocab_list = vocab;

// For Llama & Llama2 model
// extern const char *vocab_llama[];
// const int max_size = 32000;
// const char **vocab_list = vocab_llama;

// For ChatGLM model
// extern const char *vocab_chatglm[];
// const int max_size = 130344;
// const char **vocab_list = vocab_chatglm;

// For ChatGLM2 model
// extern const char *vocab_chatglm2[];
// const int max_size = 64791;
// const char **vocab_list = vocab_chatglm2;

static const char *getWord(int32_t id) {
    // const int size = 50265;
    if (id < max_size) {
        return vocab_list[id];
    } else {
        return "(null)";
    }
}

static void printResult(std::vector<int32_t> &ids, int batchSize) {
    if (batchSize > 2) {
        printf("[%d]%s [%d]%s ... [%d]%s\n", ids[0], getWord(ids[0]), ids[1], getWord(ids[1]), ids[batchSize - 1],
                getWord(ids[batchSize - 1]));
    } else if (batchSize > 1) {
        printf("[%d]%s [%d]%s\n", ids[0], getWord(ids[0]), ids[batchSize - 1], getWord(ids[batchSize - 1]));
    } else {
        printf("[%d]%s ", ids[0], getWord(ids[0]));
    }
}

int main(int argc, char **argv) {
    printf("Usage: %s [model_path] [weight_type:FP16(0),BF16(1),INT8(2)] [input_token_size]\n", argv[0]);

    std::string modelPath = "/data/1-gpu";
    if (argc > 1) { modelPath = argv[1]; }
    std::cout << "Model path is " << modelPath << std::endl;

    xft::DataType dtype = xft::DataType::fp16;
    if (argc > 2) {
        switch (atoi(argv[2])) {
            case 0:
                dtype = xft::DataType::fp16;
                std::cout << "Data type is float16." << std::endl;
                break;
            case 1:
                dtype = xft::DataType::bf16;
                std::cout << "Data type is bfloat16." << std::endl;
                break;
            case 2:
                std::cout << "Data type is int8." << std::endl;
                dtype = xft::DataType::int8;
                break;
            default: std::cout << "Unsupport dtype index: " << argv[2] << std::endl; return 0;
        }
    } else {
        std::cout << "Use default data type float16." << std::endl;
    }

    int inputSize = -1;
    if (argc > 3) { inputSize = atoi(argv[3]); }

    xft::AutoModel model(modelPath, dtype);
    bool isMaster = (model.getRank() == 0);

    std::vector<int> input(
            {2, 100, 17, 27, 119, 10080, 10288, 6, 38, 17, 27, 119, 2602, 14, 38, 17, 27, 548, 56, 5, 945, 7});

    // Need longer prompt
    if (inputSize > 0 && inputSize > input.size()) {
        input.reserve(inputSize);
        std::vector<int> fakeTokens(inputSize - input.size(), 100);
        input.insert(input.begin() + 2, fakeTokens.begin(), fakeTokens.end());
    } else if (inputSize > 0) {
        printf("Do not support token size of %d, use %d instead.\n", inputSize, input.size());
    }

    /*
    'I’m humbled, I’m proud that I’ve had the opportunity to'
    // OPT
    2, 100, 17, 27, 119, 10080, 10288, 6, 38, 17, 27, 119, 2602, 14, 38, 17, 27,
    548, 56, 5, 945, 7
    // LLama
    1, 306, 30010, 29885, 3165, 27225, 29892, 306, 30010, 29885, 22314, 393,
    306, 30010, 345, 750, 278, 15130, 304
    // ChatGLM
    115, 30, 143, 57240, 6, 115, 30, 143, 5111, 109, 115, 30, 261, 171, 100,
    1763, 103, 130001, 130004
    // ChatGLM2
    64790, 64792,   307, 30963, 30924,  1437, 26344, 30932,   307, 30963,
    30924,  5594,   343,   307, 30963,   318,   599,   267,  2973,   289

     'Once upon a time, there existed a little girl who liked to have
    adventures'
    //  Opt
    2, 11475, 2115, 10, 86, 6, 89, 13412, 10, 410, 1816, 54, 6640, 7, 33, 18848,
    4
    // LLama
    1, 9038, 2501, 263, 931, 29892, 727, 22856, 263, 2217, 7826, 1058, 23289,
    304, 505, 17623, 1973, 29889
    // ChatGLM
    3393, 955, 104, 163, 6, 173, 9166, 104, 486, 2511, 172, 7599, 103, 127,
    17163, 7, 130001, 130004
    // ChatGLM2
    64790, 64792,  4155,  2488,   260,   622, 30932,   627, 13519,   260,
    1332,  2689,   554,  7364,   289,   431, 15672
    */

    const int loop = 10;
    for (int i = 0; i < loop; ++i) {
        model.config(input.size() + 100, 1);
        model.input(input, 1);

        {
            Timer t(isMaster, "Fisrt token");
            auto nextIds = model.generate();
            if (isMaster) { printResult(nextIds, 1); }
        }

        if (!model.isDone()) {
            Timer t(isMaster, "Second token");
            auto nextIds = model.generate();
            if (isMaster) { printResult(nextIds, 1); }
        }

        while (!model.isDone()) {
            auto nextIds = model.generate();
            if (isMaster) { printResult(nextIds, 1); }
        }
        auto result = model.finalize();

        if (isMaster) {
            std::cout << "\nFinalzie output is:" << std::endl;
            for (auto x : result) {
                std::cout << "[" << x << "]" << getWord(x) << " ";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
