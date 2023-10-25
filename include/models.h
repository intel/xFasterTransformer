#pragma once

#include <iostream>
#include <vector>

#include "abstract_decoder.h"
#include "abstract_searcher.h"
#include "dtype.h"

namespace xft {
class Model {
public:
    Model() : decoder(nullptr), searcher(nullptr), isNewInput(true) {}
    ~Model();

    void input(std::vector<int32_t> &inputIds_, int batchSize_);

    void config(int maxLen_ = -1, int numBeams_ = 1, int numBeamHypsToKeep_ = 1, float lenPenalty_ = 1.0,
            bool doEarlyStopping_ = false, int eosTokenId_ = -1, int padTokenId_ = -1, bool doSample_ = false,
            float temperature_ = 1.0, float topK_ = 50, float topP_ = 1.0);

    bool isDone();

    std::vector<int32_t> generate();

    void createSearcher(SearcherConfig &config_);

    int getRank();

    int getBatchSize() { return batchSize; }

    int getSeqLen() { return seqLen; }

    SearcherConfig getConfig() { return configuration; }

    void setDecoder(AbstractDecoder *dec);

    std::vector<int32_t> finalize() { return searcher->finalize(); }

private:
    AbstractDecoder *decoder;
    AbstractSearcher *searcher;
    std::vector<int32_t> inputIds;
    int batchSize;
    int seqLen;
    SearcherConfig configuration;
    bool isNewInput;
};

class AutoModel : public Model {
public:
    AutoModel(std::string modelPath, xft::DataType datatype);
};
} // namespace xft