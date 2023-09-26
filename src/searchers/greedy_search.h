#pragma once
#include "abstract_decoder.h"
#include "abstract_searcher.h"
#include "timeline.h"

class GreedySearch : public AbstractSearcher {
public:
    GreedySearch(AbstractDecoder &dec, const SearcherConfig &config);

    // Get next tokens accoring to the prompt IDs
    std::vector<int> getNextToken(int *ids, int batchSize, int seqLen);

    // Get next tokens according to previous predicted ID
    std::vector<int> getNextToken();

    bool isDone();

    std::vector<int32_t> finalize() {
        TimeLine t("dump_file");
        t.dump_file("timeline.json");
        return output;
    }

private:
    std::vector<int> search(std::tuple<float *, int, int> &result);

    AbstractDecoder &decoder;

    // Predicted token IDs
    std::vector<int> nextTokens;
    std::vector<int> output;
    std::vector<bool> doneBatch;

    int batchSize;
    int step;
    int curLen;
    int maxLen;
    int eosTokenId;
    int padTokenId;
};