#include "beam_search.h"

#include <ctime>
#include <iostream>

#include "beam_search_test.h"
#include "opt_decoder.h"
#include "gtest/gtest.h"

TEST(BeamSearchTest, BeamHypothesesTest) {
    BeamSearchScorerTest beamSearchTester;
    beamSearchTester.checkBeamHypotheses();
}

TEST(BeamSearchTest, beamScorerUpdateTest) {
    BeamSearchScorerTest beamSearchTester;
    beamSearchTester.checkBeamScorerUpdate();
}

TEST(BeamSearchTest, BeamScorerFinalizeTest) {
    BeamSearchScorerTest beamSearchTester;
    beamSearchTester.checkBeamScoresFinalize();
}

TEST(BeamSearchTest, BeamSearchTest) {
    const char *modelPath = "/data/1-gpu";
    OptDecoder<float16_t> decoder(modelPath);
    const int batchSize = 2;
    const int beamSize = 3;

    BeamSearch beam_searcher(decoder, beamSize, batchSize, /*maxLen*/ 100);

    // Initial input
    const int seqLen = 22;
    int samples[batchSize * beamSize][seqLen] = {
            {2, 100, 17, 27, 119, 10080, 10288, 6, 38, 17, 27, 119, 2602, 14, 38, 17, 27, 548, 56, 5, 945, 7},
            {2, 4993, 41, 1946, 9, 1431, 6, 9469, 6, 5, 5385, 9, 5, 446, 9, 7395, 174, 1865, 5, 80, 2380, 2442},
    };

    // Subsequent input IDs, dumped from transformers solution
    int inputIds[][6] = {
            {28, 1807, 3594, 444, 44, 22},
            {10, 5, 5, 4102, 48, 5525},
            {233, 82, 82, 15, 4, 8},
            {9, 9, 9, 50118, 5, 10},
            {42, 5, 402, 50118, 346, 696},
    };

    // Beam indices dumped from transformers solution
    int beamIndices[][6] = {
            {0, 0, 0, 3, 3, 3},
            {0, 2, 1, 3, 4, 5},
            {0, 2, 1, 3, 3, 3},
            {0, 1, 2, 4, 3, 3},
    };

    // Logits output dumped from transformers solution
    float expectedLogits[][6] = {
            {-1.9482, -2.1622, -2.2831, -0.7081, -1.9935, -2.2053},
            {-3.2823, -3.7191, -3.9653, -0.7214, -2.0227, -3.1980},
            {-3.6371, -4.9550, -5.9593, -1.6068, -1.9373, -3.0038},
            {-3.6461, -5.0756, -6.1114, -2.5195, -3.2698, -3.5165},
    };

    // generate first token
    auto nextTokens = beam_searcher.getNextToken((int *)samples, batchSize, seqLen);
    auto nextScores = beam_searcher.getNextScores();
    auto nextIndices = beam_searcher.getNextIndices();

    if (decoder.getRank() == 0) {
        for (int i = 0; i < batchSize * beamSize; ++i) {
            EXPECT_EQ(nextTokens[i], inputIds[0][i]);
        }

        for (int i = 0; i < batchSize * beamSize; ++i) {
            EXPECT_NEAR(nextScores[i], expectedLogits[0][i], 0.0002);
        }

        for (int i = 0; i < batchSize * beamSize; ++i) {
            EXPECT_EQ(nextIndices[i], beamIndices[0][i]);
        }
    }

    for (int times = 1; times < 4; ++times) {
        nextTokens = beam_searcher.getNextToken();
        nextScores = beam_searcher.getNextScores();
        nextIndices = beam_searcher.getNextIndices();

        if (decoder.getRank() == 0) {
            for (int i = 0; i < batchSize * beamSize; ++i) {
                EXPECT_EQ(nextTokens[i], inputIds[times][i]);
            }

            for (int i = 0; i < batchSize * beamSize; ++i) {
                EXPECT_NEAR(nextScores[i], expectedLogits[times][i], 0.001);
            }

            for (int i = 0; i < batchSize * beamSize; ++i) {
                EXPECT_EQ(nextIndices[i], beamIndices[times][i]);
            }
        }
    }
}

int main(int argc, char **argv) {
    // Use system clock for seed
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
