#pragma once

#include <numeric>

#include "beam_search.h"
#include "gtest/gtest.h"

class BeamSearchScorerTest {
public:
    int batchSize;
    int seqLen;
    int vacabSize;
    int padTokenId;
    int eosTokenId;
    int maxLen;
    int numBeams;
    float lenPenalty;
    bool doEarlyStopping;
    int numBeamHypsToKeep;

    std::vector<int32_t> inputIds;
    std::vector<int32_t> nextTokens;
    std::vector<int32_t> nextIndices;
    std::vector<float> nextScores;

    BeamSearchScorerTest(int batchSize = 3, int seqLen = 10, int vacabSize = 999, int padTokenId = 0, int maxLen = 20,
            int numBeams = 4, float lenPenalty = 2.0, bool doEarlyStopping = true, int numBeamHypsToKeep = 2)
        : batchSize(batchSize)
        , seqLen(seqLen)
        , vacabSize(vacabSize)
        , padTokenId(padTokenId)
        , eosTokenId(vacabSize + 1)
        , maxLen(maxLen)
        , numBeams(numBeams)
        , lenPenalty(lenPenalty)
        , doEarlyStopping(doEarlyStopping)
        , numBeamHypsToKeep(numBeamHypsToKeep)
        , inputIds(batchSize * numBeams * seqLen)
        , nextTokens(batchSize * 2 * numBeams, 1)
        , nextIndices(batchSize * 2 * numBeams)
        , nextScores(batchSize * 2 * numBeams) {
        assert(batchSize * numBeams * seqLen < vacabSize);
        std::iota(inputIds.begin(), inputIds.end(), 1);
        std::iota(nextTokens.begin(), nextTokens.end(), 1);
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            for (int beamId = 0; beamId < numBeams; ++beamId) {
                float scores = (float)beamId / (2 * numBeams);
                nextScores[batchId * 2 * numBeams + beamId] = scores;
                nextIndices[batchId * 2 * numBeams + beamId] = beamId;
                nextScores[batchId * 2 * numBeams + beamId + numBeams] = scores;
                nextIndices[batchId * 2 * numBeams + beamId + numBeams] = beamId;
            }
        }
    }

    //   bool checkBeamHypotheses(std::vector<int32_t> inputIds) {
    void checkBeamHypotheses() {
        // check that correct number of beam hypotheses is set in beam scorer
        BeamSearchScorer beamScorer(batchSize, maxLen, numBeams, lenPenalty,
                /*doEarlyStopping*/ false, numBeamHypsToKeep);
        BeamHypotheses &beamHyp = beamScorer.beamHyps[0];
        EXPECT_EQ(beamScorer.beamHyps.size(), batchSize);

        // check correct type
        EXPECT_EQ(typeid(beamHyp), typeid(BeamHypotheses));

        // check that numBeams is correctly set
        EXPECT_EQ(beamScorer.numBeams, numBeams);

        // add `numBeams + 1` beams to change `worstScore`
        for (int beamIdx = 0; beamIdx < numBeams + 1; beamIdx++) {
            beamHyp.add(std::vector<int32_t>(
                                inputIds.begin() + beamIdx * seqLen, inputIds.begin() + (beamIdx + 1) * seqLen),
                    -10.0 + float(beamIdx));
        }

        // -10.0 is removed => -9.0 is worst score
        EXPECT_FLOAT_EQ(beamHyp.worstScore, -9.0 / pow(seqLen, lenPenalty));

        // -5.0 is better than worst score => should not be finished
        EXPECT_FALSE(beamHyp.isDone(-5.0, seqLen));

        // -20.0 is worse than worst score => should be finished
        EXPECT_TRUE(beamHyp.isDone(-20.0, seqLen));

        // re-init
        beamScorer = BeamSearchScorer(batchSize, maxLen, numBeams, lenPenalty,
                /*doEarlyStopping*/ true, numBeamHypsToKeep);
        // check for early stopping deactivated
        // if early stopping True -> score does not matter
        for (int beamIdx = 0; beamIdx < numBeams; beamIdx++) {
            beamScorer.beamHyps[0].add(std::vector<int32_t>(inputIds.begin() + beamIdx * seqLen,
                                               inputIds.begin() + (beamIdx + 1) * seqLen),
                    -10.0);
        }
        EXPECT_TRUE(beamScorer.beamHyps[0].isDone(-10.0, 5));
    }

    void checkBeamScorerUpdate() {
        // check too many eos tokens
        BeamSearchScorer beamScorer(batchSize, maxLen, numBeams, lenPenalty, doEarlyStopping, numBeamHypsToKeep);
        std::vector<int32_t> tokens(nextTokens.size(), eosTokenId);
        EXPECT_THROW(beamScorer.process(inputIds, nextScores, tokens, nextIndices, padTokenId, eosTokenId),
                std::runtime_error);

        //  check all batches are done
        beamScorer = BeamSearchScorer(batchSize, maxLen, numBeams, lenPenalty, doEarlyStopping, numBeamHypsToKeep);
        tokens = std::vector<int32_t>(nextTokens.size(), 1);
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            for (int beamId = 0; beamId < numBeams; ++beamId) {
                tokens[batchId * 2 * numBeams + beamId] = eosTokenId;
            }
        }
        beamScorer.process(inputIds, nextScores, tokens, nextIndices, padTokenId, eosTokenId);
        EXPECT_TRUE(beamScorer.isDone());

        // check
        beamScorer = BeamSearchScorer(batchSize, maxLen, numBeams, lenPenalty, doEarlyStopping, numBeamHypsToKeep);
        tokens = nextTokens;
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            tokens[batchId * 2 * numBeams + 1] = eosTokenId;
        }
        auto beam_outputs = beamScorer.process(inputIds, nextScores, tokens, nextIndices, padTokenId, eosTokenId);
        auto outputScores = std::get<0>(beam_outputs);
        auto outputTokens = std::get<1>(beam_outputs);
        auto outputIndices = std::get<2>(beam_outputs);

        // check all outptus
        // cut out id of eos token and take best `numBeams` outputs
        std::vector<int32_t> expectedOutputTokens(batchSize * numBeams);
        std::vector<float> expectedOutputScores(batchSize * numBeams);
        std::vector<int32_t> expectedOutputIndices(batchSize * numBeams);

        for (int batchId = 0; batchId < batchSize; ++batchId) {
            expectedOutputTokens[batchId * numBeams] = nextTokens[batchId * 2 * numBeams];
            expectedOutputScores[batchId * numBeams] = nextScores[batchId * 2 * numBeams];
            //  add numBeams * batchIdx
            expectedOutputIndices[batchId * numBeams] = nextIndices[batchId * 2 * numBeams] + batchId * numBeams;
            for (int beamId = 1; beamId < numBeams; ++beamId) {
                expectedOutputTokens[batchId * numBeams + beamId] = nextTokens[batchId * 2 * numBeams + beamId + 1];
                expectedOutputScores[batchId * numBeams + beamId] = nextScores[batchId * 2 * numBeams + beamId + 1];
                expectedOutputIndices[batchId * numBeams + beamId]
                        = nextIndices[batchId * 2 * numBeams + beamId + 1] + batchId * numBeams;
            }
        }

        EXPECT_EQ(outputTokens.size(), expectedOutputTokens.size());
        EXPECT_EQ(outputScores.size(), expectedOutputScores.size());
        EXPECT_EQ(outputIndices.size(), expectedOutputIndices.size());

        for (int i = 0; i < batchSize * numBeams; ++i) {
            EXPECT_EQ(outputTokens[i], expectedOutputTokens[i]);
            EXPECT_FLOAT_EQ(outputScores[i], expectedOutputScores[i]);
            EXPECT_EQ(outputIndices[i], expectedOutputIndices[i]);
        }

        // make sure ids of eos token are correctly saved in beamHyps of beam
        // scorer
        for (int batchId; batchId < batchSize; ++batchId) {
            int correctIdx = batchId * numBeams + nextIndices[batchId * 2 * numBeams + 1];
            std::vector<int32_t> correctTokens(
                    inputIds.begin() + correctIdx * seqLen, inputIds.begin() + (correctIdx + 1) * seqLen);
            auto savedTokens = beamScorer.beamHyps[batchId].beams[0].second;
            EXPECT_EQ(correctTokens.size(), savedTokens.size());
            EXPECT_TRUE(std::equal(correctTokens.begin(), correctTokens.end(), savedTokens.begin()));
        }
    }

    void checkBeamScoresFinalize() {
        // maxLen should be only one more than current inputIds to check that
        // eos is correctly appended
        BeamSearchScorer beamScorer(batchSize, /*maxLen*/ seqLen + 1, numBeams,
                /*lenPenalty*/ 1.0,
                /*doEarlyStopping*/ false,
                /*numBeamHypsToKeep*/ 1);
        // update beams and append to inputIds
        auto tokens = nextTokens;
        //  first batch, first output has to finish with eos token id since scores
        //  are correctly sorted
        tokens[0] = eosTokenId;
        // make sure corresponding score is as good as possible to surely be picked
        // first
        nextScores[0] = 10.0;
        auto beam_outputs = beamScorer.process(inputIds, nextScores, tokens, nextIndices, padTokenId, eosTokenId);
        auto outputScores = std::get<0>(beam_outputs);
        auto outputTokens = std::get<1>(beam_outputs);
        auto outputIndices = std::get<2>(beam_outputs);

        std::vector<int32_t> newInputIds(batchSize * numBeams * (seqLen + 1));
        for (int batchId = 0; batchId < batchSize; ++batchId) {
            for (int beamId = 0; beamId < numBeams; ++beamId) {
                for (int i = 0; i < seqLen; ++i) {
                    newInputIds[(batchId * numBeams + beamId) * (seqLen + 1) + i]
                            = inputIds[outputIndices[batchId * numBeams + beamId] * seqLen + i];
                }
                newInputIds[(batchId * numBeams + beamId) * (seqLen + 1) + seqLen]
                        = outputTokens[batchId * numBeams + beamId];
            }
        }
        auto sequenceOutput
                = beamScorer.finalize(newInputIds, outputScores, outputTokens, outputIndices, padTokenId, eosTokenId);
        // since `numBeamHypsToKeep` = 1 => only return `batchSize` x
        // `maxLen`
        EXPECT_EQ(sequenceOutput.size(), batchSize * (seqLen + 1));

        // check sequence_scores
        for (auto i : sequenceOutput) {
            EXPECT_GE(i, 0);
        }
        // first batch has to finish with eos_token
        EXPECT_EQ(sequenceOutput[seqLen], eosTokenId);
        // other batches cannot finish with eos token
        for (int batchId = 1; batchId < batchSize; ++batchId) {
            EXPECT_NE(sequenceOutput[batchId * (seqLen + 1) + seqLen], eosTokenId);
        }

        // now test that if `numBeamHypsToKeep` is 3 => all beams are returned
        beamScorer = BeamSearchScorer(batchSize, /*maxLen*/ seqLen + 1, numBeams,
                /*lenPenalty*/ 1.0,
                /*doEarlyStopping*/ false,
                /*numBeamHypsToKeep*/ numBeams);
        sequenceOutput
                = beamScorer.finalize(newInputIds, outputScores, outputTokens, outputIndices, padTokenId, eosTokenId);
        EXPECT_EQ(sequenceOutput.size(), batchSize * numBeams * (seqLen + 1));
    }
};
