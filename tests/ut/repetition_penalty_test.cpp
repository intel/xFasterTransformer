// Copyright (c) 2024 Intel Corporation
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
#include "search_utils.h"
#include "gtest/gtest.h"

namespace xft {
TEST(RepetitionPenaltyTest, repetitionPenaltyLogitsProcessTest) {
    // Test input
    float logits[] = {0.2, 0.2, 0.2, 0.2, 0.2};
    int sampleOffset = 0;
    int sampleSize = 5;
    std::vector<SequenceGroupMeta *> seqGroups;
    std::vector<int32_t> promptTokens = {0, 2, 1};
    seqGroups.push_back(new SequenceGroupMeta(promptTokens));
    seqGroups[0]->getSamplingMeta()->config.repetitionPenalty = 2;

    // Call the function
    repetitionPenaltyLogitsProcess(logits, sampleOffset, sampleSize, seqGroups);

    // Check logits
    float expectedLogits_1[] = {0.1, 0.1, 0.1, 0.2, 0.2};
    for (int i = 0; i < sampleSize; i++) {
        EXPECT_NEAR(logits[i], expectedLogits_1[i], 0.001);
    }

    seqGroups[0]->get(0)->stepForward(3);

    repetitionPenaltyLogitsProcess(logits, sampleOffset, sampleSize, seqGroups);

    // Check logits
    float expectedLogits_2[] = {0.05, 0.05, 0.05, 0.1, 0.2};
    for (int i = 0; i < sampleSize; i++) {
        EXPECT_NEAR(logits[i], expectedLogits_2[i], 0.001);
    }
}
} // namespace xft

int main(int argc, char **argv) {
    srand(time(NULL));
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
