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
#pragma once
#include "sequence.h"

// Insert an element into a sorted vector while maintaining the order
void insertAndSort(std::vector<int> &targetVector, int num);

void repetitionPenaltyLogitsProcess(float penalty, float *logits, int sampleOffset, int sampleSize,
        std::vector<int> &inputIds, int batchSize, std::vector<std::vector<int>> &cachedVec, int step, bool multiRank);

void stopWordsCheck(std::vector<int> &nextTokenIds, std::vector<std::vector<int>> &stopWordsList,
        std::vector<std::vector<int>> &stopWordsIndex, std::vector<int> &doneBatch);

namespace xft {
void repetitionPenaltyLogitsProcess(
        float *logits, int sampleOffset, int sampleSize, std::vector<SequenceGroupMeta *> &seqGroups);
} // namespace xft