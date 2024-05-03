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
#include <vector>
#include "abstract_searcher.h"

namespace xft {
struct SamplingMeta {
    bool done;
    std::vector<std::vector<int>> stopWordsList;
    std::vector<int> stopWordsIndex;
    std::vector<int> cachedRepetVec;
    SearcherConfig config;

    SamplingMeta() : done(false) {}

    SamplingMeta(SearcherConfig config, std::vector<std::vector<int>> stopWordsList_)
        : done(false), config(config), stopWordsList(stopWordsList_) {
        // Remove empty words, eos id, and words containing non-positive elements.
        for (auto it = stopWordsList.rbegin(); it != stopWordsList.rend(); ++it) {
            if ((*it).empty() || ((*it).size() == 1 && (*it)[0] == config.eosTokenId)) {
                stopWordsList.erase(std::next(it).base());
                continue;
            }
            for (auto x : *it) {
                if (x <= 0) { stopWordsList.erase(std::next(it).base()); }
            }
        }
    }
};

}; // namespace xft