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
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>
#include <json/json.h>

#include "timeline.h"
#include "gtest/gtest.h"

struct CEvent {
    CEvent(std::shared_ptr<std::string> tag, int64_t ts, int64_t dur, int64_t pid, int64_t tid)
        : tag_(tag), ts_(ts), dur_(dur), pid_(pid), tid_(tid) {}

    bool Validate() {
        if (ts_ <= 0 || pid_ <= 0 || tid_ <= 0 || tag_ == nullptr) return false;
        return true;
    }

    int64_t dur_;
    int64_t pid_;
    int64_t tid_;
    int64_t ts_;
    std::shared_ptr<std::string> tag_;
};

class CTimelineEvents {
public:
    CTimelineEvents() : len_(0) {};
    CTimelineEvents(std::string &jsonfile) {
        FromJsonFile(std::move(jsonfile));
        CTimelineEvents();
    }

    uint32_t Len() { return len_; }

    bool Validate() {
        for (auto &evt : events_) {
            if (!evt->Validate()) return false;
        }
        return true;
    }

    bool HavingTag(std::string &&tag) {
        for (auto &evt : events_) {
            if (*evt->tag_ == tag) return true;
        }
        return false;
    }

    void FromJsonFile(std::string &&jsonfile) {
        std::ifstream fJson(jsonfile, std::ios::binary);
        if (fJson.is_open()) {
            Json::Value records;
            fJson >> records;
            for (auto r : records["traceEvents"]) {
                auto tag = r["name"].asString();
                auto dur = r["dur"].asInt64();
                auto ts = r["ts"].asInt64();
                auto pid = r["pid"].asInt64();
                auto tid = r["tid"].asInt64();
                auto event = std::make_shared<CEvent>(std::make_shared<std::string>(tag), ts, dur, pid, tid);
                events_.push_back(event);
                len_++;
            }
        }
    }

private:
    std::vector<std::shared_ptr<CEvent>> events_;
    uint32_t len_;
};

std::vector<std::string> GetJsonFiles(std::string folderPath, std::string prefix) {
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(folderPath)) {
        const auto &file = entry.path().filename().string();
        if (entry.path().filename().extension() == ".json" && file.find(prefix) == 0) files.push_back(file);
    }
    return files;
}

TEST(timeline, test1) {
    {
        TimeLine t1("Stage1");
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
    {
        TimeLine t2("Stage2");
        std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
    TimeLine t("dump_file");
#ifdef TIMELINE
    auto jsonFiles = GetJsonFiles("./", "timeline");
#endif
    t.dump_file("timeline.json");
#ifdef TIMELINE
    auto jsonFilesAferTest = GetJsonFiles("./", "timeline");
    for (auto &f : jsonFiles)
        jsonFilesAferTest.remove(f);
    ASSERT_EQ(jsonFilesAferTest.size(), 1);
    CTimelineEvents evts(jsonFilesAferTest.front());
    ASSERT_TRUE(evts.Validate());
    ASSERT_TRUE(evts.HavingTag("Stage1"));
    ASSERT_TRUE(evts.HavingTag("Stage2"));
    ASSERT_TRUE(evts.HavingTag("dump_file"));
#endif
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
