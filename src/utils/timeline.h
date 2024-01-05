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
#pragma once

#include <string>

#ifdef TIMELINE
#include <fcntl.h>
#include <unistd.h>
#include <json/json.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

class TimeLine {
public:
    explicit TimeLine(const std::string &name) : duringTime(0) {
        // std::lock_guard<std::mutex> lock(get_lock()); // Prevent start times from coinciding
        if (hasWhitelist && eventWhitelist.find(name) == eventWhitelist.end()) return;
        startEvent(name);
    }

    ~TimeLine() { release(); }

    void release() {
        if (duringTime >= 0) return;
        end = std::chrono::high_resolution_clock::now();
        startTimestamp = std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count();
        duringTime = std::chrono::duration<float, std::micro>(end - start).count();

        traceEvent["ts"] = startTimestamp;
        traceEvent["dur"] = duringTime;

        // std::lock_guard<std::mutex> lock(get_lock());
        TimeLine::getPool().push_back(traceEvent);
    }

    void dumpFile(const std::string &fileName) {
        release();

        std::string timeFileName = extractName(fileName);

        std::string timeStamp = getTimestamp();
        timeFileName = timeFileName + "_" + timeStamp + ".json";

        // write into json file
        std::lock_guard<std::mutex> lock(get_lock());
        int lock_file = waitingLock(fileName);

        Json::Value root;
        createIfNotExists(timeFileName);
        std::ifstream inputFile(timeFileName);
        if (inputFile.is_open()) {
            std::string json_data((std::istreambuf_iterator<char>(inputFile)), std::istreambuf_iterator<char>());
            inputFile.close();

            // Parse JSON data from string
            if (json_data.length() != 0) {
                Json::CharReaderBuilder builder;
                std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
                std::string errs;
                reader->parse(json_data.c_str(), json_data.c_str() + json_data.length(), &root, &errs);
            }

            for (int i = 0; i < TimeLine::getPool().size(); ++i) {
                root["traceEvents"].append(TimeLine::getPool()[i]);
            }

            Json::StreamWriterBuilder builder;
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            std::ofstream ofile(timeFileName);
            if (ofile.is_open()) {
                writer->write(root, &ofile);
                ofile.close();
            }
        }
        emptyPool();

        releaseLock(lock_file);
    }

    static void init() {
        initWhitelist();
        pool.reserve(40 * 2000 * 20); // 40 layers * 2000 time * 20 promotes
    }

private:
    static std::mutex &get_lock() {
        static std::mutex mutex;
        return mutex;
    }

    static std::vector<Json::Value> &getPool() { return pool; }

    static void emptyPool() { pool.clear(); }

    std::string getTimestamp() {
        std::time_t now = std::time(nullptr);
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&now));
        return buffer;
    }

    std::string extractName(const std::string &fileName) {
        size_t dotPosition = fileName.find_last_of('.');
        if (dotPosition == std::string::npos) { return fileName; }

        std::string name = fileName.substr(0, dotPosition);
        return name;
    }

    void createIfNotExists(const std::string &fileName) {
        std::ofstream file(fileName, std::ios::app);
        if (file.is_open()) { file.close(); }
    }

    int waitingLock(const std::string &fileName) {
        std::string lockFile = "/tmp/" + extractName(fileName) + ".lock";

        // Open the lock file for writing, creating it if it doesn't exist.
        int lockFileDescriptor = open(lockFile.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (lockFileDescriptor == -1) {
            std::cerr << "Error opening lock file: " << lockFile << std::endl;
            return -1;
        }

        // Try to acquire an exclusive lock on the lock file without blocking.
        struct flock fl;
        fl.l_type = F_WRLCK; // Exclusive write lock
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0; // Lock the entire file

        while (fcntl(lockFileDescriptor, F_SETLK, &fl) == -1)
            ;

        return lockFileDescriptor;
    }

    void releaseLock(int lockFileDescriptor) {
        // Release the lock and close the lock file.
        struct flock fl;
        fl.l_type = F_UNLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0; // Lock the entire file

        fcntl(lockFileDescriptor, F_SETLK, &fl);
        close(lockFileDescriptor);
    }

    inline void startEvent(const std::string &name) {
        tagName = name;
        pid = getpid();
        pid = pthread_self();
        traceEvent["ph"] = "X";
        traceEvent["cat"] = "cat";
        traceEvent["name"] = tagName.c_str();
        traceEvent["pid"] = pid;
        traceEvent["tid"] = tid;
        duringTime = -1;
        start = std::chrono::high_resolution_clock::now();
        end = start;
    };

    static void strip(std::string &word) {
        std::string whitespace("\n\t ");
        std::size_t found = word.find_last_not_of(whitespace);
        if (found != std::string::npos) {
            word.erase(found + 1);
            found = word.find_first_not_of(whitespace);
            if (found != std::string::npos) word.erase(0, found);
        } else
            word.clear();
    }

    static void initWhitelist() {
        char *value = getenv("XFT_TIMELINE_WHITELIST");
        eventWhitelist.clear();
        if (value) {
            std::string env(value);
            size_t start = 0, end;
            while ((end = env.find(',', start)) != std::string::npos) {
                auto event = std::string(env.substr(start, end - start));
                strip(event);
                if (!event.empty()) eventWhitelist[event] = 1;
                start = end + 1;
            }
            auto event = std::string(env.substr((start)));
            strip(event);

            if (!event.empty()) eventWhitelist[event] = 1;
            if (!eventWhitelist.empty()) hasWhitelist = true;
        }
    }

    std::string tagName;
    int64_t pid;
    int64_t tid;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    int64_t startTimestamp, duringTime;
    Json::Value traceEvent;
    static inline std::vector<Json::Value> pool {};
    static inline std::unordered_map<std::string, int32_t> eventWhitelist {};
    static inline bool hasWhitelist = false; // any tag list provided by env XFT_TIMELINE_WHITELIST?
};

#else

class TimeLine {
public:
    TimeLine(const std::string &tagName) { (void)tagName; }
    ~TimeLine() {}
    void release() {}
    void dumpFile(const std::string &file_name) { (void)file_name; }
    static void init() {};
};

#endif
