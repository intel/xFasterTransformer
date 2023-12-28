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

#include <cstdlib>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>

class TimeLine {
public:
    explicit TimeLine(const std::string &tag_name): during_time(0) {
        // std::lock_guard<std::mutex> lock(get_lock()); // Prevent start times from coinciding
        if (having_whitelist && tag_whitelist.find(tag_name) == tag_whitelist.end())
            return;
        startTimeLineEvent(tag_name);
    }

    ~TimeLine() {
        release();
    }

    void release() {
        if (during_time >= 0)
            return;
        this->end = std::chrono::high_resolution_clock::now();
        this->start_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count();
        this->during_time = std::chrono::duration<float, std::micro>(end - start).count();

        this->trace_event["ts"] = this->start_timestamp;
        this->trace_event["dur"] = this->during_time;

        // std::lock_guard<std::mutex> lock(get_lock());
        TimeLine::get_pool().push_back(trace_event);
    }

    void dump_file(const std::string &file_name) {
        release();

        std::string time_file_name = extract_name(file_name);
        std::string timeStamp = get_time_stamp();
        time_file_name = time_file_name + "_" + timeStamp + ".json";

        // write into json file
        std::lock_guard<std::mutex> lock(get_lock());
        int lock_file = waiting_lock(file_name);

        Json::Value root;
        create_if_not_exists(time_file_name);
        std::ifstream inputFile(time_file_name);
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

            for (int i = 0; i < TimeLine::get_pool().size(); ++i) {
                root["traceEvents"].append(TimeLine::get_pool()[i]);
            }

            Json::StreamWriterBuilder builder;
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            std::ofstream ofile(time_file_name);
            if (ofile.is_open()) {
                writer->write(root, &ofile);
                ofile.close();
            }
        }
        empty_pool();

        release_lock(lock_file);
    }

    static void init(){
        init_tag_whitelist();
        pool.reserve(40*2000*20); // 40 layers * 2000 time * 20 promotes
    }

private:
    static std::mutex &get_lock() {
        static std::mutex mutex;
        return mutex;
    }

    static std::vector<Json::Value> &get_pool() {
        return pool;
    }

    static void empty_pool() {
        pool.clear();
    }

    std::string get_time_stamp() {
        std::time_t now = std::time(nullptr);
        char buffer[80];
        std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", std::localtime(&now));
        return buffer;
    }

    std::string extract_name(const std::string &file_name) {
        size_t dotPosition = file_name.find_last_of('.');
        if (dotPosition == std::string::npos) { return file_name; }

        std::string name = file_name.substr(0, dotPosition);
        return name;
    }

    void create_if_not_exists(const std::string &file_name) {
        std::ofstream file(file_name, std::ios::app);
        if (file.is_open()) { file.close(); }
    }

    int waiting_lock(const std::string &file_name) {
        std::string lock_file = "/tmp/" + extract_name(file_name) + ".lock";

        // Open the lock file for writing, creating it if it doesn't exist.
        int lockFileDescriptor = open(lock_file.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        if (lockFileDescriptor == -1) {
            std::cerr << "Error opening lock file: " << lock_file << std::endl;
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

    void release_lock(int lockFileDescriptor) {
        // Release the lock and close the lock file.
        struct flock fl;
        fl.l_type = F_UNLCK;
        fl.l_whence = SEEK_SET;
        fl.l_start = 0;
        fl.l_len = 0; // Lock the entire file

        fcntl(lockFileDescriptor, F_SETLK, &fl);
        close(lockFileDescriptor);
    }

    inline void startTimeLineEvent(const std::string &name){
        tag_name = name;
        pid = getpid();
        pid = pthread_self();
        trace_event["ph"] = "X";
        trace_event["cat"] = "cat";
        trace_event["name"] = tag_name.c_str();
        trace_event["pid"] = pid;
        trace_event["tid"] = tid;
        during_time = -1;
        start = std::chrono::high_resolution_clock::now();
        end = start;
    };

    static void strip(std::string &word){
        std::string whitespace("\n\t ");
        std::size_t found = word.find_last_not_of(whitespace);
        if (found!=std::string::npos) {
            word.erase(found+1);
            found = word.find_first_not_of(whitespace);
            if (found!=std::string::npos)
                word.erase(0, found);
        }
        else
            word.clear();
    }

    static void init_tag_whitelist() {
        char *value = getenv("XFT_TIMELINE_WHITELIST");
        tag_whitelist.clear();
        if (value) {
            std::string env(value);
            size_t start = 0, end;
            while ((end = env.find(',', start)) != std::string::npos) {
                auto event = std::string(env.substr(start, end - start));
                strip(event);
                if(!event.empty())
                    tag_whitelist[event] = 1;
                start = end + 1;
            }
            auto event = std::string(env.substr((start)));
            strip(event);

            if(!event.empty())
                tag_whitelist[event] = 1;
            if (!tag_whitelist.empty())
                having_whitelist = true;
        }
    }

    std::string tag_name;
    int64_t pid;
    int64_t tid;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    int64_t start_timestamp, during_time;
    Json::Value trace_event;
    static inline std::vector<Json::Value> pool{};
    static inline std::unordered_map<std::string, int32_t> tag_whitelist{};
    static inline bool having_whitelist= false; // any tag list provided by env XFT_TIMELINE_WHITELIST?
};

#else

class TimeLine {
public:
    TimeLine(const std::string &tag_name) {(void) tag_name;}
    ~TimeLine() {}
    void release() {}
    void dump_file(const std::string &file_name) {(void) file_name;}
    static void init(){};
};

#endif
