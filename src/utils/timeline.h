#pragma once

#include <string>

#ifdef TIMELINE

#include <fcntl.h>
#include <unistd.h>
#include <json/json.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

class TimeLine {
public:
    explicit TimeLine(const std::string &tag_name) {
        // std::lock_guard<std::mutex> lock(get_lock()); // Prevent start times from coinciding
        this->tag_name = tag_name;
        this->pid = getpid();
        this->tid = pthread_self();
        this->trace_event["ph"] = "X";
        this->trace_event["cat"] = "cat";
        this->trace_event["name"] = tag_name.c_str();
        this->trace_event["pid"] = pid;
        this->trace_event["tid"] = tid;
        this->during_time = -1;
        this->start = std::chrono::high_resolution_clock::now();
        this->end = this->start;
    }

    ~TimeLine() {
        if (during_time < 0) release();
    }

    void release() {
        this->end = std::chrono::high_resolution_clock::now();
        this->start_timestap = std::chrono::duration_cast<std::chrono::microseconds>(start.time_since_epoch()).count();
        this->during_time = std::chrono::duration<float, std::micro>(end - start).count();

        this->trace_event["ts"] = this->start_timestap;
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

        release_lock(lock_file);
    }

private:
    static std::mutex &get_lock() {
        static std::mutex mutex;
        return mutex;
    }

    static std::vector<Json::Value> &get_pool() {
        static std::vector<Json::Value> pool;
        pool.reserve(40*2000*20); // 40 layers * 2000 time * 20 promotes
        return pool;
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

    std::string tag_name;
    int64_t pid;
    int64_t tid;
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    int64_t start_timestap, during_time;
    Json::Value trace_event;
};

#else

class TimeLine {
public:
    TimeLine(const std::string &tag_name) {}
    ~TimeLine() {}
    void release() {}
    void dump_file(const std::string &file_name) {}
};

#endif
