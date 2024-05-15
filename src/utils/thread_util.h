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
#include <omp.h>

#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>

namespace xft {

template <typename Lambda>
void parallel_for(int tasks, const Lambda &fn) {
#pragma omp parallel for
    for (int i = 0; i < tasks; i++) {
        fn(i);
    }
}

template <typename Lambda>
void parallel_for_dschedule(int tasks, const Lambda &fn) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < tasks; i++) {
        fn(i);
    }
}

class ThreadPool {
public:
    static ThreadPool &getInstance() {
        static ThreadPool instance;
        return instance;
    }

    template <typename F, typename... Args>
    void addTask(F &&f, Args &&...args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    ThreadPool() : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) { return; }
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    static constexpr size_t numThreads = 1;
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

} // namespace xft