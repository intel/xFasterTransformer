#pragma once
#include <omp.h>

#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <condition_variable>

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
    static ThreadPool& getInstance() {
        static ThreadPool instance;
        return instance;
    }

    template<typename F, typename... Args>
    void addTask(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        stop = true;
        condition.notify_all();
        for (std::thread& worker : workers) {
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
                        if (stop && tasks.empty()) {
                            return;
                        }
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