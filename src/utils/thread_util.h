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
    ThreadPool(size_t numThreads) : stop(false) {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queueMutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });

                        if (this->stop && this->tasks.empty()) { return; }

                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }

                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    void enqueue(F &&f, Args &&...args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            tasks.emplace([f, args...] { f(args...); });
        }

        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            stop = true;
        }

        condition.notify_all();

        for (std::thread &worker : workers) {
            worker.join();
        }
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};