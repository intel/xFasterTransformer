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

#include <mutex>

template <typename T>
class SingletonBase {
public:
    static T &getInstance() {
        // 使用双检锁机制确保线程安全性
        if (instance_ == nullptr) {
            std::lock_guard<std::mutex> lock(mutex_);
            if (instance_ == nullptr) {
                instance_ = new T;
                atexit(cleanup);
            }
        }
        return *instance_;
    }

    // 禁用拷贝构造函数和赋值运算符
    SingletonBase(const SingletonBase &) = delete;
    SingletonBase &operator=(const SingletonBase &) = delete;

protected:
    // 构造函数和析构函数为受保护，确保只能通过派生类访问
    SingletonBase() {
        // 可以在这里初始化一些资源
    }

    virtual ~SingletonBase() {
        // 可以在这里释放资源
    }

private:
    static void cleanup() {
        if (instance_ != nullptr) {
            delete instance_;
            instance_ = nullptr;
        }
    }

    static T *instance_;
    static std::mutex mutex_;
};

// 静态成员初始化
template <typename T>
T *SingletonBase<T>::instance_ = nullptr;

template <typename T>
std::mutex SingletonBase<T>::mutex_;
