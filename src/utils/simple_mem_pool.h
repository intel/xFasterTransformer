
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

#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

#include "allocator.h"

class SimpleMemPool {
private:
    std::unordered_map<std::string, std::tuple<void *, size_t, void *>> memoryMap;

    // Private constructor to enforce Singleton pattern
    SimpleMemPool() {}

    SimpleMemPool(const SimpleMemPool &mgr) = delete;
    SimpleMemPool &operator=(const SimpleMemPool &mgr) = delete;

public:
    // Static method to get the singleton instance
    static SimpleMemPool &instance() {
        static SimpleMemPool memManager;
        return memManager;
    }

    bool cached(const std::string &name) {
        auto it = memoryMap.find(name);
        if (it != memoryMap.end()) return true;
        return false;
    }

    // Allocate or reallocate memory buffer based on name and size
    void *getBuffer(const std::string &name, size_t size, void *device = nullptr, size_t alignment = 64) {
        if (name.empty()) return nullptr;

        if (size == 0) {
            // std::cout << "[Warning] Try to allocate 0 bytes for buffer:" << name << std::endl;
            return nullptr;
        }
        auto it = memoryMap.find(name);

        if (it != memoryMap.end()) {
            // Buffer with the given name found
            if (std::get<1>(it->second) >= size) {
                // Existing buffer size is sufficient, return it
                return std::get<0>(it->second);
            } else {
                // Reallocate the buffer
                xft::dealloc(std::get<0>(it->second), std::get<2>(it->second));
            }
        }

        // Allocate new aligned buffer
        void *buffer = xft::alloc(size, device, alignment);
        if (buffer == nullptr) {
            // Allocation failed
            std::cerr << "Memory allocation failed for buffer:" << name << " size:" << size << std::endl;
            exit(-1);
        }

        // Update or insert entry in the mapping
        memoryMap[name] = std::make_tuple(buffer, size, device);

        return buffer;
    }

    // Free allocated memory based on name
    void freeBuffer(const std::string &name) {
        auto it = memoryMap.find(name);

        if (it != memoryMap.end()) {
            xft::dealloc(std::get<0>(it->second), std::get<2>(it->second));
            memoryMap.erase(it->first);
        }
    }

    // Destructor to free all allocated memory on program termination
    ~SimpleMemPool() {
#ifndef XFT_GPU
        for (auto &entry : memoryMap) {
            xft::dealloc(std::get<0>(entry.second), std::get<2>(entry.second));
        }
#endif
        memoryMap.clear();
    }
};