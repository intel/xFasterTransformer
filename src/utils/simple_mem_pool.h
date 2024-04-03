
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
    std::unordered_map<std::string, std::pair<void *, size_t>> memoryMap;

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
    void *getBuffer(const std::string &name, size_t size, size_t alignment = 64) {
        if (size == 0) {
            // std::cout << "[Warning] Try to allocate 0 bytes for buffer:" << name << std::endl;
            return nullptr;
        }
        auto it = memoryMap.find(name);

        if (it != memoryMap.end()) {
            // Buffer with the given name found
            if (it->second.second >= size) {
                // Existing buffer size is sufficient, return it
                return it->second.first;
            } else {
                // Reallocate the buffer
                free(it->second.first);
            }
        }

        // Allocate new aligned buffer
        void *buffer = xft::alloc(size, alignment);
        if (buffer == nullptr) {
            // Allocation failed
            std::cerr << "Memory allocation failed for buffer:" << name << " size:" << size << std::endl;
            exit(-1);
        }

        // Update or insert entry in the mapping
        memoryMap[name] = std::make_pair(buffer, size);

        return buffer;
    }

    // Destructor to free all allocated memory on program termination
    ~SimpleMemPool() {
        for (auto &entry : memoryMap) {
            free(entry.second.first);
        }
        memoryMap.clear();
    }
};