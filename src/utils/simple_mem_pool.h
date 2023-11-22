#pragma once

#include <iostream>
#include <unordered_map>
#include <string>
#include <cstdlib>

class SimpleMemPool {
private:
    std::unordered_map<std::string, std::pair<void*, size_t>> memoryMap;

    // Private constructor to enforce Singleton pattern
    SimpleMemPool() {}

    SimpleMemPool(const SimpleMemPool &mgr) = delete;
    SimpleMemPool &operator=(const SimpleMemPool &mgr) = delete;

public:
    // Static method to get the singleton instance
    static SimpleMemPool& instance() {
        static SimpleMemPool memManager;
        return memManager;
    }

    // Allocate or reallocate memory buffer based on name and size
    void* getBuffer(const std::string& name, size_t size, size_t alignment = 64) {
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
        void* buffer = aligned_alloc(alignment, size);
        if (buffer == nullptr) {
            // Allocation failed
            std::cerr << "Memory allocation failed for buffer: " << name << std::endl;
            exit(-1);
        }

        // Update or insert entry in the mapping
        memoryMap[name] = std::make_pair(buffer, size);

        return buffer;
    }

    // Destructor to free all allocated memory on program termination
    ~SimpleMemPool() {
        for (auto& entry : memoryMap) {
            free(entry.second.first);
        }
        memoryMap.clear();
    }
};