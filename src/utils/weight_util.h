#pragma once

#include <fstream>
#include <string>

namespace xft{

// Read weights from file
template <typename T>
int readFile(const std::string &path, T *values, int num) {
    int count = 0;
    int nthreads = std::min(omp_get_max_threads(), 16);
    int chunk_size = (num + nthreads - 1) / nthreads;
    #pragma omp parallel num_threads(nthreads) reduction(+:count)
    {
        int tid = omp_get_thread_num();
        int start_idx = tid * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, num);
        
        std::ifstream file(path, std::ios::binary);
        if (file.is_open()) {
            file.seekg(start_idx * sizeof(T), std::ios::beg);
            file.read(reinterpret_cast<char *>(values + start_idx), (end_idx - start_idx) * sizeof(T));
            count += file.gcount() / sizeof(T);
            file.close();
        }
    }
    return count;
}
} // namespace xft