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

#include <cstdarg>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "environment.h"
#include "my_types.h"
#include "normal_float4x2.h"
#include "uint4x2.h"

class Debugger {
public:
    Debugger() {
        debugFile = nullptr;
        ownByMe = true;
    }

    Debugger(const std::string &filename) {
        std::string debugFilePath = filename;
        std::string debugDir = Env::getInstance().getDebugDir();
        if (!debugDir.empty()) {
            if (debugDir.back() != '/') { debugDir += '/'; }
            debugFilePath = debugDir + filename;

            if (!std::filesystem::exists(debugDir)) {
                if (!std::filesystem::create_directories(debugDir)) {
                    std::cout << "[Error] Error creating debug directory: " << debugDir << std::endl;
                    exit(-1);
                }
            }
        }

        debugFile = fopen(debugFilePath.c_str(), "w");
        ownByMe = true;
    }

    ~Debugger() {
        if (debugFile && ownByMe) { fclose(debugFile); }
    }

    Debugger(const Debugger &debugger) {
        this->debugFile = debugger.debugFile;
        this->ownByMe = false;
    }

    Debugger &operator=(const Debugger &debugger) {
        if (debugFile && ownByMe) { fclose(debugFile); }

        this->debugFile = debugger.debugFile;
        this->ownByMe = false;

        return *this;
    }

    static std::string formatStr(const char *format, ...) {
        char buffer[256];
        va_list args;
        va_start(args, format);
        vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);
        return buffer;
    }

    void debugPrint(const char *format, ...) {
        char buffer[256];
        va_list args;
        va_start(args, format);
        int size = vsnprintf(buffer, sizeof(buffer), format, args);
        va_end(args);

        // Write to file
        if (size > 0) {
            if (debugFile) {
                fwrite(buffer, sizeof(char), size, debugFile);
                fflush(debugFile);
            }
        }
    }

    template <typename T>
    void outputStream(std::ostringstream &oss, T val, bool isFinished = false) {
        if (isFinished == false) {
            if constexpr (std::is_same_v<T, uint4x2_t> || std::is_same_v<T, nf4x2_t>) {
                oss << std::setprecision(8) << std::setw(15) << std::right << float(val.get_v1()) << ", "
                    << float(val.get_v2()) << ", ";
            } else {
                oss << std::setprecision(8) << std::setw(15) << std::right << float(val) << ", ";
            }
        } else {
            if constexpr (std::is_same_v<T, uint4x2_t> || std::is_same_v<T, nf4x2_t>) {
                oss << std::setprecision(8) << std::setw(15) << std::right << float(val.get_v1()) << ", "
                    << float(val.get_v2()) << std::endl;
            } else {
                oss << std::setprecision(8) << std::setw(15) << std::right << float(val) << std::endl;
            }
        }
    }

    template <typename T>
    void dumpMatrix(xft::Matrix<T> &matrix, bool print_all = false, void *device = nullptr) {
        std::ostringstream oss;
        xft::Matrix<T> m;
        uint64_t rows = matrix.Rows();
        uint64_t cols = matrix.Cols();
        uint64_t stride = matrix.Stride();
        T *data = nullptr;

        if (device != nullptr) {
            size_t size = rows * stride * sizeof(T);
            data = (T *)xft::alloc(size);
            xft::memcopy(data, matrix.Data(), size, device);
            m.Assign(data, rows, cols, stride);
        } else {
            m.Assign(matrix.Data(), rows, cols, stride);
        }

        // Collect values to string
        if (print_all == true) {
            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t j = 0; j < cols - 1; ++j) {
                    outputStream(oss, m(i, j));
                }
                outputStream(oss, m(i, cols - 1), true);
            }
        } else {
            if (rows <= 12) {
                for (uint64_t i = 0; i < rows; ++i) {
                    if (cols <= 12) {
                        for (uint64_t j = 0; j < cols - 1; ++j) {
                            outputStream(oss, m(i, j));
                        }
                    } else {
                        for (uint64_t j = 0; j < 6; ++j) {
                            outputStream(oss, m(i, j));
                        }

                        oss << std::setprecision(8) << std::setw(15) << std::right << "..., ";

                        for (uint64_t j = cols - 6; j < cols - 1; ++j) {
                            outputStream(oss, m(i, j));
                        }
                    }
                    outputStream(oss, m(i, cols - 1), true);
                }
            } else {
                for (uint64_t i = 0; i < 6; ++i) {
                    if (cols <= 12) {
                        for (uint64_t j = 0; j < cols - 1; ++j) {
                            outputStream(oss, m(i, j));
                        }
                    } else {
                        for (uint64_t j = 0; j < 6; ++j) {
                            outputStream(oss, m(i, j));
                        }

                        oss << std::setprecision(8) << std::setw(15) << std::right << "..., ";

                        for (uint64_t j = cols - 6; j < cols - 1; ++j) {
                            outputStream(oss, m(i, j));
                        }
                    }
                    outputStream(oss, m(i, cols - 1), true);
                }

                oss << std::setprecision(8) << std::setw(15) << std::right << "..." << std::endl;

                for (uint64_t i = rows - 6; i < rows; ++i) {
                    if (cols < 10) {
                        for (uint64_t j = 0; j < cols - 1; ++j) {
                            outputStream(oss, m(i, j));
                        }
                    } else {
                        for (uint64_t j = 0; j < 6; ++j) {
                            outputStream(oss, m(i, j));
                        }

                        oss << std::setprecision(8) << std::setw(15) << std::right << "..., ";

                        for (uint64_t j = cols - 6; j < cols - 1; ++j) {
                            outputStream(oss, m(i, j));
                        }
                    }
                    outputStream(oss, m(i, cols - 1), true);
                }
            }
        }

        // Write to file
        if (debugFile) {
            std::string str = oss.str();
            fwrite(str.c_str(), sizeof(char), str.size(), debugFile);
            fflush(debugFile);
        }

        if (device != nullptr) xft::dealloc(data);
    }

    template <typename T>
    void dumpMatrix(
            T *input, uint64_t rows, uint64_t cols, uint64_t stride, bool print_all = false, void *device = nullptr) {
        std::ostringstream oss;
        T *data = nullptr;

        if (device != nullptr) {
            size_t size = rows * stride * sizeof(T);
            data = (T *)xft::alloc(size);
            xft::memcopy(data, input, size, device);
        } else {
            data = input;
        }

        // Collect values to string
        if (print_all == true) {
            for (uint64_t i = 0; i < rows; ++i) {
                for (uint64_t j = 0; j < cols - 1; ++j) {
                    oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j]) << ", ";
                }
                oss << float(data[i * stride + cols - 1]) << std::endl;
            }
        } else {
            if (rows <= 12) {
                for (uint64_t i = 0; i < rows; ++i) {
                    if (cols <= 12) {
                        for (uint64_t j = 0; j < cols - 1; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }
                    } else {
                        for (uint64_t j = 0; j < 6; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }

                        oss << std::setprecision(8) << std::setw(15) << std::right << "..., ";

                        for (uint64_t j = cols - 6; j < cols - 1; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }
                    }
                    oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + cols - 1])
                        << std::endl;
                }
            } else {
                for (uint64_t i = 0; i < 6; ++i) {
                    if (cols <= 12) {
                        for (uint64_t j = 0; j < cols - 1; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }
                    } else {
                        for (uint64_t j = 0; j < 6; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }

                        oss << std::setprecision(8) << std::setw(15) << std::right << "..., ";

                        for (uint64_t j = cols - 6; j < cols - 1; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }
                    }
                    oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + cols - 1])
                        << std::endl;
                }

                oss << std::setprecision(8) << std::setw(15) << std::right << "..." << std::endl;

                for (uint64_t i = rows - 6; i < rows; ++i) {
                    if (cols < 10) {
                        for (uint64_t j = 0; j < cols - 1; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }
                    } else {
                        for (uint64_t j = 0; j < 6; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }

                        oss << std::setprecision(8) << std::setw(15) << std::right << "..., ";

                        for (uint64_t j = cols - 6; j < cols - 1; ++j) {
                            oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + j])
                                << ", ";
                        }
                    }
                    oss << std::setprecision(8) << std::setw(15) << std::right << float(data[i * stride + cols - 1])
                        << std::endl;
                }
            }
        }

        // Write to file
        if (debugFile) {
            std::string str = oss.str();
            fwrite(str.c_str(), sizeof(char), str.size(), debugFile);
            fflush(debugFile);
        }

        if (device != nullptr) xft::dealloc(data);
    }

    // Function to store float* data to a file
    template <typename T>
    void storeMatrix(const std::string &filename, const T *data, uint64_t rows, uint64_t cols) {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char *>(&rows), sizeof(uint64_t));
            file.write(reinterpret_cast<const char *>(&cols), sizeof(uint64_t));
            file.write(reinterpret_cast<const char *>(data), rows * cols * sizeof(T));
            file.close();
        } else {
            std::cerr << "Unable to open file for writing: " << filename << std::endl;
        }
    }

    // Function to load float* data from a file
    void loadMatrixSize(const std::string &filename, uint64_t &rows, uint64_t &cols) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char *>(&rows), sizeof(uint64_t));
            file.read(reinterpret_cast<char *>(&cols), sizeof(uint64_t));
            file.close();
        } else {
            std::cerr << "Unable to open file for reading: " << filename << std::endl;
        }
    }

    template <typename T>
    void loadMatrixData(const std::string &filename, T *data, uint64_t rows, uint64_t cols) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char *>(&rows), sizeof(uint64_t));
            file.read(reinterpret_cast<char *>(&cols), sizeof(uint64_t));
            file.read(reinterpret_cast<char *>(data), rows * cols * sizeof(T));
            file.close();
        } else {
            std::cerr << "Unable to open file for reading: " << filename << std::endl;
        }
    }

private:
    // Note: the file handle may be shared between multiple objects
    FILE *debugFile;

    // True when the debugFile should be closed by me
    bool ownByMe;
};