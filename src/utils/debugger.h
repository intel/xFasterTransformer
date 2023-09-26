#pragma once

#include <cstdarg>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "my_types.h"

class Debugger {
public:
    Debugger() {
        debugFile = nullptr;
        ownByMe = true;
    }

    Debugger(const std::string &filename) {
        debugFile = fopen(filename.c_str(), "w");
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
    void dumpMatrix(hpj::Matrix<T> &m, bool print_all = false) {
        std::ostringstream oss;
        int rows = m.Rows();
        int cols = m.Cols();

        // Collect values to string
        if (print_all == true) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols - 1; ++j) {
                    oss << float(m(i, j)) << ", ";
                }
                oss << float(m(i, cols - 1)) << std::endl;
            }
        } else {
            if (rows <= 12) {
                for (int i = 0; i < rows; ++i) {
                    if (cols <= 12) {
                        for (int j = 0; j < cols - 1; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }
                    } else {
                        for (int j = 0; j < 6; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }

                        oss << "..., ";

                        for (int j = cols - 6; j < cols - 1; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }
                    }
                    oss << float(m(i, cols - 1)) << std::endl;
                }
            } else {
                for (int i = 0; i < 6; ++i) {
                    if (cols <= 12) {
                        for (int j = 0; j < cols - 1; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }
                    } else {
                        for (int j = 0; j < 6; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }

                        oss << "..., ";

                        for (int j = cols - 6; j < cols - 1; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }
                    }
                    oss << float(m(i, cols - 1)) << std::endl;
                }

                oss << "..." << std::endl;

                for (int i = rows - 6; i < rows; ++i) {
                    if (cols < 10) {
                        for (int j = 0; j < cols - 1; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }
                    } else {
                        for (int j = 0; j < 6; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }

                        oss << "..., ";

                        for (int j = cols - 6; j < cols - 1; ++j) {
                            oss << float(m(i, j)) << ", ";
                        }
                    }
                    oss << float(m(i, cols - 1)) << std::endl;
                }
            }
        }

        // Write to file
        if (debugFile) {
            std::string str = oss.str();
            fwrite(str.c_str(), sizeof(char), str.size(), debugFile);
            fflush(debugFile);
        }
    }

    template <typename T>
    void dumpMatrix(T *data, int rows, int cols, int stride, bool print_all = false) {
        std::ostringstream oss;

        // Collect values to string
        if (print_all == true) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols - 1; ++j) {
                    oss << float(data[i * stride + j]) << ", ";
                }
                oss << float(data[i * stride + cols - 1]) << std::endl;
            }
        } else {
            if (rows <= 12) {
                for (int i = 0; i < rows; ++i) {
                    if (cols <= 12) {
                        for (int j = 0; j < cols - 1; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }
                    } else {
                        for (int j = 0; j < 6; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }

                        oss << "..., ";

                        for (int j = cols - 6; j < cols - 1; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }
                    }
                    oss << float(data[i * stride + cols - 1]) << std::endl;
                }
            } else {
                for (int i = 0; i < 6; ++i) {
                    if (cols <= 12) {
                        for (int j = 0; j < cols - 1; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }
                    } else {
                        for (int j = 0; j < 6; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }

                        oss << "..., ";

                        for (int j = cols - 6; j < cols - 1; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }
                    }
                    oss << float(data[i * stride + cols - 1]) << std::endl;
                }

                oss << "..." << std::endl;

                for (int i = rows - 6; i < rows; ++i) {
                    if (cols < 10) {
                        for (int j = 0; j < cols - 1; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }
                    } else {
                        for (int j = 0; j < 6; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }

                        oss << "..., ";

                        for (int j = cols - 6; j < cols - 1; ++j) {
                            oss << float(data[i * stride + j]) << ", ";
                        }
                    }
                    oss << float(data[i * stride + cols - 1]) << std::endl;
                }
            }
        }

        // Write to file
        if (debugFile) {
            std::string str = oss.str();
            fwrite(str.c_str(), sizeof(char), str.size(), debugFile);
            fflush(debugFile);
        }
    }

    // Function to store float* data to a file
    template <typename T>
    void storeMatrix(const std::string &filename, const T *data, int rows, int cols) {
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char *>(&rows), sizeof(int));
            file.write(reinterpret_cast<const char *>(&cols), sizeof(int));
            file.write(reinterpret_cast<const char *>(data), rows * cols * sizeof(T));
            file.close();
        } else {
            std::cerr << "Unable to open file for writing: " << filename << std::endl;
        }
    }

    // Function to load float* data from a file
    void loadMatrixSize(const std::string &filename, int &rows, int &cols) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char *>(&rows), sizeof(int));
            file.read(reinterpret_cast<char *>(&cols), sizeof(int));
            file.close();
        } else {
            std::cerr << "Unable to open file for reading: " << filename << std::endl;
        }
    }

    template <typename T>
    void loadMatrixData(const std::string &filename, T *data, int rows, int cols) {
        std::ifstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char *>(&rows), sizeof(int));
            file.read(reinterpret_cast<char *>(&cols), sizeof(int));
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