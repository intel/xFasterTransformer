// Copyright (c) 2025 Intel Corporation
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
#include <ctime>
#include <iostream>
#include <string>

namespace xft {

class Logger {
public:
    enum LogLevel { DEBUG, INFO, WARNING, ERROR };

    static void setLogLevel(LogLevel level) { logLevel = level; }

    static void debug(const char *format, ...) {
        if (logLevel <= DEBUG) {
            va_list args;
            va_start(args, format);
            log("DEBUG", format, args);
            va_end(args);
        }
    }

    static void info(const char *format, ...) {
        if (logLevel <= INFO) {
            va_list args;
            va_start(args, format);
            log("INFO", format, args);
            va_end(args);
        }
    }

    static void warning(const char *format, ...) {
        if (logLevel <= WARNING) {
            va_list args;
            va_start(args, format);
            log("WARNING", format, args);
            va_end(args);
        }
    }

    static void error(const char *format, ...) {
        if (logLevel <= ERROR) {
            va_list args;
            va_start(args, format);
            log("ERROR", format, args);
            va_end(args);
        }
    }

private:
    static LogLevel logLevel;

    static void log(const std::string &level, const char *format, va_list args) {
        std::time_t now = std::time(nullptr);
        char timeStr[20];
        std::strftime(timeStr, sizeof(timeStr), "%Y-%m-%d %H:%M:%S", std::localtime(&now));

        const char *colorCode = "";
        const char *resetCode = "\033[0m";

        if (level == "ERROR") {
            colorCode = "\033[31m"; // Red
        } else if (level == "WARNING") {
            colorCode = "\033[33m"; // Yellow
        }

        printf("%s[%s] [%s] ", colorCode, timeStr, level.c_str());
        vprintf(format, args);
        printf("%s\n", resetCode);
    }
};

} // namespace xft