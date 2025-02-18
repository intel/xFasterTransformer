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
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

class ProgressBar {
public:
    ProgressBar(int total, std::string desc = "", int width = 48, float frequency = 0.1)
        : total(total), desc(desc), width(width), frequency(frequency), count(0), last_elapsed(0) {
        start_time_ = std::chrono::steady_clock::now();
    }

    void update() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_time_).count();

        count++;
        if (elapsed - last_elapsed < frequency) { return; }

        float progress = static_cast<float>(count) / total;
        int pos = static_cast<int>(width * progress);

        std::string bar = "[" + std::string(pos, '=') + ">" + std::string(width - pos, ' ') + "]";
        double remaining = (elapsed - last_elapsed) * (total - count);

        std::cout << "\r" << desc << bar << " " << std::fixed << std::setprecision(1) << (progress * 100.0) << "% "
                  << count << "/" << total << " [Elapsed: " << formatTime(elapsed)
                  << " | Remaining: " << formatTime(remaining) << "]";
        std::cout.flush();

        last_elapsed = elapsed;
    }

    std::string formatTime(double seconds) {
        int h = static_cast<int>(seconds) / 3600;
        int m = (static_cast<int>(seconds) % 3600) / 60;
        int s = static_cast<int>(seconds) % 60;
        std::ostringstream oss;
        oss << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m << ":" << std::setw(2) << s;
        return oss.str();
    }

    void finish() {
        std::string bar = "[" + std::string(width, '=') + "]";
        std::cout << "\r" << desc << bar << " 100% " << " [Elapsed: " << formatTime(last_elapsed)
                  << " | Completed]\033[K" << std::endl;
        std::cout.flush();
    }

private:
    int total;
    int width;
    int count;
    float frequency;
    double last_elapsed;
    std::string desc;
    std::chrono::steady_clock::time_point start_time_;
};