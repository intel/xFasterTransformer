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
#ifndef __TIMER_H_
#define __TIMER_H_
#include <sys/time.h>

#include <iostream>
#include <string>

class Timer {
public:
    Timer(bool _auto_print = false, std::string _name = "Timer") : auto_print(_auto_print), name(_name) {
        gettimeofday(&start, NULL);
    }

    ~Timer() {
        if (auto_print) {
            gettimeofday(&end, NULL);
            float duration = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
            std::cout << name << " time: " << duration << " ms" << std::endl;
        }
    }

    float getTime() {
        gettimeofday(&end, NULL);
        float duration = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
        return duration;
    }

private:
    bool auto_print;
    std::string name;
    struct timeval start;
    struct timeval end;
};

#endif
