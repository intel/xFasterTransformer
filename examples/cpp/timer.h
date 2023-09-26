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
