#pragma once
#ifndef VERBOSE_HPP
#define VERBOSE_HPP

#include <cinttypes>
#include <mutex>
#include <stdio.h>
#include <sys/time.h>

static double get_msec() {
    struct timeval time;
    gettimeofday(&time, nullptr);
    return 1e+3 * static_cast<double>(time.tv_sec)
            + 1e-3 * static_cast<double>(time.tv_usec);
}

static int xft_get_verbose() {
    char* xft_verbose_value = getenv("XFT_VERBOSE");
    if (xft_verbose_value != NULL) {
        int value = atoi(xft_verbose_value);
        return value;
    } else {
        return 0; 
    }
}

#endif