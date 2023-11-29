#!/bin/bash

# Prepare the dependency lib oneCCL
dir="oneccl"
if [ ! -d "$dir" ]; then
    git clone https://github.com/oneapi-src/oneCCL.git $dir
    if [ -d "$dir" ]; then
        cd $dir
        git checkout 2021.10
        sed -i 's/cpu_gpu_dpcpp/./g' cmake/templates/oneCCLConfig.cmake.in
        mkdir build && cd build
        cmake ..
        make -j install
    fi
fi
