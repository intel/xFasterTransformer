// Copyright (c) 2023-2024 Intel Corporation
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

#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <stdlib.h>
#include <initializer_list>

#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_ocl.hpp"

inline void memcpy_cpu2gpu(void *dst_gpu, const void *src_cpu, size_t bytes) {
}

inline void memcpy_cpu2gpu(dnnl::memory &dst_gpu_mem, const void *src_cpu) {
    dnnl::engine eng = dst_gpu_mem.get_engine();
    size_t size = dst_gpu_mem.get_desc().get_size();

    if (!src_cpu) throw std::runtime_error("src_cpu is nullptr.");

    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(dst_gpu_mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(dst_gpu_mem);
            auto dst = buffer.get_host_access();
            uint8_t *dst_ptr = dst.get_pointer();
            if (!dst_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                dst_ptr[i] = ((uint8_t *)src_cpu)[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *dst_ptr = (uint8_t *)dst_gpu_mem.get_data_handle();
            if (!dst_ptr)
                throw std::runtime_error("dst_gpu_mem.get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    dst_ptr[i] = ((uint8_t *)src_cpu)[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_ptr, src_cpu, size).wait();
            }
        }
        return;
    }

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(dst_gpu_mem.get_data_handle());
        if (!dst) throw std::runtime_error("dst_gpu_mem.get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            dst[i] = ((uint8_t *)src_cpu)[i];
        return;
    }

    assert(!"memcpy_cpu2gpu: not expected");
}

inline void memcpy_cpu2gpu(void *dst_gpu, const dnnl::memory &src_cpu_mem) {
}

inline void memcpy_cpu2gpu(dnnl::memory &dst_gpu_mem, const dnnl::memory &src_cpu_mem) {
}

inline void memcpy_gpu2cpu(void *dst_cpu, const void *src_gpu, size_t bytes) {
}

inline void memcpy_gpu2cpu(void *dst_cpu, const dnnl::memory &src_gpu_mem) {
    dnnl::engine eng = src_gpu_mem.get_engine();
    size_t size = src_gpu_mem.get_desc().get_size();

    if (!dst_cpu) throw std::runtime_error("dst_cpu is nullptr.");

    bool is_cpu_sycl = (DNNL_CPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::cpu);
    bool is_gpu_sycl = (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
            && eng.get_kind() == dnnl::engine::kind::gpu);
    if (is_cpu_sycl || is_gpu_sycl) {
        auto mkind = dnnl::sycl_interop::get_memory_kind(src_gpu_mem);
        if (mkind == dnnl::sycl_interop::memory_kind::buffer) {
            auto buffer = dnnl::sycl_interop::get_buffer<uint8_t>(src_gpu_mem);
            auto src = buffer.get_host_access();
            uint8_t *src_ptr = src.get_pointer();
            if (!src_ptr)
                throw std::runtime_error("get_pointer returned nullptr.");
            for (size_t i = 0; i < size; ++i)
                ((uint8_t *)dst_cpu)[i] = src_ptr[i];
        } else {
            assert(mkind == dnnl::sycl_interop::memory_kind::usm);
            uint8_t *src_ptr = (uint8_t *)src_gpu_mem.get_data_handle();
            if (!src_ptr)
                throw std::runtime_error("src_gpu_mem.get_data_handle returned nullptr.");
            if (is_cpu_sycl) {
                for (size_t i = 0; i < size; ++i)
                    ((uint8_t *)dst_cpu)[i] = src_ptr[i];
            } else {
                auto sycl_queue
                        = dnnl::sycl_interop::get_queue(dnnl::stream(eng));
                sycl_queue.memcpy(dst_cpu, src_ptr, size).wait();
            }
        }
        return;
    }

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(src_gpu_mem.get_data_handle());
        if (!src) throw std::runtime_error("src_gpu_mem.get_data_handle returned nullptr.");
        for (size_t i = 0; i < size; ++i)
            ((uint8_t *)dst_cpu)[i] = src[i];
        return;
    }

    assert(!"memcpy_gpu2cpu: not expected");
}

inline void memcpy_gpu2cpu(dnnl::memory &dst_cpu_mem, const void *src_gpu) {
}

inline void memcpy_gpu2cpu(dnnl::memory &dst_cpu_mem, const dnnl::memory &src_gpu_mem) {
}