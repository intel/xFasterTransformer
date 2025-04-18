// Copyright (c) 2024 Intel Corporation
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
#include <cstdio>
#include <cstring>
#include <omp.h>
#include "add_util.h"
#include "aligned_type.h"
#include "amx_sgemm_bf16bf16bf16.h"
#include "amx_sgemm_f16f16f16.h"
#include "bfloat16.h"
#include "compile_util.h"
#include "copy_util.h"
#include "decoder_util.h"
#include "gemm_kernel_ext.h"
#include "quantize_util.h"
#include "simple_mem_pool.h"
#include "softmax.h"
#include "thread_util.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

#ifndef ALIGNED_SIZE
#define ALIGNED_SIZE(size, align) (((size) + (align) - 1) / (align) * (align))
#endif

namespace xft {

// T1 could be float, bfloat16_t, float16_t, or std::pair<int8_t *, float *>
template <typename T1, typename T2>
void storeKVCache(T1 &dst, T2 *src, int headSize) {
    if constexpr (std::is_same_v<T1, float *> || std::is_same_v<T1, bfloat16_t *> || std::is_same_v<T1, float16_t *>) {
        xft::copy(dst, src, headSize);
    } else if constexpr (std::is_same_v<T1, std::pair<int8_t *, typename T1::second_type>>) {
        xft::quantize(dst.first, dst.second, src, headSize);
    } else {
        xft::copy(dst.first, src, headSize);
    }
}

template <typename T1, typename T2, typename T3>
void storeKVCache(
        std::tuple<T1, int, T2> &cacheHead, T3 *kv, int pastSeqLen, int inputSeqLen, int headSize, int kvStride) {
    auto [baseAddr, stride, scale] = cacheHead;
    for (int i = 0; i < inputSeqLen; ++i) {
        auto dst = baseAddr + (i + pastSeqLen) * stride;
        auto src = kv + i * kvStride;
        if constexpr (std::is_same_v<T1, int8_t *>) {
            xft::quantize(dst, scale + pastSeqLen + i, src, headSize);
        } else {
            xft::copy(dst, src, headSize);
        }
    }
}

// query: M * headSize, key: N * headSize, score: M * N
// ldq: leading dimension of query; lds: LD of score
// keyMat: key matrix, which is a tuple of (addr, strde, scale)
template <typename T1, typename T2, typename T3, typename T4>
void gemmQK(T1 *query, const std::tuple<T2, int, T3> &keyMat, T4 *score, int M, int N, int headSize, int ldq, int lds) {
    auto A = query;
    auto [B, ldb, scale] = keyMat;
    auto C = score;
    const int K = headSize;
    if constexpr (std::is_same_v<T2, int8_t *>) {
        small_gemm_transb(A, B, scale, C, M, N, K, ldq, ldb, lds);
    } else {
        small_gemm_transb(A, B, C, M, N, K, ldq, ldb, lds);
    }
}

// Compute Score * Value
// score: M * K(keyLen), value: K * headSize, output: M * headSize
// valueMat: value matrix, which is a tuple of (addr, strde, scale)
template <typename T1, typename T2, typename T3, typename T4>
void gemmSV(
        T1 *score, const std::tuple<T2, int, T3> &valueMat, T4 *output, int M, int headSize, int K, int lds, int ldo) {
    auto A = score;
    auto [B, ldv, scale] = valueMat;
    auto C = output;
    const int N = headSize;
    if constexpr (std::is_same_v<T2, int8_t *>) {
        xft::small_gemm(A, B, scale, C, M, N, K, lds, ldv, ldo);
    } else if constexpr (std::is_same_v<T2, const e4m3_t *>) {
        static_assert(std::is_same_v<std::remove_const_t<T1>, bfloat16_t>
                        && std::is_same_v<std::remove_const_t<T4>, bfloat16_t>,
                "Only support BF16 activation for E4M3 gemm.");
        xdnn_small_amx_sgemm_bf16f8bf16_compute_single(M, N, K, (XDNN_BF16 *)A, lds, (XDNN_E4M3 *)B, (XDNN_BF16 *)C,
                ldo, scale, N / 128, 128, 1.0f, 0, nullptr);
    } else {
        xft::small_gemm(A, B, C, M, N, K, lds, ldv, ldo);
    }
}

// T is bfloat16_t or float16_t
// ldb is the K value during packing
template <typename T>
void small_amx_gemm_16bits_compute(int m, int n, int k, T *A, int lda, T *packedB, int ldb, T *C, int ldc, float beta = 0) {
    static_assert(std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>, "AMX gemm only supports BF16/FP16.");

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        xdnn_small_amx_sgemm_bf16bf16bf16_compute(
                m, n, k, (XDNN_BF16 *)A, lda, (XDNN_BF16 *)packedB, ldb, (XDNN_BF16 *)C, ldc, beta);
    } else {
        xdnn_small_amx_sgemm_f16f16f16_compute(
                m, n, k, (XDNN_FP16 *)A, lda, (XDNN_FP16 *)packedB, ldb, (XDNN_FP16 *)C, ldc, beta);
    }
}

template <typename T>
void small_softmax(T *data, float scale, int elements) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, bfloat16_t> || std::is_same_v<T, float16_t>,
            "Unsupported data type for small_softmax");

    if constexpr (std::is_same_v<T, float>) {
        small_softmax_f32(data, scale, elements);
    } else if constexpr (std::is_same_v<T, bfloat16_t>) {
        small_softmax_bf16((XDNN_BF16 *)data, scale, elements);
    } else if constexpr (std::is_same_v<T, float16_t>) {
        DecoderUtil::computeSoftmax(data, scale, elements);
    }
}

// Split heads into threads, in a balanced approach
// tokenSizes: in array of token size for each sample
// threadEndHead: out array of global end head index for each thread
static void splitHeads(const int *tokenSizes, int *threadEndHead, const int batchSize, const int qHeadNum,
        const int mBlockSize, const int threadNum) {
    int computes[batchSize];
    int64_t totalComputes = 0;
    for (int i = 0; i < batchSize; ++i) {
        int blocks = (tokenSizes[i] + mBlockSize - 1) / mBlockSize;
        computes[i] = blocks * (blocks + 1) / 2;
        totalComputes += computes[i];
    }

    totalComputes *= qHeadNum;
    float computesPerTask = 1.0f * totalComputes / threadNum;
    int64_t lastEndCompute = 0; // end compute index for previous thread
    int curBatchIdx = 0; // current search position of batch
    int curHeadInBatch = 0; // current search position of head

    for (int i = 0; i < threadNum; ++i) {
        float targetEndCompute = (i + 1) * computesPerTask;
        int64_t curCompute = lastEndCompute;
        int endHead = (i > 0 ? threadEndHead[i - 1] : 0);
        while (curCompute < targetEndCompute && curBatchIdx < batchSize) {
            // Try all remaining heads of current sample
            auto endCompute = curCompute + (qHeadNum - curHeadInBatch) * computes[curBatchIdx];
            if (endCompute < targetEndCompute) {
                curCompute = endCompute;
                endHead += qHeadNum - curHeadInBatch;
                ++curBatchIdx;
                curHeadInBatch = 0;
            } else {
                int increaseHead = (int)((targetEndCompute - curCompute) / computes[curBatchIdx] + 0.5);
                if (curHeadInBatch + increaseHead > qHeadNum) { // revise if too big
                    increaseHead = qHeadNum - curHeadInBatch;
                }
                curCompute += increaseHead * computes[curBatchIdx];
                endHead += increaseHead;
                curHeadInBatch += increaseHead;
                if (curHeadInBatch >= qHeadNum) {
                    ++curBatchIdx;
                    curHeadInBatch = 0;
                }
                break;
            }
        }

        threadEndHead[i] = endHead;
        lastEndCompute = curCompute;
    }
}

// Self attention while KV cache copy is separated
template <bool FusedPack, typename T, typename Lambda1, typename Lambda2>
void selfAttention_SeparateCopy(T *output, T *query, T *key, T *value, int qHeadNum, int kvHeadNum, int headSize,
        int oStride, int qStride, int kvStride, int batchSize, const int *tokenSizes, const float scale,
        const float *alibiSlopes, int threadNum, const Lambda1 &getKCache, const Lambda2 &getVCache,
        std::function<int(int)> headMap = nullptr) {
    constexpr int mBlockSize = 32;

    int totalTokenSize = 0; // total token size
    int maxTokenSize = 0; // max token size of all inputs
    int offsets[batchSize]; // offset for each input
    int blkEndIndex[batchSize]; // end block index for each input
    for (int i = 0; i < batchSize; ++i) {
        offsets[i] = (i == 0 ? 0 : offsets[i - 1] + tokenSizes[i - 1]);
        auto curBlks = (tokenSizes[i] + mBlockSize - 1) / mBlockSize;
        blkEndIndex[i] = (i == 0 ? curBlks : blkEndIndex[i - 1] + curBlks);
        if (tokenSizes[i] > maxTokenSize) { maxTokenSize = tokenSizes[i]; }
        totalTokenSize += tokenSizes[i];
    }

    const int kPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(maxTokenSize, headSize, 32, 32);
    const int vPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(headSize, maxTokenSize, 32, 32);

    // When packing is fused, allocate packing buffer per thread
    auto totalPackSize
            = FusedPack ? threadNum * (kPackSize + vPackSize) : (batchSize * kvHeadNum) * (kPackSize + vPackSize);

    T *packBuf
            = (T *)SimpleMemPool::instance().getBuffer("kv_packing", totalPackSize * sizeof(T));

    // Copy key/value to cache and pack them
    // If packing is not fused into computing, then pack it here
    if constexpr (!FusedPack) {
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < kvHeadNum; ++i) {
                const int tokens = tokenSizes[b];

                T *packedB = packBuf + (b * kvHeadNum + i) * (kPackSize + vPackSize);
                T *packedV = packedB + kPackSize;

                auto B = key + offsets[b] * kvStride + i * headSize;
                for (int s = 0; s < tokens; ++s) {
                    auto dst = getKCache(b, i, s);
                    auto src = B + s * kvStride;
                    storeKVCache(dst, src, headSize);
                }

                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);

                auto V = value + offsets[b] * kvStride + i * headSize;
                for (int s = 0; s < tokens; ++s) {
                    auto dst = getVCache(b, i, s);
                    auto src = V + s * kvStride;
                    storeKVCache(dst, src, headSize);
                }

                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        false, headSize, tokens, (XDNN_BF16 *)V, kvStride, (XDNN_BF16 *)packedV, vPackSize);
            }
        }
    } else { // just copy
        parallel_for(kvHeadNum * totalTokenSize, [&](int taskIdx) {
            auto gSeqIdx = taskIdx / kvHeadNum; // global seq index
            auto it = std::upper_bound(offsets, offsets + batchSize, gSeqIdx);
            int b = (it == offsets ? 0 : std::distance(offsets, --it)); // batch index
            int i = taskIdx % kvHeadNum; // kv head index
            int s = gSeqIdx - offsets[b]; // seq index inside the batch

            auto B = key + offsets[b] * kvStride + i * headSize;
            auto dst = getKCache(b, i, s);
            storeKVCache(dst, B + s * kvStride, headSize);

            auto V = value + offsets[b] * kvStride + i * headSize;
            dst = getVCache(b, i, s);
            storeKVCache(dst, V + s * kvStride, headSize);
        });
    }

    // Prepare score buffer
    auto maxScoreStride = (maxTokenSize + 31) / 32 * 32;
    T *scores = (T *)SimpleMemPool::instance().getBuffer(
            "qkscore", threadNum * mBlockSize * maxScoreStride * sizeof(T));

    auto totalBlocks = blkEndIndex[batchSize - 1];
    std::pair<int, int> packInfo[threadNum];
    for (int idx = 0; idx < threadNum; ++idx) {
        packInfo[idx].first = -1;
    }

    // Equivalent impl. of below parallel for loop
    // for (int b = 0; b < batchSize; ++b) {
    //     for (int i = 0; i < qHeadNum; ++i) {
    //         for (int mb = 0; mb < blocks[b]; ++mb) {
    parallel_for(qHeadNum * totalBlocks, [&](int taskIdx) {
        // Calculate batch index, head index and block index
        auto it = std::upper_bound(blkEndIndex, blkEndIndex + batchSize, taskIdx / qHeadNum);
        int b = std::distance(blkEndIndex, it); // batch index
        int batchStartIdx = b > 0 ? qHeadNum * blkEndIndex[b - 1] : 0;
        int offset = taskIdx - batchStartIdx;
        int blkSize = b > 0 ? blkEndIndex[b] - blkEndIndex[b - 1] : blkEndIndex[b];
        int i = offset / blkSize; // head index
        int mb = offset % blkSize; // block index along M dimension inside the sample
        int groupNum = qHeadNum / kvHeadNum;

        int tid = omp_get_thread_num();
        int kvHeadIdx = (headMap == nullptr) ? i / groupNum : headMap(i);
        int locationIdx = (FusedPack ? tid : b * kvHeadNum + kvHeadIdx);
        T *packedB = packBuf + locationIdx * (kPackSize + vPackSize);
        T *packedV = packedB + kPackSize;

        const int tokens = tokenSizes[b];
        const int startSeq = mb * mBlockSize;
        const int endSeq = startSeq + mBlockSize < tokens ? startSeq + mBlockSize : tokens;

        // Q * Kᵀ
        int m = endSeq - startSeq;
        int k = headSize;
        int n = tokens;
        int lda = qStride;
        int ldb = kvStride;
        int ldc = (tokens + 31) / 32 * 32;
        auto A = query + (offsets[b] + startSeq) * qStride + i * headSize;
        auto C = scores + omp_get_thread_num() * mBlockSize * maxScoreStride;

        if constexpr (FusedPack) {
            if (packInfo[tid].first != b || packInfo[tid].second != kvHeadIdx) {
                auto B = key + offsets[b] * kvStride + kvHeadIdx * headSize;
                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);
            }
        }

        // Causal mask (either with or without Alibi), use endSeq as N
        small_amx_gemm_16bits_compute(m, endSeq, k, A, lda, packedB, headSize, C, ldc);

#ifdef XFT_DEBUG
        if (b == 0 && i == 0) {
            auto B = key + offsets[b] * kvStride + kvHeadIdx * headSize;
            printf("mnk=%d,%d,%d, ldabc=%d,%d,%d, A[0]=%f, B[0]=%f, packedB[0]=%f\n", m, n, k, lda, ldb, ldc,
                    (float)A[0], (float)B[0], (float)packedB[0]);
            printf("Q * Kᵀ, first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", p[0] * scale, p[1] * scale, p[2] * scale, p[tokens - 3] * scale,
                    p[tokens - 2] * scale, p[tokens - 1] * scale);
        }
#endif

        // Softmax(Q * Kᵀ)
        for (int seq = 0; seq < endSeq - startSeq; ++seq) {
            int elements = startSeq + seq + 1;
            if (alibiSlopes == nullptr) {
                small_softmax(C + seq * ldc, scale, elements);
            } else {
                DecoderUtil::alibiSoftmax(C + seq * ldc, scale, alibiSlopes[i], elements);
            }
            memset(C + seq * ldc + elements, 0, (tokens - elements) * sizeof(T));
        }

#ifdef XFT_DEBUG
        if (b == 0 && i == 0) {
            printf("Softmax(Q * Kᵀ), first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[tokens - 3],
                    (float)p[tokens - 2], (float)p[tokens - 1]);
        }
#endif

        // Softmax(Q * Kᵀ) * V
        std::swap(k, n);
        lda = ldc;
        ldc = oStride;
        A = C;
        C = (T *)output + (offsets[b] + startSeq) * ldc + i * headSize;

        if constexpr (FusedPack) {
            if (packInfo[tid].first != b || packInfo[tid].second != kvHeadIdx) {
                auto V = value + offsets[b] * kvStride + kvHeadIdx * headSize;
                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        false, headSize, tokens, (XDNN_BF16 *)V, kvStride, (XDNN_BF16 *)packedV, vPackSize);
                // Update pack info
                packInfo[tid].first = b;
                packInfo[tid].second = kvHeadIdx;
            }
        }

        small_amx_gemm_16bits_compute(m, n, endSeq, A, lda, packedV, tokens, C, ldc);

#ifdef XFT_DEBUG
        if (b == 0 && i == 0) {
            printf("Softmax(Q * Kᵀ) * V, first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[headSize - 3],
                    (float)p[headSize - 2], (float)p[headSize - 1]);
        }
#endif
    });
}

template <typename T, typename Lambda1, typename Lambda2>
void selfAttention_FusedCopy(T *output, T *query, T *key, T *value, int qHeadNum, int kvHeadNum, int headSize,
        int oStride, int qStride, int kvStride, int batchSize, const int *tokenSizes, const float scale,
        const float *alibiSlopes, int threadNum, const Lambda1 &getKCache, const Lambda2 &getVCache) {
#ifdef XFT_DEBUG
    printf("Q[0]=%f, K[0]=%f, V[0]=%f\n", (float)query[0], (float)key[0], (float)value[0]);
    printf("kvHeadNum=%d, headSize=%d, qStride=%d, kvStride=%d, batchSize=%d\n", kvHeadNum, headSize, qStride, kvStride,
            batchSize);
#endif

    constexpr int mBlockSize = 32;

    if (unlikely(kvHeadNum != qHeadNum)) {
        printf("Error: Incorrect function call when kvHeadNum != qHeadNum.\n");
        exit(-1);
    }

    int maxTokenSize = 0; // max token size of all inputs
    int curOff = 0; // current offset
    int offsets[batchSize]; // offset for each input
    bool allSameSize = true; // if all sequences are in the same size
    int firstTokenSize = tokenSizes[0]; // the size of the first sequence
    for (int i = 0; i < batchSize; ++i) {
        offsets[i] = curOff;
        curOff += tokenSizes[i];
        if (tokenSizes[i] > maxTokenSize) { maxTokenSize = tokenSizes[i]; }
        if (tokenSizes[i] != firstTokenSize) { allSameSize = false; }
    }

    // Prepare buffers (packing buffer and score buffer)
    const int kPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(maxTokenSize, headSize, 32, 32);
    const int vPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(headSize, maxTokenSize, 32, 32);
    T *packBuf = (T *)SimpleMemPool::instance().getBuffer(
            "kv_packing", threadNum * (kPackSize + vPackSize) * sizeof(T));
    int maxScoreStride = (maxTokenSize + 31) / 32 * 32;
    T *scores = (T *)SimpleMemPool::instance().getBuffer(
            "qkscore", threadNum * mBlockSize * maxScoreStride * sizeof(T));

#ifdef XFT_DEBUG
    printf("maxTokenSize=%d, tokenSizes[0]=%d, offsets[0]=%d, kvStride=%d\n", maxTokenSize, tokenSizes[0], offsets[0],
            kvStride);
#endif

    auto oneHeadCompute = [&](int b, int i) {
            int tid = omp_get_thread_num();
            const int tokens = tokenSizes[b];
            const int mBlockNum = (tokens + mBlockSize - 1) / mBlockSize;

            T *packedB = packBuf + tid * (kPackSize + vPackSize);
            T *packedV = packedB + kPackSize;

            // Copy key/value to cache and pack them
            auto B = key + offsets[b] * kvStride + i * headSize;
            for (int s = 0; s < tokens; ++s) {
                auto dst = getKCache(b, i, s);
                storeKVCache(dst, B + s * kvStride, headSize);
            }

            xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                    true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);

            auto V = value + offsets[b] * kvStride + i * headSize;
            for (int s = 0; s < tokens; ++s) {
                auto dst = getVCache(b, i, s);
                storeKVCache(dst, V + s * kvStride, headSize);
            }

            xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                    false, headSize, tokens, (XDNN_BF16 *)V, kvStride, (XDNN_BF16 *)packedV, vPackSize);

            // Compute softmax(Q * Kᵀ) * V, block by block
            for (int mb = 0; mb < mBlockNum; ++mb) {
                const int startSeq = mb * mBlockSize;
                const int endSeq = startSeq + mBlockSize < tokens ? startSeq + mBlockSize : tokens;

                // Q * Kᵀ
                int m = endSeq - startSeq;
                int k = headSize;
                int n = tokens;
                int lda = qStride;
                int ldb = kvStride;
                int ldc = (tokens + 31) / 32 * 32;
                auto A = query + (offsets[b] + startSeq) * qStride + i * headSize;
                auto C = scores + tid * mBlockSize * maxScoreStride;

                small_amx_gemm_16bits_compute(
                        m, endSeq, k, A, lda, packedB, headSize, C, ldc);

#ifdef XFT_DEBUG
                if (b == 0 && i == 0) {
                    printf("mnk=%d,%d,%d, ldabc=%d,%d,%d, A[0]=%f, B[0]=%f, packedB[0]=%f\n", m, n, k, lda, ldb, ldc,
                            (float)A[0], (float)B[0], (float)packedB[0]);
                    printf("Q * Kᵀ, first head:\n");
                    auto p = C;
                    printf("%f, %f, %f ... %f %f %f\n", p[0] * scale, p[1] * scale, p[2] * scale, p[tokens - 3] * scale,
                            p[tokens - 2] * scale, p[tokens - 1] * scale);
                }
#endif

                // Softmax(Q * Kᵀ)
                for (int seq = 0; seq < endSeq - startSeq; ++seq) {
                    int elements = startSeq + seq + 1;
                    if (alibiSlopes == nullptr) {
                        small_softmax(C + seq * ldc, scale, elements);
                    } else {
                        DecoderUtil::alibiSoftmax(C + seq * ldc, scale, alibiSlopes[i], elements);
                    }
                    memset(C + seq * ldc + elements, 0, (tokens - elements) * sizeof(T));
                }

#ifdef XFT_DEBUG
                if (b == 0 && i == 0) {
                    printf("Softmax(Q * Kᵀ), first head:\n");
                    auto p = C;
                    printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[tokens - 3],
                            (float)p[tokens - 2], (float)p[tokens - 1]);
                }
#endif

                // Softmax(Q * Kᵀ) * V
                std::swap(k, n);
                lda = ldc;
                ldc = oStride;
                A = C;
                C = (T *)output + (offsets[b] + startSeq) * ldc + i * headSize;

                small_amx_gemm_16bits_compute(m, n, endSeq, A, lda, packedV, tokens, C, ldc);

#ifdef XFT_DEBUG
                if (b == 0 && i == 0) {
                    printf("Softmax(Q * Kᵀ) * V, first head:\n");
                    auto p = C;
                    printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[headSize - 3],
                            (float)p[headSize - 2], (float)p[headSize - 1]);
                }
#endif
            } // end for mb
        };

    if (allSameSize) {
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int h = 0; h < qHeadNum; ++h) {
                oneHeadCompute(b, h);
            }
        }
    }
    else {
        int threadEndHead[threadNum];
        splitHeads(tokenSizes, threadEndHead, batchSize, qHeadNum, mBlockSize, threadNum);

        parallel_for(threadNum, [&](int taskIdx) {
            int startIdx = (taskIdx == 0 ? 0 : threadEndHead[taskIdx - 1]);
            int endIdx = threadEndHead[taskIdx];
            for (int idx = startIdx; idx < endIdx; ++idx) {
                int b = idx / qHeadNum;
                int h = idx % qHeadNum;
                oneHeadCompute(b, h);
            }
        });
    }
}

/**
 * For DeepSeek like MLA
 * @param output output tensor
 * @param query query tensor, include nope and rope part
 * @param keyRope key tensor for rope part
 * @param keyValue key tensor for nope part & value tensor
 * @param headNum head number
 * @param nopeDim nope dimension
 * @param ropeDim rope dimension
 * @param vHeadDim value head dimension
 * @param oStride output stride
 * @param qStride query stride
 * @param krStride key rope stride
 * @param kvStride key (nope part) and value stride
 */
template <bool FusedPack, typename T>
void selfAttention_MLA(T *output, T *query, T *keyRope, T *keyValue, int headNum, int nopeDim, int ropeDim,
        int vHeadDim, int oStride, int qStride, int krStride, int kvStride, int batchSize, const int *tokenSizes,
        const float scale, int threadNum) {
    constexpr int mBlockSize = 32;

    int totalTokenSize = 0; // total token size
    int maxTokenSize = 0; // max token size of all inputs
    int offsets[batchSize]; // offset for each input
    int blkEndIndex[batchSize]; // end block index for each input
    for (int i = 0; i < batchSize; ++i) {
        offsets[i] = (i == 0 ? 0 : offsets[i - 1] + tokenSizes[i - 1]);
        auto curBlks = (tokenSizes[i] + mBlockSize - 1) / mBlockSize;
        blkEndIndex[i] = (i == 0 ? curBlks : blkEndIndex[i - 1] + curBlks);
        if (tokenSizes[i] > maxTokenSize) { maxTokenSize = tokenSizes[i]; }
        totalTokenSize += tokenSizes[i];
    }

    const int kNopePackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(maxTokenSize, nopeDim, 32, 32);
    const int kRopePackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(maxTokenSize, ropeDim, 32, 32);
    const int vPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(vHeadDim, maxTokenSize, 32, 32);

    // When packing is fused, allocate packing buffer per thread
    auto totalPackSize
            = FusedPack ? threadNum * (kNopePackSize + vPackSize) : (batchSize * headNum) * (kNopePackSize + vPackSize);
    totalPackSize += batchSize * kRopePackSize;

    // The packing buffer is organized as:
    // BS0_[kNope_h0, value_h0, kNope_h1, value_h1, ...]
    // BS1_[kNope_h0, value_h0, kNope_h1, value_h1, ...]
    // ...
    // BS0_kRope, BS1_kRope, ...
    T *packBuf = (T *)SimpleMemPool::instance().getBuffer("kv_packing", totalPackSize * sizeof(T));

    // Pack keyRope, keyNope and value
    // If packing is not fused into computing, then pack it here
    if constexpr (!FusedPack) {
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            // keyRope is taken as the last head
            for (int i = 0; i < headNum + 1; ++i) {
                const int tokens = tokenSizes[b];

                if (i == headNum) { // pack keyRope for batch b
                    T *packedRope = packBuf + totalPackSize - batchSize * kRopePackSize + b * kRopePackSize;
                    auto B = keyRope + offsets[b] * krStride;
                    xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                            true, tokens, ropeDim, (XDNN_BF16 *)B, krStride, (XDNN_BF16 *)packedRope, kRopePackSize);
                    continue;
                }

                T *packedB = packBuf + (b * headNum + i) * (kNopePackSize + vPackSize);
                T *packedV = packedB + kNopePackSize;

                auto B = keyValue + offsets[b] * kvStride + i * (nopeDim + vHeadDim);

                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        true, tokens, nopeDim, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kNopePackSize);

                auto V = B + nopeDim;

                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        false, vHeadDim, tokens, (XDNN_BF16 *)V, kvStride, (XDNN_BF16 *)packedV, vPackSize);
            }
        }
    } else { // Only pack keyRope
        for (int b = 0; b < batchSize; ++b) {
            const int tokens = tokenSizes[b];
            T *packedRope = packBuf + totalPackSize - batchSize * kRopePackSize + b * kRopePackSize;
            auto B = keyRope + offsets[b] * krStride;
            xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                    true, tokens, ropeDim, (XDNN_BF16 *)B, krStride, (XDNN_BF16 *)packedRope, kRopePackSize);
            continue;
        }
    }

    // Prepare score buffer
    auto maxScoreStride = (maxTokenSize + 31) / 32 * 32;
    T *scores = (T *)SimpleMemPool::instance().getBuffer(
            "qkscore", threadNum * mBlockSize * maxScoreStride * sizeof(T));

    auto totalBlocks = blkEndIndex[batchSize - 1];
    std::pair<int, int> packInfo[threadNum];
    for (int idx = 0; idx < threadNum; ++idx) {
        packInfo[idx].first = -1;
    }

    // Equivalent impl. of below parallel for loop
    // for (int b = 0; b < batchSize; ++b) {
    //     for (int i = 0; i < headNum; ++i) {
    //         for (int mb = 0; mb < blocks[b]; ++mb) {
    parallel_for(headNum * totalBlocks, [&](int taskIdx) {
        // Calculate batch index, head index and block index
        auto it = std::upper_bound(blkEndIndex, blkEndIndex + batchSize, taskIdx / headNum);
        int b = std::distance(blkEndIndex, it); // batch index
        int batchStartIdx = b > 0 ? headNum * blkEndIndex[b - 1] : 0;
        int offset = taskIdx - batchStartIdx;
        int blkSize = b > 0 ? blkEndIndex[b] - blkEndIndex[b - 1] : blkEndIndex[b];
        int i = offset / blkSize; // head index
        int mb = offset % blkSize; // block index along M dimension inside the sample

        int tid = omp_get_thread_num();
        int kvHeadIdx = i;
        int locationIdx = (FusedPack ? tid : b * headNum + kvHeadIdx);
        T *packedB = packBuf + locationIdx * (kNopePackSize + vPackSize);
        T *packedV = packedB + kNopePackSize;

        const int tokens = tokenSizes[b];
        const int startSeq = mb * mBlockSize;
        const int endSeq = startSeq + mBlockSize < tokens ? startSeq + mBlockSize : tokens;

        // Q * Kᵀ (Nope)
        int m = endSeq - startSeq;
        int k = nopeDim;
        int n = tokens;
        int lda = qStride;
        int ldb = kvStride;
        int ldc = (tokens + 31) / 32 * 32;
        auto A = query + (offsets[b] + startSeq) * qStride + i * (nopeDim + ropeDim);
        auto C = scores + tid * mBlockSize * maxScoreStride;

        if constexpr (FusedPack) {
            if (packInfo[tid].first != b || packInfo[tid].second != kvHeadIdx) {
                auto B = keyValue + offsets[b] * kvStride + kvHeadIdx * (nopeDim + vHeadDim);
                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        true, tokens, nopeDim, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kNopePackSize);
            }
        }

        // Causal mask, use endSeq as N
        small_amx_gemm_16bits_compute(m, endSeq, k, A, lda, packedB, nopeDim, C, ldc);

        // Q * Kᵀ (rope), accumulate to C
        k = ropeDim;
        lda = qStride;
        A = A + nopeDim;
        auto packedRope = packBuf + totalPackSize - batchSize * kRopePackSize + b * kRopePackSize;
        small_amx_gemm_16bits_compute(m, endSeq, k, A, lda, packedRope, ropeDim, C, ldc, 1.0f);

#ifdef XFT_DEBUG
        if (b == 0 && i == 0) {
            printf("Q * Kᵀ, first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", p[0] * scale, p[1] * scale, p[2] * scale, p[tokens - 3] * scale,
                    p[tokens - 2] * scale, p[tokens - 1] * scale);
        }
#endif

        // Softmax(Q * Kᵀ)
        // Note: need to access even elements which is required by AMX instructions
        // for example, for 5 tokens, need to make sure the 6th is not NaN
        int reqElems = (tokens + 1) / 2 * 2;
        for (int seq = 0; seq < endSeq - startSeq; ++seq) {
            int elements = startSeq + seq + 1;
            small_softmax(C + seq * ldc, scale, elements);
            memset(C + seq * ldc + elements, 0, (reqElems - elements) * sizeof(T));
        }

#ifdef XFT_DEBUG
        if (b == 0 && i == 0) {
            printf("Softmax(Q * Kᵀ), first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[tokens - 3],
                    (float)p[tokens - 2], (float)p[tokens - 1]);
        }
#endif

        // Softmax(Q * Kᵀ) * V
        k = tokens;
        n = vHeadDim;
        lda = ldc;
        ldc = oStride;
        A = C;
        C = (T *)output + (offsets[b] + startSeq) * ldc + i * vHeadDim;

        if constexpr (FusedPack) {
            if (packInfo[tid].first != b || packInfo[tid].second != kvHeadIdx) {
                auto V = keyValue + offsets[b] * kvStride + kvHeadIdx * (nopeDim + vHeadDim) + nopeDim;
                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        false, vHeadDim, tokens, (XDNN_BF16 *)V, kvStride, (XDNN_BF16 *)packedV, vPackSize);
                // Update pack info
                packInfo[tid].first = b;
                packInfo[tid].second = kvHeadIdx;
            }
        }

        small_amx_gemm_16bits_compute(m, n, endSeq, A, lda, packedV, tokens, C, ldc);

#ifdef XFT_DEBUG
        if (b == 0 && i == 0) {
            printf("Softmax(Q * Kᵀ) * V, first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[vHeadDim - 3],
                    (float)p[vHeadDim - 2], (float)p[vHeadDim - 1]);
        }
#endif
    });
}

template <typename T, typename Lambda1, typename Lambda2>
void selfAttention(T *output, T *query, T *key, T *value, int qHeadNum, int kvHeadNum, int headSize, int oStride,
        int qStride, int kvStride, int batchSize, const int *tokenSizes, const float scale, const float *alibiSlopes,
        int threadNum, const Lambda1 &getKCache, const Lambda2 &getVCache, std::function<int(int)> headMap = nullptr) {
    // Revise threadNum if not set
    if (unlikely(threadNum <= 0)) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) { threadNum = omp_get_num_threads(); }
        }
    }

    float efficiency = (batchSize * qHeadNum) / (std::ceil(1.0f * batchSize * qHeadNum / threadNum) * threadNum);

    // TODO: .9f is the estimation, change it when have more data
    if (kvHeadNum == qHeadNum && efficiency > .9f) {
        selfAttention_FusedCopy(output, query, key, value, qHeadNum, kvHeadNum, headSize, oStride, qStride, kvStride,
                batchSize, tokenSizes, scale, alibiSlopes, threadNum, getKCache, getVCache);
    } else {
        if (batchSize * kvHeadNum >= threadNum) {
            selfAttention_SeparateCopy<true>(output, query, key, value, qHeadNum, kvHeadNum, headSize, oStride, qStride,
                    kvStride, batchSize, tokenSizes, scale, alibiSlopes, threadNum, getKCache, getVCache, headMap);
        } else {
            selfAttention_SeparateCopy<false>(output, query, key, value, qHeadNum, kvHeadNum, headSize, oStride, qStride,
                    kvStride, batchSize, tokenSizes, scale, alibiSlopes, threadNum, getKCache, getVCache, headMap);
        }
    }
}

template <typename T, typename U>
inline std::pair<T, T> balance211(T n, U team, U tid) {
    T n_start;
    T n_mine;

    if (team <= 1 || n == 0) {
        n_start = 0;
        n_mine = n;
    } else {
        // team = T1 + T2
        // n = T1*n1 + T2*n2  (n1 - n2 = 1)
        T n1 = (n + team - 1) / team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_mine = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? (T)tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    return std::make_pair(n_start, n_mine);
}

/**
 * @brief Cross attention with sharded heads (When #heads is few, need to split each head to use more resources)
 * @note KV head number is not here because it is handled by getKHead and getVHead by providing the query head index
 * @tparam T1 Query and output data type
 * @param output Output tensor
 * @param query Query tensor
 * @param attnMask Attention mask
 * @param inputSeqLen Input sequence length
 * @param presentSeqLen Present sequence length (input sequence len + past sequence len)
 * @param qHeadNum Query head number
 * @param kvHeadNum KV head number
 * @param headSize Head size
 * @param oStride Output stride
 * @param qStride Query stride
 * @param kvStride KV stride
 * @param batchSize Batch size
 * @param scale Scale factor
 * @param threadNum Thread number
*/
template <typename T1, typename KVCacheT, typename Lambda1, typename Lambda2, typename LambdaM>
void crossAttnShardedHead(T1 *output, const T1 *query, int inputSeqLen, int presentSeqLen, int qHeadNum, int headSize,
        int oStride, int qStride, int batchSize, const float scale, int threadNum, const Lambda1 &getKHead,
        const Lambda2 &getVHead, const LambdaM &getMask) {

    const int responsibleHeads = qHeadNum;

    int N = presentSeqLen;
    int splits = threadNum / (batchSize * responsibleHeads);

    REQUIRES(splits > 1, "Do not call me when splits=%d, threadNum=%d, batchSize=%d, heads=%d\n", splits, threadNum,
            batchSize, responsibleHeads);

    // AVX512 is used and the case where headSize is not multiple of 16 hasn't been taken into account
    REQUIRES(headSize % 16 == 0, "Head size (%d) is not supported.", headSize);

    int NB = (N + splits - 1) / splits; // Max block size in one thread

    // The first element is for max, the second is for sum, the third is for finish flag
    // [max(xi), sum(exp(xi)), finish_tag] for each thread
    // totalTasks <= threadNum, thus we could not synchronize among threads
    int totalTasks = batchSize * responsibleHeads * splits;
    AlignedType<std::tuple<float, float, float>, 32> splitInfo[totalTasks];
    for (int i = 0; i < totalTasks; ++i) {
        std::get<1>(splitInfo[i].data) = 0;
        std::get<2>(splitInfo[i].data) = 0;
    }

    // In each thread, sizeof(score_buf) = inputSeqLen * NB, sizeof(output) = inputSeqLen * headSize
    // In tmpBuf, score_buf and output are stored in a continuous memory block, the layout is:
    // (score_buf, output) for thread 1
    // (score_buf, output) for thread 2
    // ...
    size_t sizePerThread = inputSeqLen * (NB + headSize);
    sizePerThread = ALIGNED_SIZE(sizePerThread, 16);
    size_t totalBufSize = threadNum * sizePerThread * sizeof(float);
    float *tmpBuf = (float *)SimpleMemPool::instance().getBuffer("tmpBuf", totalBufSize);

#pragma omp parallel for collapse(3)
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < responsibleHeads; ++i) {
            for (int s = 0; s < splits; ++s) {
                int headStartIdx = b * responsibleHeads * splits + i * splits;
                int threadIdx = b * responsibleHeads * splits + i * splits + s;

                // Q * K
                auto myTask = balance211(N, splits, s);
                int nOff = myTask.first;
                auto keyMatInfo = getKHead(b, i);
                int m = inputSeqLen;
                int k = headSize;
                int n = myTask.second;
                int lda = qStride;
                int ldb = std::get<1>(keyMatInfo);
                int ldc = n;
                auto A = query + b * inputSeqLen * lda + i * headSize;
                auto B = std::get<0>(keyMatInfo) + nOff * ldb;
                auto C = tmpBuf + threadIdx * sizePerThread;

                const int queryLen = inputSeqLen;
                const int keyLen = N;

                if constexpr (std::is_same_v<KVCacheT, int8_t>) { // INT8 KV cache
                    auto bScale = std::get<2>(keyMatInfo);
                    small_gemm_transb(A, B, bScale, C, m, n, k, lda, ldb, ldc);
                } else {
                    small_gemm_transb(A, B, C, m, n, k, lda, ldb, ldc);
                }

                // Softmax and the stats info
                auto mask = getMask(b, i, queryLen, keyLen) + nOff;
                auto smInfo = DecoderUtil::softmaxWithStats(C, mask, n, scale);
                std::get<0>(splitInfo[threadIdx].data) = smInfo.first;
                std::get<1>(splitInfo[threadIdx].data) = smInfo.second;

                // Softmax * V
                {
                    auto valueMatInfo = getVHead(b, i);
                    int k = n;
                    int n = headSize;
                    int lda = ldc;
                    int ldb = std::get<1>(valueMatInfo);
                    int ldc = n;
                    float *A = C;
                    auto B = std::get<0>(valueMatInfo) + nOff * ldb;
                    auto C = tmpBuf + threadIdx * sizePerThread + m * NB;

                    if constexpr (std::is_same_v<KVCacheT, int8_t>) {
                        auto bScale = std::get<2>(valueMatInfo) + nOff;
                        xft::small_gemm(A, B, bScale, C, m, n, k, lda, ldb, ldc);
                    } else {
                        xft::small_gemm(A, B, C, m, n, k, lda, ldb, ldc);
                    }
                }

                std::get<2>(splitInfo[threadIdx].data) = 1; // set finished flag

                // Wait for all threads to finish and reduce the result
                // Firstly get the max value, and then revise the value by considering the factor on numerator and denominator
                if (s == 0) {
                    float realMax = std::get<0>(splitInfo[threadIdx].data);
                    for (int idx = headStartIdx + 1; idx < headStartIdx + splits; ++idx) {
                        while (std::get<2>(splitInfo[idx].data) == 0) {
                            _mm_pause();
                        }
                        if (std::get<0>(splitInfo[idx].data) > realMax) { realMax = std::get<0>(splitInfo[idx].data); }
                    }

                    float realSum = 0;
                    for (int idx = headStartIdx; idx < headStartIdx + splits; ++idx) {
                        float splitMax = std::get<0>(splitInfo[idx].data);
                        float splitSum = std::get<1>(splitInfo[idx].data);
                        float revFactor = std::exp(splitMax - realMax); // revise factor
                        std::get<2>(splitInfo[idx].data) = revFactor; // borrow finish flag for revise factor
                        realSum += splitSum * revFactor;
                    }

                    // Accumulate in float
                    float acc[headSize];
                    memset(acc, 0, headSize * sizeof(float));

                    for (int idx = headStartIdx; idx < headStartIdx + splits; ++idx) {
                        float splitMax = std::get<0>(splitInfo[idx].data);
                        float splitSum = std::get<1>(splitInfo[idx].data);
                        float revFactor = std::get<2>(splitInfo[idx].data);

                        float factor = revFactor * (splitSum / realSum);
                        auto vfactor = xft::set_avx512(factor);

                        // Note: here need to make sure the right address of the buffer
                        float *p = tmpBuf + idx * sizePerThread + m * NB;
                        for (int off = 0; off < headSize; off += 16) {
                            auto vacc = xft::load_avx512(acc + off);
                            vacc = vacc + xft::load_avx512(p + off) * vfactor;
                            xft::store_avx512(acc + off, 0xffff, vacc);
                        }
                    }

                    // Store the result (acc -> result)
                    T1 *pResult = output + b * inputSeqLen * oStride + i * headSize;
                    for (int off = 0; off < headSize; off += 16) {
                        auto vacc = xft::load_avx512(acc + off);
                        xft::store_avx512(pResult + off, 0xffff, vacc);
                    }
                }
            } // end for s
        } // end for i
    } // end for b
}

/**
 * @brief Cross attention with head granularity (including copy key/value to KV Cache)
 * @note if causal = True, attnMask is not used
 * @tparam T Data type
 * @tparam KVCacheT KV cache data type
 * @tparam Lambda1 Lambda to get head of cached keys, return tuple of (addr, stride, scale)
 * @tparam Lambda2 Lambda to get head of cached values
 * @param output Output tensor
 * @param query Query tensor, in shape of (input_seqlen, head_num, head_size)
 * @param key Key tensor (not include cached keys)
 * @param value Value tensor (not include cached values)
 * @param qHeadNum Query head number
 * @param kvHeadNum KV head number
 * @param headSize Head size
 * @param oStride Output stride
 * @param qStride Query stride
 * @param kvStride KV stride
 * @param batchSize Batch size
 * @param inputSeqLens Input sequence lengths
 * @param pastSeqLens Past sequence lengths
 * @param causal Whether causal attention
 * @param alibiSlopes Alibi slopes
 * @param scale Scale factor for softmax
 * @param threadNum Thread number
 * @param getKHead Lambda to get head of cached keys
 * @param getVHead Lambda to get head of cached values
 */
template <typename T, typename KVCacheT, typename Lambda1, typename Lambda2>
void crossAttnByHead(T *output, const T *query, const T *key, const T *value, int qHeadNum, int kvHeadNum, int headSize,
        int oStride, int qStride, int kvStride, int batchSize, const int *inputSeqLens, const int *pastSeqLens,
        bool causal, const float scale, const float *alibiSlopes, int threadNum, const Lambda1 &getKHead,
        const Lambda2 &getVHead, std::function<int(int)> headMap = nullptr) {

    int responsibleHeads = qHeadNum;
    int groupNum = qHeadNum / kvHeadNum;

    // To get row offset for each sample/sequence inside the batch, and prepare score buffer
    int inputOffsets[batchSize];
    size_t scoreSizePerThr = 0;
    for (int i = 0; i < batchSize; ++i) {
        scoreSizePerThr = std::max(scoreSizePerThr, (size_t)inputSeqLens[i] * (inputSeqLens[i] + pastSeqLens[i]));
        inputOffsets[i] = (i > 0 ? inputOffsets[i - 1] + inputSeqLens[i - 1] : 0);
    }

    scoreSizePerThr = ALIGNED_SIZE(scoreSizePerThr, 16);
    size_t scoreSize = scoreSizePerThr * threadNum;
    float *scoreBuf = (float *)SimpleMemPool::instance().getBuffer("scoreBuf", sizeof(float) * scoreSize);

    int outerHeadNum = kvHeadNum;
    int innerHeadNum = groupNum;
    if (headMap != nullptr) {
        outerHeadNum = 1;
        innerHeadNum = responsibleHeads;
    }

#pragma omp parallel for collapse(3)
    for (int outh = 0; outh < outerHeadNum; ++outh) {
        for (int b = 0; b < batchSize; ++b) {
            for (int inh = 0; inh < innerHeadNum; ++inh) {
                int i = outh * innerHeadNum + inh;
                // Copy current key to cached keys (if needed)
                int kvHdx = (headMap == nullptr) ? i / groupNum : headMap(i);
                auto keyMatInfo = getKHead(b, kvHdx);
                auto valueMat = getVHead(b, kvHdx);
                bool bCopyCache = (headMap == nullptr) ? (i % groupNum == 0) : (i == 0 || headMap(i - 1) != headMap(i));

                // Q * K
                auto Q = query + inputOffsets[b] * qStride + i * headSize;
                auto S = scoreBuf + omp_get_thread_num() * scoreSizePerThr;

                const int queryLen = inputSeqLens[b];
                const int keyLen = pastSeqLens[b] + inputSeqLens[b];

                if (bCopyCache) {
                    int m = queryLen;
                    int n = keyLen;
                    int lda = qStride;
                    int ldc = keyLen;

                    // Copy to Key cache and compute Query * Key
                    auto src = key + inputOffsets[b] * kvStride + kvHdx * headSize;
                    storeKVCache(keyMatInfo, src, pastSeqLens[b], inputSeqLens[b], headSize, kvStride);

                    gemmQK(Q, keyMatInfo, S, m, n, headSize, lda, ldc);
                } else {
                    // Note: when KV cache is not copied by me, then 2 times gemm to avoid synchronization
                    int m = queryLen;
                    int n = pastSeqLens[b];
                    int lda = qStride;
                    int ldc = keyLen;
                    gemmQK(Q, keyMatInfo, S, m, n, headSize, lda, ldc);

                    int ldb = kvStride;
                    auto B = key + inputOffsets[b] * kvStride + kvHdx * headSize;
                    small_gemm_transb(Q, B, S + n, m, inputSeqLens[b], headSize, lda, ldb, ldc);
                }

                // Softmax(Q * K)
                for (int seq = 0; seq < queryLen; ++seq) {
                    int elements = pastSeqLens[b] + seq + 1;
                    if (alibiSlopes == nullptr) {
                        small_softmax(S + seq * keyLen, scale, elements);
                    } else {
                        DecoderUtil::alibiSoftmax(S + seq * keyLen, scale, alibiSlopes[i], elements);
                    }
                    if (keyLen > elements) { memset(S + seq * keyLen + elements, 0, (keyLen - elements) * sizeof(float)); }
                }

                // Softmax * V
                if (bCopyCache) {
                    // Copy current value to cached values
                    auto src = value + inputOffsets[b] * kvStride + kvHdx * headSize;
                    storeKVCache(valueMat, src, pastSeqLens[b], inputSeqLens[b], headSize, kvStride);

                    int m = queryLen;
                    auto result = output + inputOffsets[b] * oStride + i * headSize;
                    gemmSV(S, valueMat, result, m, headSize, keyLen, keyLen, oStride);
                } else {
                    // Note: when KV cache is not copied by me, then 2 times gemm to avoid synchronization
                    int m = queryLen;
                    float f32Out[m * headSize]; // accumulate in FP32
                    gemmSV(S, valueMat, f32Out, m, headSize, pastSeqLens[b], keyLen, headSize);

                    auto B = value + inputOffsets[b] * kvStride + kvHdx * headSize;
                    small_gemm(S + pastSeqLens[b], B, f32Out, m, headSize, m, keyLen, kvStride, headSize, true);

                    // f32Out -> output
                    auto result = output + inputOffsets[b] * oStride + i * headSize;
                    for (int t = 0; t < m; ++t) {
                        xft::copy(result + t * oStride, f32Out + t * headSize, headSize);
                    }
                }
            } // end for inner head
        } // end for b
    } // end for outer head
}

// Cross attention for MLA, modified from crossAttnByHead
// Note: 
//     1) keyRope: array of pointers, each element is a pointer to the keyRope of a sample
//     2) kvDowns: array of pointers, each element is a pointer to the KV_Down value of a sample
//     3) wkt: packed weight for key up in transpose format, thus the shape is 128x512
//     4) wvp: packed weight for value up, shape is 512x128
//     5) wScales: weight scale for wkt and wvt, each has 4 scales, thus 8 elements for each head
template <typename T, typename WT>
void crossAttnByHead_DS(T *output, const T *query, const T **keyRope, const T **kvDowns, const WT *wkt, const WT *wvp,
        float *wScales, int headNum, int nopeDim, int ropeDim, int vHeadDim, int oStride, int qStride, int krStride,
        int kvStride /* unsed */, int batchSize, const int *inputSeqLens, const int *pastSeqLens, const float scale,
        int threadNum) {
    const int qkHeadDim = nopeDim + ropeDim;

    // To get row offset for each sample/sequence inside the batch, and prepare score buffer
    int qOffsets[batchSize];
    int kvOffsets[batchSize];
    size_t scoreSizePerThr = 0;
    for (int i = 0; i < batchSize; ++i) {
        scoreSizePerThr = std::max(scoreSizePerThr, (size_t)inputSeqLens[i] * (inputSeqLens[i] + pastSeqLens[i]));
        qOffsets[i] = (i > 0 ? qOffsets[i - 1] + inputSeqLens[i - 1] : 0);
        kvOffsets[i] = (i > 0 ? kvOffsets[i - 1] + (pastSeqLens[i - 1] + inputSeqLens[i - 1]) : 0);
    }

    scoreSizePerThr *= 2; // 2 times for Q * K (nope and rope), can be 1 after optimization
    scoreSizePerThr = ALIGNED_SIZE(scoreSizePerThr, 16);
    size_t scoreSize = scoreSizePerThr * threadNum;
    float *scoreBuf = (float *)SimpleMemPool::instance().getBuffer("scoreBuf", sizeof(float) * scoreSize);

#pragma omp parallel for collapse(2)
    for (int h = 0; h < headNum; ++h) {
        for (int b = 0; b < batchSize; ++b) {
            const int queryLen = inputSeqLens[b];
            const int keyLen = pastSeqLens[b] + inputSeqLens[b];

            // Temporary buffer for Q*Wₖᵀ and S*Down
            bfloat16_t tmp[queryLen * 512];

            // Q*Kᵀ, Q*Kᵀ=Q1*(DWₖ)ᵀ+Q2*Krᵀ=Q1*Wₖᵀ*Dᵀ+Q2*Krᵀ
            auto S = scoreBuf + omp_get_thread_num() * scoreSizePerThr;
            {
                auto Q = query + qOffsets[b] * qStride + h * qkHeadDim;
                int m = queryLen;
                int n = keyLen;
                int lda = qStride;
                int ldc = keyLen;

                // nope part
                auto wMatInfo = std::make_tuple(wkt + h * 128 * 512, 512, wScales + h * 8);
                gemmSV(Q, wMatInfo, tmp, m, 512, 128, lda, 512); // borrow gemmSV for Q*Wₖᵀ
                auto dMatInfo = std::make_tuple(kvDowns[b], 512, 1.0f);
                gemmQK(tmp, dMatInfo, S, m, n, 512, 512, ldc);

                // rope part
                Q += nopeDim;
                auto keyMatInfo = std::make_tuple(keyRope[b], krStride, 1.0f);
                gemmQK(Q, keyMatInfo, S + m * n, m, n, ropeDim, lda, ldc);

                // merge/add nope and rope
                xft::addto(S, S + m * n, m * n);
            }

            // Softmax(Q * K)
            for (int seq = 0; seq < queryLen; ++seq) {
                int elements = pastSeqLens[b] + seq + 1;
                small_softmax(S + seq * keyLen, scale, elements);
                if (keyLen > elements) { memset(S + seq * keyLen + elements, 0, (keyLen - elements) * sizeof(float)); }
            }

            // Softmax*V, S*(D*Wᵥ)=(S*D)*Wᵥ
            {
                bfloat16_t *sd = tmp;
                auto result = output + qOffsets[b] * oStride + h * vHeadDim;
                small_gemm(S, kvDowns[b], sd, queryLen, 512, keyLen, keyLen, 512, 512);

                auto valueMat = std::make_tuple(
                        wvp + 512 * 128 * 2 * h, 128, wScales + h * 8 + 4); // wvp: 512x128, key and value are interleaved
                gemmSV(sd, valueMat, result, queryLen, 128, 512, keyLen, oStride);
            }
        } // end for b
    } // end for outer head
}

// Cross attention for MLA, modified from crossAttnByHead
// Note: keyRope is an array of pointers, each element is a pointer to the keyRope of a sample
template <typename T>
void crossAttnByHead_DS(T *output, const T *query, const T **keyRope, const T *keyValue, int headNum, int nopeDim,
        int ropeDim, int vHeadDim, int oStride, int qStride, int krStride, int kvStride, int batchSize,
        const int *inputSeqLens, const int *pastSeqLens, const float scale, int threadNum) {
    const int qkHeadDim = nopeDim + ropeDim;

    // Split keyValue into keyNope and value
    const T *keyNope = keyValue;
    const T *value = keyValue + nopeDim;

    // To get row offset for each sample/sequence inside the batch, and prepare score buffer
    int qOffsets[batchSize];
    int kvOffsets[batchSize];
    size_t scoreSizePerThr = 0;
    for (int i = 0; i < batchSize; ++i) {
        scoreSizePerThr = std::max(scoreSizePerThr, (size_t)inputSeqLens[i] * (inputSeqLens[i] + pastSeqLens[i]));
        qOffsets[i] = (i > 0 ? qOffsets[i - 1] + inputSeqLens[i - 1] : 0);
        kvOffsets[i] = (i > 0 ? kvOffsets[i - 1] + (pastSeqLens[i - 1] + inputSeqLens[i - 1]) : 0);
    }

    scoreSizePerThr *= 2; // 2 times for Q * K (nope and rope), can be 1 after optimization
    scoreSizePerThr = ALIGNED_SIZE(scoreSizePerThr, 16);
    size_t scoreSize = scoreSizePerThr * threadNum;
    float *scoreBuf = (float *)SimpleMemPool::instance().getBuffer("scoreBuf", sizeof(float) * scoreSize);

#pragma omp parallel for collapse(2)
    for (int h = 0; h < headNum; ++h) {
        for (int b = 0; b < batchSize; ++b) {
            const int queryLen = inputSeqLens[b];
            const int keyLen = pastSeqLens[b] + inputSeqLens[b];

            // Q * K
            auto S = scoreBuf + omp_get_thread_num() * scoreSizePerThr;
            {
                auto Q = query + qOffsets[b] * qStride + h * qkHeadDim;
                int m = queryLen;
                int n = keyLen;
                int lda = qStride;
                int ldc = keyLen;

                // nope part
                const T *K = keyNope + kvOffsets[b] * kvStride + h * (nopeDim + vHeadDim);
                auto keyMatInfo = std::make_tuple(K, kvStride, 1.0f);
                gemmQK(Q, keyMatInfo, S, m, n, nopeDim, lda, ldc);

                // rope part
                Q += nopeDim;
                K = keyRope[b];
                keyMatInfo = std::make_tuple(K, krStride, 1.0f);
                gemmQK(Q, keyMatInfo, S + m * n, m, n, ropeDim, lda, ldc);

                // merge/add nope and rope
                xft::addto(S, S + m * n, m * n);
            }

            // Softmax(Q * K)
            for (int seq = 0; seq < queryLen; ++seq) {
                int elements = pastSeqLens[b] + seq + 1;
                small_softmax(S + seq * keyLen, scale, elements);
                if (keyLen > elements) { memset(S + seq * keyLen + elements, 0, (keyLen - elements) * sizeof(float)); }
            }

            // Softmax * V
            {
                auto result = output + qOffsets[b] * oStride + h * vHeadDim;
                auto V = value + kvOffsets[b] * kvStride + h * (nopeDim + vHeadDim);
                auto valueMat = std::make_tuple(V, kvStride, 1.0f);
                gemmSV(S, valueMat, result, queryLen, vHeadDim, keyLen, keyLen, oStride);
            }
        } // end for b
    } // end for outer head
}

// scaled dot-product attention: bmm1 + softmax + bmm2
// query key value are all in [*, seqLen, headnum, headsize] order
template <typename T, typename AttnT>
void selfScaledDpAttention(T *output, const T *query, const AttnT *key, const AttnT *value, int qHeadNum, int kvHeadNum,
        int headSize, int oStride, int qStride, int kvStride, int batchSize, const int *inputSeqLens,
        const int *pastSeqLens, bool causal, const float *alibiSlopes, const float *attnMask, const float scale,
        int threadNum, std::function<int(int)> headMap = nullptr) {
    // output = softmax(query * trans(key)) * value
    // causal = True: llama-family, chatglm2; extra alibiSlopes for baichuan
    // causal = False: just chatglm (prefixLLM, 0:startid) need attnMask for now

    // get the max seqLen
    int maxSrcLen = 0, maxTgtLen = 0;
    for (int i = 0; i < batchSize; ++i) {
        maxSrcLen = std::max(maxSrcLen, inputSeqLens[i]);
        maxTgtLen = std::max(maxTgtLen, inputSeqLens[i] + pastSeqLens[i]);
    }
    // compute the seqStartLoc
    int seqStartLoc[batchSize + 1];
    seqStartLoc[0] = 0;
    for (int i = 0; i < batchSize; ++i) {
        seqStartLoc[i + 1] = seqStartLoc[i] + inputSeqLens[i];
    }

    // closest value of power of 2
    int minBlk = (int)std::pow(2, int(std::log2((maxSrcLen + 1) / 2)));
    // Split sequence to make sure a moderate sync frequency and the intermediate
    // result [srcSeq * tgtSeq] in cache. The current block size is derived from practical experience.
    int srcBlk = std::min(256, minBlk);
    int tgtBlk = std::min(512, maxTgtLen);

    int groupNum = qHeadNum / kvHeadNum;

    int numArr = 7;
    int arrStride = (4 + tgtBlk + 2 * headSize) * srcBlk;
    float *thrBuf
            = (float *)SimpleMemPool::instance().getBuffer("threadBuffers", sizeof(float) * threadNum * arrStride);
    float **thrPtrBuf
            = (float **)SimpleMemPool::instance().getBuffer("threadPtrBuffers", sizeof(float *) * threadNum * numArr);

    float **preSum = thrPtrBuf;
    float **sum = thrPtrBuf + threadNum;
    float **preMax = thrPtrBuf + threadNum * 2;
    float **max = thrPtrBuf + threadNum * 3;
    float **qkArr = thrPtrBuf + threadNum * 4;
    float **expQkvArr = thrPtrBuf + threadNum * 5;
    float **qArr = thrPtrBuf + threadNum * 6;

    for (int i = 0; i < threadNum; ++i) {
        preSum[i] = thrBuf + srcBlk * i;
        sum[i] = thrBuf + srcBlk * threadNum + srcBlk * i;
        preMax[i] = thrBuf + srcBlk * threadNum * 2 + srcBlk * i;
        max[i] = thrBuf + srcBlk * threadNum * 3 + srcBlk * i;
        qkArr[i] = thrBuf + srcBlk * threadNum * 4 + srcBlk * tgtBlk * i;
        expQkvArr[i] = thrBuf + srcBlk * threadNum * (4 + tgtBlk) + srcBlk * headSize * i;
        qArr[i] = thrBuf + srcBlk * threadNum * (4 + tgtBlk + headSize) + srcBlk * headSize * i;
    }

#pragma omp parallel for collapse(3) schedule(dynamic)
    for (uint64_t b = 0; b < batchSize; ++b) {
        for (int h = 0; h < qHeadNum; ++h) {
            for (int m = 0; m < maxSrcLen; m += srcBlk) {
                int srcLen = inputSeqLens[b];
                int tgtLen = inputSeqLens[b] + pastSeqLens[b];
                if (m >= srcLen) { continue; }

                int tid = omp_get_thread_num();
                int qRealBlk = std::min(srcBlk, srcLen - m);
                uint64_t srcOff = seqStartLoc[b] * qStride + h * headSize;
                uint64_t outOff = seqStartLoc[b] * oStride + h * headSize;
                const T *qbuf = query + srcOff + m * qStride;
                AttnT *q = (AttnT *)qArr[tid];
                T *out = output + outOff + m * oStride;

                // reset out
                for (int ii = 0; ii < qRealBlk; ++ii) {
#pragma omp simd
                    for (int jj = 0; jj < headSize; ++jj) {
                        out[ii * oStride + jj] = 0; // reset output
                        q[ii * headSize + jj] = (AttnT)(qbuf[ii * qStride + jj]); // reset output
                    }
                }
                // reset sum
#pragma omp simd
                for (int ii = 0; ii < qRealBlk; ++ii) {
                    preSum[tid][ii] = 0;
                    sum[tid][ii] = 0;
                    preMax[tid][ii] = std::numeric_limits<float>::lowest();
                    max[tid][ii] = std::numeric_limits<float>::lowest();
                }

                int kvHeadIdx = (headMap == nullptr) ? h / groupNum : headMap(h);
                uint64_t tgtOff = seqStartLoc[b] * kvStride + kvHeadIdx * headSize;
                const AttnT *k = key + tgtOff;
                const AttnT *v = value + tgtOff;
                // split the target len dimension
                for (int n = 0; n < tgtLen; n += tgtBlk) {
                    int kvRealBlk = std::min(tgtBlk, tgtLen - n);
                    // mask out. TODO: for prefixLLM
                    if (causal && m + qRealBlk - 1 < n) {
                        //printf("Skip bs %d head %d src %d tgt %d\n", b, h, m, n);
                        break;
                    }

                    const AttnT *kBlk = k + n * kvStride;
                    const AttnT *vBlk = v + n * kvStride;

                    if (causal) {
                        // causal=True, build-in mask
                        float headSlope = alibiSlopes != nullptr ? alibiSlopes[h] : 0.0f;
                        DecoderUtil::incrementalTileAttentionCausal(q, kBlk, vBlk, headSlope, m, n, qRealBlk, headSize,
                                kvRealBlk, preSum[tid], sum[tid], preMax[tid], max[tid], scale, qkArr[tid],
                                expQkvArr[tid], out, headSize, kvStride, kvStride, oStride);
                    } else {
                        // causal=False, need mask matrix for now
                        const float *attnMsk = attnMask + seqStartLoc[b] * tgtLen + m * tgtLen + n;
                        DecoderUtil::incrementalTileAttention(q, kBlk, vBlk, attnMsk, qRealBlk, headSize, kvRealBlk,
                                tgtLen, preSum[tid], sum[tid], preMax[tid], max[tid], scale, qkArr[tid], expQkvArr[tid],
                                out, headSize, kvStride, kvStride, oStride);
                    }
                }
            }
        }
    }
    return;
}

} // namespace xft
