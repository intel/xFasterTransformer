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
#include "amx_sgemm_bf16bf16bf16.h"
#include "bfloat16.h"
#include "copy_util.h"
#include "simple_mem_pool.h"
#include "softmax.h"
#include "thread_util.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

namespace xft {

// Self attention while KV cache copy is separated
template <bool fusedPack, typename Lambda1, typename Lambda2>
void selfAttention_SeparateCopy(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int oStride, int qStride, int kvStride, int batchSize, const int *tokenSizes,
        const float scale, int threadNum, const Lambda1 &getKCache, const Lambda2 &getVCache) {
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
            = fusedPack ? threadNum * (kPackSize + vPackSize) : (batchSize * kvHeadNum) * (kPackSize + vPackSize);

    bfloat16_t *packBuf
            = (bfloat16_t *)SimpleMemPool::instance().getBuffer("kv_packing", totalPackSize * sizeof(bfloat16_t));

    // Copy key/value to cache and pack them
    // If packing is not fused into computing, then pack it here
    if constexpr (!fusedPack) { 
#pragma omp parallel for collapse(2)
        for (int b = 0; b < batchSize; ++b) {
            for (int i = 0; i < kvHeadNum; ++i) {
                const int tokens = tokenSizes[b];

                bfloat16_t *packedB = packBuf + (b * kvHeadNum + i) * (kPackSize + vPackSize);
                bfloat16_t *packedV = packedB + kPackSize;

                auto B = key + offsets[b] * kvStride + i * headSize;
                for (int s = 0; s < tokens; ++s) {
                    auto dst = getKCache(b, i, s);
                    xft::copy(dst, B + s * kvStride, headSize);
                }

                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);

                auto V = value + offsets[b] * kvStride + i * headSize;
                for (int s = 0; s < tokens; ++s) {
                    auto dst = getVCache(b, i, s);
                    xft::copy(dst, V + s * kvStride, headSize);
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
            xft::copy(dst, B + s * kvStride, headSize);

            auto V = value + offsets[b] * kvStride + i * headSize;
            dst = getVCache(b, i, s);
            xft::copy(dst, V + s * kvStride, headSize);
        });
    }

    // Prepare score buffer
    auto maxScoreStride = (maxTokenSize + 31) / 32 * 32;
    bfloat16_t *scores = (bfloat16_t *)SimpleMemPool::instance().getBuffer(
            "qkscore", threadNum * mBlockSize * maxScoreStride * sizeof(bfloat16_t));

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
        int kvHeadIdx = i / groupNum;
        int locationIdx = (fusedPack ? tid : b * kvHeadNum + kvHeadIdx);
        bfloat16_t *packedB = packBuf + locationIdx * (kPackSize + vPackSize);
        bfloat16_t *packedV = packedB + kPackSize;

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

        if constexpr (fusedPack) {
            if (packInfo[tid].first != b || packInfo[tid].second != kvHeadIdx) {
                auto B = key + offsets[b] * kvStride + kvHeadIdx * headSize;
                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);
            }
        }

        xdnn_small_amx_sgemm_bf16bf16bf16_compute(
                m, n, k, (XDNN_BF16 *)A, lda, (XDNN_BF16 *)packedB, (XDNN_BF16 *)C, ldc);

#ifdef DEBUG
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
            small_softmax_bf16((XDNN_BF16 *)(C + seq * ldc), scale, elements);
            memset(C + seq * ldc + elements, 0, (tokens - elements) * sizeof(bfloat16_t));
        }

#ifdef DEBUG
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
        C = (bfloat16_t *)output + (offsets[b] + startSeq) * ldc + i * headSize;

        if constexpr (fusedPack) {
            if (packInfo[tid].first != b || packInfo[tid].second != kvHeadIdx) {
                auto V = value + offsets[b] * kvStride + kvHeadIdx * headSize;
                xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                        false, headSize, tokens, (XDNN_BF16 *)V, kvStride, (XDNN_BF16 *)packedV, vPackSize);
                // Update pack info
                packInfo[tid].first = b;
                packInfo[tid].second = kvHeadIdx;
            }
        }

        xdnn_small_amx_sgemm_bf16bf16bf16_compute(
                m, n, k, (XDNN_BF16 *)A, lda, (XDNN_BF16 *)packedV, (XDNN_BF16 *)C, ldc);

#ifdef DEBUG
        if (b == 0 && i == 0) {
            printf("Softmax(Q * Kᵀ) * V, first head:\n");
            auto p = C;
            printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[headSize - 3],
                    (float)p[headSize - 2], (float)p[headSize - 1]);
        }
#endif
    });
}

template <typename Lambda1, typename Lambda2>
void selfAttention_FusedCopy(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int oStride, int qStride, int kvStride, int batchSize, const int *tokenSizes,
        const float scale, int threadNum, const Lambda1 &getKCache, const Lambda2 &getVCache) {
#ifdef DEBUG
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
    for (int i = 0; i < batchSize; ++i) {
        offsets[i] = curOff;
        curOff += tokenSizes[i];
        if (tokenSizes[i] > maxTokenSize) { maxTokenSize = tokenSizes[i]; }
    }

    // Prepare buffers (packing buffer and score buffer)
    const int kPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(maxTokenSize, headSize, 32, 32);
    const int vPackSize = xdnn_small_amx_sgemm_bf16bf16bf16_packb_size(headSize, maxTokenSize, 32, 32);
    bfloat16_t *packBuf = (bfloat16_t *)SimpleMemPool::instance().getBuffer(
            "kv_packing", threadNum * (kPackSize + vPackSize) * sizeof(bfloat16_t));
    int maxScoreStride = (maxTokenSize + 31) / 32 * 32;
    bfloat16_t *scores = (bfloat16_t *)SimpleMemPool::instance().getBuffer(
            "qkscore", threadNum * mBlockSize * maxScoreStride * sizeof(bfloat16_t));

#ifdef DEBUG
    printf("maxTokenSize=%d, tokenSizes[0]=%d, offsets[0]=%d, kvStride=%d\n", maxTokenSize, tokenSizes[0], offsets[0],
            kvStride);
#endif

#pragma omp parallel for collapse(2)
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < qHeadNum; ++i) {
            int tid = omp_get_thread_num();
            const int tokens = tokenSizes[b];
            const int mBlockNum = (tokens + mBlockSize - 1) / mBlockSize;

            bfloat16_t *packedB = packBuf + tid * (kPackSize + vPackSize);
            bfloat16_t *packedV = packedB + kPackSize;

            // Copy key/value to cache and pack them
            auto B = key + offsets[b] * kvStride + i * headSize;
            for (int s = 0; s < tokens; ++s) {
                auto dst = getKCache(b, i, s);
                xft::copy(dst, B + s * kvStride, headSize);
            }

            xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                    true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);

            auto V = value + offsets[b] * kvStride + i * headSize;
            for (int s = 0; s < tokens; ++s) {
                auto dst = getVCache(b, i, s);
                xft::copy(dst, V + s * kvStride, headSize);
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

                xdnn_small_amx_sgemm_bf16bf16bf16_compute(
                        m, n, k, (XDNN_BF16 *)A, lda, (XDNN_BF16 *)packedB, (XDNN_BF16 *)C, ldc);

#ifdef DEBUG
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
                    small_softmax_bf16((XDNN_BF16 *)(C + seq * ldc), scale, elements);
                    memset(C + seq * ldc + elements, 0, (tokens - elements) * sizeof(bfloat16_t));
                }

#ifdef DEBUG
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
                C = (bfloat16_t *)output + (offsets[b] + startSeq) * ldc + i * headSize;

                xdnn_small_amx_sgemm_bf16bf16bf16_compute(
                        m, n, k, (XDNN_BF16 *)A, lda, (XDNN_BF16 *)packedV, (XDNN_BF16 *)C, ldc);

#ifdef DEBUG
                if (b == 0 && i == 0) {
                    printf("Softmax(Q * Kᵀ) * V, first head:\n");
                    auto p = C;
                    printf("%f, %f, %f ... %f %f %f\n", (float)p[0], (float)p[1], (float)p[2], (float)p[headSize - 3],
                            (float)p[headSize - 2], (float)p[headSize - 1]);
                }
#endif
            } // end for mb
        } // end for i
    } // end for b
}

template <typename Lambda1, typename Lambda2>
void selfAttention(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int oStride, int qStride, int kvStride, int batchSize, const int *tokenSizes,
        const float scale, int threadNum, const Lambda1 &getKCache, const Lambda2 &getVCache) {
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
                batchSize, tokenSizes, scale, threadNum, getKCache, getVCache);
    } else {
        selfAttention_SeparateCopy<true>(output, query, key, value, qHeadNum, kvHeadNum, headSize, oStride, qStride,
                kvStride, batchSize, tokenSizes, scale, threadNum, getKCache, getVCache);
    }
}

} // namespace xft