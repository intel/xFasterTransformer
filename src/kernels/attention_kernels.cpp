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
#include "attention_kernels.h"
#include "decoder_util.h"
#include "gemm_kernel_ext.h"
#include "layers_attention.h"

namespace xft {

static int threadNum = 0;

void crossAttention(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int qStride, int kvStride, int batchSize, int cacheBlkStride, int cacheBlkSize,
        const int *contextSizes, const float scale, const float *alibiSlopes, const void *kcache, const void *vcache,
        int *blockTables, int *blockNums, int *slots) {
    int maxCtxSize = 0;
    int blkOffsets[batchSize]; // offset in blockTables
    int curOff = 0;

    // blocktables dim = 2
    for (int i = 0; i < batchSize; ++i) {
        if (contextSizes[i] > maxCtxSize) { maxCtxSize = contextSizes[i]; }
    }

    int max_block_num = (maxCtxSize + cacheBlkSize - 1) / cacheBlkSize;
    for (int i = 0; i < batchSize; ++i) {
        blkOffsets[i] = curOff;
        curOff += max_block_num;
    }

    int thrScoreSize = (maxCtxSize + 15) / 16 * 16;
    float *scores = (float *)SimpleMemPool::instance().getBuffer("qkscore", threadNum * thrScoreSize * sizeof(float));

#pragma omp parallel for collapse(2)
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < qHeadNum; ++i) {
            int *blkIndices = blockTables + blkOffsets[b];

            // Copy one head of current key to cached keys
            auto dst = (bfloat16_t *)kcache + slots[b] * kvHeadNum * headSize + i * headSize;
            xft::copy(dst, key + i * headSize, headSize);

            // Q * K
            int m = 1;
            int k = headSize;
            int n = contextSizes[b] + 1;
            int lda = qStride;
            int ldb = kvHeadNum * headSize;
            int ldc = n;
            auto A = query + i * headSize;
            auto baseB = (bfloat16_t *)kcache + i * headSize;
            auto C = scores + omp_get_thread_num() * thrScoreSize;

            small_sgemm_bf16bf16f32_b(true, m, n, k, (XDNN_BF16 *)A, lda, (XDNN_BF16 *)baseB, ldb, C, ldc, blkIndices,
                    cacheBlkStride, cacheBlkSize);

#ifdef DEBUG
            if (b == 0 && i == 0) {
                printf("Q * K, first head:\n");
                auto p = C;
                printf("%f, %f, %f ... %f %f %f\n", p[0] * scale, p[1] * scale, p[2] * scale, p[n - 3] * scale,
                        p[n - 2] * scale, p[n - 1] * scale);
            }
#endif

            // Softmax(Q * K)
            small_softmax_f32(C, scale, n);

#ifdef DEBUG
            if (b == 0 && i == 0) {
                printf("Softmax(Q * K), first head:\n");
                auto p = C;
                printf("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[n - 3], p[n - 2], p[n - 1]);
            }
#endif

            // Copy current value to cached values
            dst = (bfloat16_t *)vcache + slots[b] * kvHeadNum * headSize + i * headSize;
            xft::copy(dst, value + i * headSize, headSize);

            // Softmax * V
            std::swap(k, n);
            lda = ldc;
            ldb = kvHeadNum * headSize;
            ldc = qHeadNum * headSize;
            baseB = (bfloat16_t *)vcache + i * headSize;
            auto baseC = output + b * ldc + i * headSize;
            small_sgemm_f32bf16bf16_b(false, m, n, k, C, lda, (XDNN_BF16 *)baseB, ldb, (XDNN_BF16 *)baseC, ldc,
                    blkIndices, cacheBlkStride, cacheBlkSize);

#ifdef DEBUG
            if (b == 0 && i == 0) {
                printf("Softmax(Q * K) * V, first head:\n");
                auto p = C;
                printf("%f, %f, %f ... %f %f %f\n", p[0], p[1], p[2], p[headSize - 3], p[headSize - 2],
                        p[headSize - 1]);
            }
#endif
        } // end for i
    } // end for b
}

void invokeAttention(DataType dt, void *__restrict__ output, const void *__restrict__ query,
        const void *__restrict__ key, const void *__restrict__ value, int *query_shape, int *kv_shape,
        const int q_stride, const int kv_stride, const float scale, const float *alibiSlopes, const int batch_size,
        const int *token_lens, const void *kcache, const void *vcache, int *kvcache_shape, int *block_tables,
        int *block_nums, int *context_lens, int layer_id, bool is_prefill, int *slot_mapping) {
    // query_shape is like [total_tokens, query_head_num, head_size]
    int qHeadNum = query_shape[1];
    int headSize = query_shape[2];
    int kvHeadNum = kv_shape[1];

    if (unlikely(threadNum == 0)) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) { threadNum = omp_get_num_threads(); }
        }
    }

    if (dt == DataType::bf16) {
        if (!is_prefill) { // generate phase
            int cacheBlkSize = kvcache_shape[1]; // typically 4 on GPU
            int cacheBlkStride = kvcache_shape[1] * kvcache_shape[2] * kvcache_shape[3];
            crossAttention((bfloat16_t *)output, (bfloat16_t *)query, (bfloat16_t *)key, (bfloat16_t *)value, qHeadNum,
                    kvHeadNum, headSize, q_stride, kv_stride, batch_size, cacheBlkStride, cacheBlkSize, context_lens,
                    scale, alibiSlopes, kcache, vcache, block_tables, block_nums, slot_mapping);
        } else { // prefill phase
            selfAttention((bfloat16_t *)output, (bfloat16_t *)query, (bfloat16_t *)key, (bfloat16_t *)value, qHeadNum,
                    kvHeadNum, headSize, qHeadNum * headSize, q_stride, kv_stride, batch_size, token_lens, scale,
                    alibiSlopes, threadNum,
                    [&](int b, int headIdx, int seqIdex) {
                        // TODO: debug and fix
                        return (bfloat16_t *)kcache + slot_mapping[b] * kvHeadNum * headSize + headIdx * headSize;
                    },
                    [&](int b, int headIdx, int seqIdex) {
                        return (bfloat16_t *)vcache + slot_mapping[b] * kvHeadNum * headSize + headIdx * headSize;
                    });
        }
    } else {
        printf("Error: data type (%d) is not supported yet!\n", dt);
        exit(-1);
    }
}

} // namespace xft
