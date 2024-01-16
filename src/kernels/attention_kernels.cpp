#include <cstdio>
#include <omp.h>
#include "amx_sgemm_bf16bf16bf16.h"
#include "bfloat16.h"
#include "copy_util.h"
#include "decoder_util.h"
#include "gemm_kernel_ext.h"
#include "layers_attention.h"
#include "simple_mem_pool.h"
#include "softmax.h"

#ifndef unlikely
#define unlikely(x) __builtin_expect((x), 0)
#endif

namespace xft {

static int threadNum = 0;

// TODO: group attention
void selfAttention(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int qStride, int kvStride, int batchSize, const int *tokenSizes, const float scale,
        const void *kcache, const void *vcache, int *slots) {
#ifdef DEBUG
    printf("Q[0]=%f, K[0]=%f, V[0]=%f\n", (float)query[0], (float)key[0], (float)value[0]);
    printf("kvHeadNum=%d, headSize=%d, qStride=%d, kvStride=%d, batchSize=%d\n", kvHeadNum, headSize, qStride, kvStride,
            batchSize);
#endif

    constexpr int mBlockSize = 32;

    if (unlikely(threadNum == 0)) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) { threadNum = omp_get_num_threads(); }
        }
    }

    if (unlikely(kvHeadNum != qHeadNum)) {
        printf("Error: grouped attention is not implemented yet.\n");
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
    printf("maxTokenSize=%d, tokenSizes[0]=%d, offsets[0]=%d, kvStride=%d\n", maxTokenSize, tokenSizes[0], offsets[0],
            kvStride);

#pragma omp parallel for collapse(2) num_threads(1)
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < qHeadNum; ++i) {
            int tid = omp_get_thread_num();
            const int tokens = tokenSizes[b];
            const int mBlockNum = (tokens + mBlockSize - 1) / mBlockSize;

            bfloat16_t *packedB = packBuf + tid * (kPackSize + vPackSize);
            bfloat16_t *packedV = packedB + kPackSize;

            // Copy one head of current key to cached keys
            auto dst = (bfloat16_t *)kcache + slots[b] * kvHeadNum * headSize + i * headSize;
            xft::copy(dst, key + i * headSize, headSize);

            // Pack the key and value
            auto B = key + offsets[b] * kvStride + i * headSize;
            xdnn_small_amx_sgemm_bf16bf16bf16_packb(
                    true, tokens, headSize, (XDNN_BF16 *)B, kvStride, (XDNN_BF16 *)packedB, kPackSize);
            auto V = value + offsets[b] * kvStride + i * headSize;
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

                // Copy current value to cached values
                dst = (bfloat16_t *)vcache + slots[b] * kvHeadNum * headSize + i * headSize;
                xft::copy(dst, value + i * headSize, headSize);

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
                ldc = qHeadNum * headSize;
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

void crossAttention(bfloat16_t *output, bfloat16_t *query, bfloat16_t *key, bfloat16_t *value, int qHeadNum,
        int kvHeadNum, int headSize, int qStride, int kvStride, int batchSize, int cacheBlkStride, int cacheBlkSize,
        const int *contextSizes, const float scale, const void *kcache, const void *vcache, int *blockTables,
        int *blockNums, int *slots) {
    int maxCtxSize = 0;
    int blkOffsets[batchSize]; // offset in blockTables
    int curOff = 0;

    if (unlikely(threadNum == 0)) {
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid == 0) { threadNum = omp_get_num_threads(); }
        }
    }

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
        const int q_stride, const int kv_stride, const float scale, const int batch_size, const int *token_lens,
        const void *kcache, const void *vcache, int *kvcache_shape, int *block_tables, int *block_nums,
        int *context_lens, int layer_id, bool is_prefill, int *slot_mapping) {
    // query_shape is like [total_tokens, query_head_num, head_size]
    int qHeadNum = query_shape[1];
    int headSize = query_shape[2];
    int kvHeadNum = kv_shape[1];

    if (dt == DataType::bf16) {
        if (!is_prefill) { // generate phase
            int cacheBlkSize = kvcache_shape[1]; // typically 4 on GPU
            int cacheBlkStride = kvcache_shape[1] * kvcache_shape[2] * kvcache_shape[3];
            crossAttention((bfloat16_t *)output, (bfloat16_t *)query, (bfloat16_t *)key, (bfloat16_t *)value, qHeadNum,
                    kvHeadNum, headSize, q_stride, kv_stride, batch_size, cacheBlkStride, cacheBlkSize, context_lens,
                    scale, kcache, vcache, block_tables, block_nums, slot_mapping);
        } else { // prefill phase
            selfAttention((bfloat16_t *)output, (bfloat16_t *)query, (bfloat16_t *)key, (bfloat16_t *)value, qHeadNum,
                    kvHeadNum, headSize, q_stride, kv_stride, batch_size, token_lens, scale, kcache, vcache,
                    slot_mapping);
        }
    } else {
        printf("Error: data type (%d) is not supported yet!\n", dt);
        exit(-1);
    }
}

} // namespace xft