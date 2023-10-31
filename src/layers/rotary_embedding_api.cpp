#include "rotary_embedding_api.h"

void xft_rotary_embedding_kernel(
    const int64_t *__restrict__ position_ids, float *__restrict__ query,
    float *__restrict__ key, const float *__restrict__ emb_cos,
    const float *__restrict__ emb_sin, const int dim, const int qStride,
    const int kStride, const int num_tokens, const int head_num,
    const int head_size, const int num_kv_heads = 0) {
  REQUIRES(dim == head_size, "Incorrect shape, rot_dim is not the head size.");
  const int half = (dim + 1) / 2; // inv_freq_size

#pragma omp parallel for
  for (int head = 0; head < head_num; ++head) {
    int off = head * dim;

    for (int row = 0; row < num_tokens; ++row) {
      float *p1 = query + row * qStride + off;
      float *p2 = key + row * kStride + off;

      int pos = position_ids[row];
      const float *pcos = emb_cos + pos * dim;
      const float *psin = emb_sin + pos * dim;

#pragma omp simd
      for (int i = 0; i < half; ++i) {
        auto t1 = p1[i];
        auto t2 = p2[i];

        p1[i] = p1[i] * pcos[i] - p1[i + half] * psin[i];
        p2[i] = p2[i] * pcos[i] - p2[i + half] * psin[i];

        p1[i + half] = p1[i + half] * pcos[i + half] + t1 * psin[i + half];
        p2[i + half] = p2[i + half] * pcos[i + half] + t2 * psin[i + half];
      }
    }
  }
}
