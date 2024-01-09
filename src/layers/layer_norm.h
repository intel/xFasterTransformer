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
#pragma once

namespace xft {

// Layer normalization: only support the norm along last dimension
class LayerNorm {
public:
    LayerNorm();
    ~LayerNorm();

    void setWeight(const float *gamma, const float *beta, int cols);

    // input and output are in shape of (rows, normSize)
    // TODO: column-wise parallel
    void forward(const float *input, float *output, int rows, int iStride = -1, int oStride = -1, float epsilon = 1e-5);

private:
    int normSize;

    // the weights contains gamma and beta concated together
    float *weights;
};

} // namespace xft