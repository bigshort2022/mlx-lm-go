/*
 * Unmasked, scalar-gate kernel adapted from mlx-lm 0.32.2,
 * mlx_lm/models/gated_delta.py (Copyright © 2025 Apple Inc.).
 *
 * MIT License
 *
 * Copyright © 2023 Apple Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef MLX_GO_GATED_DELTA_METAL_H
#define MLX_GO_GATED_DELTA_METAL_H

// One SIMD group per value row. Keep the reference reduction and update
// order: the core MLX chunked algorithm changes BF16 rounding on prefill.
static const char mlx_go_gated_delta_source[] =
    "auto n = thread_position_in_grid.z;\n"
    "auto b_idx = n / Hv;\n"
    "auto hv_idx = n % Hv;\n"
    "auto hk_idx = hv_idx / (Hv / Hk);\n"
    "constexpr int n_per_t = Dk / 32;\n"
    "auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;\n"
    "auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;\n"
    "auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;\n"
    "y += b_idx * T * Hv * Dv + hv_idx * Dv;\n"
    "auto dk_idx = thread_position_in_threadgroup.x;\n"
    "auto dv_idx = thread_position_in_grid.y;\n"
    "auto i_state = state_in + (n * Dv + dv_idx) * Dk;\n"
    "auto o_state = state_out + (n * Dv + dv_idx) * Dk;\n"
    "float state[n_per_t];\n"
    "for (int i = 0; i < n_per_t; ++i) {\n"
    "  auto s_idx = n_per_t * dk_idx + i;\n"
    "  state[i] = static_cast<float>(i_state[s_idx]);\n"
    "}\n"
    "auto g_ = g + b_idx * T * Hv;\n"
    "auto beta_ = beta + b_idx * T * Hv;\n"
    "for (int t = 0; t < T; ++t) {\n"
    "  float kv_mem = 0.0f;\n"
    "  for (int i = 0; i < n_per_t; ++i) {\n"
    "    auto s_idx = n_per_t * dk_idx + i;\n"
    "    state[i] = state[i] * g_[hv_idx];\n"
    "    kv_mem += state[i] * k_[s_idx];\n"
    "  }\n"
    "  kv_mem = simd_sum(kv_mem);\n"
    "  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];\n"
    "  float out = 0.0f;\n"
    "  for (int i = 0; i < n_per_t; ++i) {\n"
    "    auto s_idx = n_per_t * dk_idx + i;\n"
    "    state[i] = state[i] + k_[s_idx] * delta;\n"
    "    out += state[i] * q_[s_idx];\n"
    "  }\n"
    "  out = simd_sum(out);\n"
    "  if (thread_index_in_simdgroup == 0) {\n"
    "    y[dv_idx] = static_cast<InT>(out);\n"
    "  }\n"
    "  q_ += Hk * Dk;\n"
    "  k_ += Hk * Dk;\n"
    "  v_ += Hv * Dv;\n"
    "  y += Hv * Dv;\n"
    "  g_ += Hv;\n"
    "  beta_ += Hv;\n"
    "}\n"
    "for (int i = 0; i < n_per_t; ++i) {\n"
    "  auto s_idx = n_per_t * dk_idx + i;\n"
    "  o_state[s_idx] = state[i];\n"
    "}\n";

#endif
