"""Regenerates small.safetensors, the reference fixture for the mlx package tests.

Run with the Python reference interpreter:

    /Users/b33j_02/sandbox/mlx/.venv/bin/python mlx/testdata/gen_small.py

Inputs are closed-form so the file is reproducible; every "*_expected" tensor
is what Python MLX computes for the inputs stored next to it.
"""

import math
import pathlib

import mlx.core as mx
import numpy as np


def wave(shape, step):
    """A deterministic float32 tensor with values in [-1, 1]."""
    n = math.prod(shape)
    return mx.array(np.sin(np.arange(n, dtype=np.float64) * step).astype(np.float32).reshape(shape))


tensors = {}

# Quantization: an [8, 64] weight matrix, 4 bits, one group per row.
w = wave((8, 64), 0.37)
w_q, scales, biases = mx.quantize(w, group_size=64, bits=4)
x = wave((2, 64), 0.11)
tensors["quant.w"] = w
tensors["quant.w_q"] = w_q
tensors["quant.scales"] = scales
tensors["quant.biases"] = biases
tensors["quant.x"] = x
tensors["quant.matmul_expected"] = mx.quantized_matmul(
    x, w_q, scales, biases, transpose=True, group_size=64, bits=4
)
# Without transpose the product is y @ w rather than x @ w.T.
y = wave((2, 8), 0.53)
tensors["quant.y"] = y
tensors["quant.matmul_untransposed_expected"] = mx.quantized_matmul(
    y, w_q, scales, biases, transpose=False, group_size=64, bits=4
)
tensors["quant.dequantize_expected"] = mx.dequantize(w_q, scales, biases, group_size=64, bits=4)

# The same matrix in mxfp4 mode, which has no biases.
fp4_w_q, fp4_scales = mx.quantize(w, group_size=32, bits=4, mode="mxfp4")
tensors["mxfp4.w_q"] = fp4_w_q
tensors["mxfp4.scales"] = fp4_scales
tensors["mxfp4.matmul_expected"] = mx.quantized_matmul(
    x, fp4_w_q, fp4_scales, None, transpose=True, group_size=32, bits=4, mode="mxfp4"
)
tensors["mxfp4.dequantize_expected"] = mx.dequantize(
    fp4_w_q, fp4_scales, None, group_size=32, bits=4, mode="mxfp4"
)

# RMS norm with eps=1e-5.
rms_x = wave((2, 4), 0.9)
rms_weight = mx.array([0.5, 1.0, 1.5, 2.0], dtype=mx.float32)
tensors["rms.x"] = rms_x
tensors["rms.weight"] = rms_weight
tensors["rms.expected"] = mx.fast.rms_norm(rms_x, rms_weight, 1e-5)

# RoPE with dims=8, base=10000, scale=1, non-traditional.
rope_x = wave((1, 2, 4, 8), 0.23)
tensors["rope.x"] = rope_x
for offset in (0, 3):
    tensors[f"rope.expected_offset{offset}"] = mx.fast.rope(
        rope_x, 8, traditional=False, base=10000.0, scale=1.0, offset=offset
    )
# The traditional variant, with every other argument off its default too.
tensors["rope.expected_traditional"] = mx.fast.rope(
    rope_x, 8, traditional=True, base=100.0, scale=0.5, offset=1
)

# Scaled dot-product attention with scale=1/sqrt(8).
q = wave((1, 2, 4, 8), 0.31)
k = wave((1, 2, 4, 8), 0.47)
v = wave((1, 2, 4, 8), 0.59)
additive_mask = mx.array(
    [[0, -1e9, 0, -1e9], [0, 0, -1e9, -1e9], [-1e9, 0, 0, 0], [0, 0, 0, 0]], dtype=mx.float32
)
tensors["sdpa.q"] = q
tensors["sdpa.k"] = k
tensors["sdpa.v"] = v
tensors["sdpa.mask"] = additive_mask
scale = 1.0 / math.sqrt(8)
tensors["sdpa.expected_unmasked"] = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)
tensors["sdpa.expected_causal"] = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
tensors["sdpa.expected_array"] = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=additive_mask)

# One tensor per dtype, to pin the DType constants to MLX's numbering. float64
# is missing because MLX cannot write it to safetensors.
for name in (
    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64",
    "float16", "float32", "bfloat16", "complex64",
):
    dtype = mx.bool_ if name == "bool" else getattr(mx, name)
    tensors[f"dtype.{name}"] = mx.zeros((2,), dtype=dtype)

mx.eval(tensors)
out = pathlib.Path(__file__).with_name("small.safetensors")
mx.save_safetensors(
    str(out), tensors, metadata={"generator": "gen_small.py", "mlx_version": mx.__version__}
)

for key in sorted(tensors):
    print(f"{key:28s} {str(tensors[key].dtype):20s} {tensors[key].shape}")
