"""
Fused rowwise FP8 quantization kernel
"""
import torch
import triton
import triton.language as tl

FP8_MAX = 448.0

@triton.jit
def fused_fp8_quant_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel: compute rowwise scale and quantize to FP8 in one pass.
    One program per row. BLOCK_K must be >= K.
    """
    row = tl.program_id(0)

    k_offs = tl.arange(0, BLOCK_K)
    mask = k_offs < K

    # Load row
    x = tl.load(x_ptr + row * stride_xm + k_offs * stride_xk, mask=mask, other=0.0)

    # Compute amax and scale
    x_abs = tl.abs(x)
    row_amax = tl.max(x_abs, axis=0)
    scale = row_amax / 448.0
    scale = tl.maximum(scale, 1e-12)

    # Store scale as bf16
    tl.store(scale_ptr + row, scale)

    # Quantize: divide then cast to fp8
    x_scaled = x / scale
    x_fp8 = x_scaled.to(tl.float8e4nv)

    tl.store(out_ptr + row * stride_om + k_offs * stride_ok, x_fp8, mask=mask)

@triton.jit
def fused_fp8_quant_large_k_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BLOCK_K: tl.constexpr,
):
    """
    Two-pass kernel for large K (> 8192).
    Pass 1: Compute amax across all K in chunks
    Pass 2: Quantize using the computed scale
    """
    row = tl.program_id(0)

    # Pass 1: Compute amax
    row_amax = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_ptr + row * stride_xm + k_offs * stride_xk, mask=mask, other=0.0)
        x_abs = tl.abs(x.to(tl.float32))
        block_max = tl.max(x_abs, axis=0)
        row_amax = tl.maximum(row_amax, block_max)

    # Compute scale
    row_amax_scalar = tl.sum(row_amax, axis=0)
    scale_bf16 = (row_amax_scalar / 448.0).to(tl.bfloat16)
    scale_bf16 = tl.maximum(scale_bf16, 1e-12)

    # Store scale
    tl.store(scale_ptr + row, scale_bf16)

    # Pass 2: Quantize
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_ptr + row * stride_xm + k_offs * stride_xk, mask=mask, other=0.0)
        x_scaled = x / scale_bf16
        x_fp8 = x_scaled.to(tl.float8e4nv)
        tl.store(out_ptr + row * stride_om + k_offs * stride_ok, x_fp8, mask=mask)


def fused_rowwise_fp8_quant(x: torch.Tensor):
    """
    Fully fused rowwise FP8 quantization.
    """
    M, K = x.shape

    out = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    scale_bf16 = torch.empty(M, device=x.device, dtype=torch.bfloat16)

    BLOCK_K = triton.next_power_of_2(K)

    if BLOCK_K <= 8192:
        grid = (M,)
        fused_fp8_quant_kernel[grid](
            x, out, scale_bf16,
            M, K,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=8,
        )
    else:
        grid = (M,)
        fused_fp8_quant_large_k_kernel[grid](
            x, out, scale_bf16,
            M, K,
            x.stride(0), x.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_K=4096,
            num_warps=8,
        )

    scale_f32 = scale_bf16.float().unsqueeze(1)
    return out, scale_f32