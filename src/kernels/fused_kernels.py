import torch
import triton
import triton.language as tl

FP8_MAX = 448.0

# Kernel 1: Fused AdaLN + FP8 Quant
# LayerNorm(no_affine) + scale/shift modulation + FP8 quantization
# Input: x [M, D] bf16, shift [B, D] bf16, scale [B, D] bf16
# Output: out_fp8 [M, D] fp8, fp8_scale [M] f32

@triton.jit
def fused_adaln_fp8_kernel(
    x_ptr, shift_ptr, scale_ptr,
    out_fp8_ptr, fp8_scale_ptr,
    M, D, L,  # M = B*L total rows, D = dim, L = seq_len
    stride_xm, stride_xd,
    stride_sm, stride_sd,  # shift/scale strides (B, D)
    stride_om, stride_od,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // L

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    # Load input row
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (no affine): mean, var, normalize
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # Load modulation params for this batch element
    shift = tl.load(shift_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(scale_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)

    # Apply AdaLN modulation: x_mod = x_norm * (1 + scale) + shift
    x_mod = x_norm * (1.0 + sc) + shift

    # FP8 quantization: compute row amax, scale, quantize
    x_abs = tl.abs(x_mod)
    row_amax = tl.max(x_abs, axis=0)
    fp8_sc = row_amax / 448.0
    fp8_sc = tl.maximum(fp8_sc, 1e-12)

    tl.store(fp8_scale_ptr + row, fp8_sc)

    x_fp8 = (x_mod / fp8_sc).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + row * stride_om + d_offs * stride_od, x_fp8, mask=mask)


@triton.jit
def fused_adaln_fp8_large_d_kernel(
    x_ptr, shift_ptr, scale_ptr,
    out_fp8_ptr, fp8_scale_ptr,
    M, D, L,
    stride_xm, stride_xd,
    stride_sm, stride_sd,
    stride_om, stride_od,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Two-pass variant for D > 8192."""
    row = tl.program_id(0)
    batch_idx = row // L

    # Pass 1: Compute mean
    acc_sum = tl.zeros([1], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        acc_sum += tl.sum(x, axis=0)
    mean = tl.sum(acc_sum, axis=0) / D

    # Pass 2: Compute variance
    acc_var = tl.zeros([1], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        x_centered = x - mean
        acc_var += tl.sum(x_centered * x_centered, axis=0)
    var = tl.sum(acc_var, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 3: Normalize + modulate + compute amax
    row_amax = tl.zeros([1], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)

        x_norm = (x - mean) * rstd
        x_mod = x_norm * (1.0 + sc) + shift
        block_max = tl.max(tl.abs(x_mod), axis=0)
        row_amax = tl.maximum(row_amax, block_max)

    fp8_sc = tl.sum(row_amax, axis=0) / 448.0
    fp8_sc = tl.maximum(fp8_sc, 1e-12)
    tl.store(fp8_scale_ptr + row, fp8_sc)

    # Pass 4: Normalize + modulate + quantize
    for d_start in range(0, D, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)
        mask = d_offs < D
        x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)

        x_norm = (x - mean) * rstd
        x_mod = x_norm * (1.0 + sc) + shift
        x_fp8 = (x_mod / fp8_sc).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + row * stride_om + d_offs * stride_od, x_fp8, mask=mask)


def fused_adaln_fp8(x, shift, scale, L, eps=1e-6):
    """
    Fused LayerNorm(no affine) + AdaLN modulate + FP8 rowwise quant.

    Args:
        x: [B*L, D] bf16 input
        shift: [B, D] bf16 shift params
        scale: [B, D] bf16 scale params
        L: sequence length (to compute batch index from row)
        eps: LayerNorm epsilon
    Returns:
        out_fp8: [B*L, D] float8_e4m3fn
        fp8_scale: [B*L, 1] float32
    """
    M, D = x.shape
    out_fp8 = torch.empty(M, D, device=x.device, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)

    if BLOCK_D <= 8192:
        fused_adaln_fp8_kernel[(M,)](
            x, shift, scale, out_fp8, fp8_scale,
            M, D, L,
            x.stride(0), x.stride(1),
            shift.stride(0), shift.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            eps=eps, BLOCK_D=BLOCK_D,
            num_warps=4,
        )
    else:
        fused_adaln_fp8_large_d_kernel[(M,)](
            x, shift, scale, out_fp8, fp8_scale,
            M, D, L,
            x.stride(0), x.stride(1),
            shift.stride(0), shift.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            eps=eps, BLOCK_D=4096,
            num_warps=4,
        )

    return out_fp8, fp8_scale.unsqueeze(1)

# Kernel 1b: Fused AdaLN (bf16 output, no FP8) - for when we need bf16 output

@triton.jit
def fused_adaln_bf16_kernel(
    x_ptr, shift_ptr, scale_ptr,
    out_ptr,
    M, D, L,
    stride_xm, stride_xd,
    stride_sm, stride_sd,
    stride_om, stride_od,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // L

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    shift = tl.load(shift_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(scale_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)

    x_mod = x_norm * (1.0 + sc) + shift
    tl.store(out_ptr + row * stride_om + d_offs * stride_od, x_mod.to(tl.bfloat16), mask=mask)


def fused_adaln_bf16(x, shift, scale, L, eps=1e-6):
    """Fused LayerNorm + modulate, bf16 output."""
    M, D = x.shape
    out = torch.empty(M, D, device=x.device, dtype=torch.bfloat16)
    BLOCK_D = triton.next_power_of_2(D)
    fused_adaln_bf16_kernel[(M,)](
        x, shift, scale, out,
        M, D, L,
        x.stride(0), x.stride(1),
        shift.stride(0), shift.stride(1),
        out.stride(0), out.stride(1),
        eps=eps, BLOCK_D=BLOCK_D,
        num_warps=8,
    )
    return out


# Kernel 2: Fused Residual + Gate + AdaLN + FP8 Quant
# residual_new = residual + gate * x
# out_fp8 = FP8(LayerNorm(residual_new) * (1+scale) + shift)

@triton.jit
def fused_residual_gate_adaln_fp8_kernel(
    residual_ptr, gate_ptr, x_ptr,
    shift_ptr, scale_ptr,
    residual_out_ptr, out_fp8_ptr, fp8_scale_ptr,
    M, D, L,
    stride_rm, stride_rd,
    stride_gm, stride_gd,  # gate is [B, D], stride_gm = stride for batch dim
    stride_xm, stride_xd,
    stride_sm, stride_sd,
    stride_rom, stride_rod,
    stride_om, stride_od,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // L

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    # Load residual, gate, x
    res = tl.load(residual_ptr + row * stride_rm + d_offs * stride_rd, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + batch_idx * stride_gm + d_offs * stride_gd, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)

    # Residual + gate
    new_res = res + gate * x

    # Store new residual (bf16)
    tl.store(residual_out_ptr + row * stride_rom + d_offs * stride_rod, new_res.to(tl.bfloat16), mask=mask)

    # LayerNorm (no affine)
    mean = tl.sum(new_res, axis=0) / D
    x_centered = new_res - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = x_centered * rstd

    # Modulate
    shift = tl.load(shift_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(scale_ptr + batch_idx * stride_sm + d_offs * stride_sd, mask=mask, other=0.0).to(tl.float32)
    x_mod = x_norm * (1.0 + sc) + shift

    # FP8 quant
    x_abs = tl.abs(x_mod)
    row_amax = tl.max(x_abs, axis=0)
    fp8_sc = row_amax / 448.0
    fp8_sc = tl.maximum(fp8_sc, 1e-12)
    tl.store(fp8_scale_ptr + row, fp8_sc)

    x_fp8 = (x_mod / fp8_sc).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + row * stride_om + d_offs * stride_od, x_fp8, mask=mask)


def fused_residual_gate_adaln_fp8(residual, gate, x, shift, scale, L, eps=1e-6):
    """
    Fused: new_res = residual + gate*x, then LayerNorm + modulate + FP8 quant.

    Args:
        residual: [B*L, D] bf16
        gate: [B, D] bf16 (broadcast over L)
        x: [B*L, D] bf16
        shift: [B, D] bf16
        scale: [B, D] bf16
        L: seq length
    Returns:
        new_residual: [B*L, D] bf16
        out_fp8: [B*L, D] float8_e4m3fn
        fp8_scale: [B*L, 1] float32
    """
    M, D = residual.shape
    new_residual = torch.empty_like(residual)
    out_fp8 = torch.empty(M, D, device=residual.device, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty(M, device=residual.device, dtype=torch.float32)

    BLOCK_D = triton.next_power_of_2(D)
    assert BLOCK_D <= 8192, f"D={D} too large for single-pass kernel"

    fused_residual_gate_adaln_fp8_kernel[(M,)](
        residual, gate, x, shift, scale,
        new_residual, out_fp8, fp8_scale,
        M, D, L,
        residual.stride(0), residual.stride(1),
        gate.stride(0), gate.stride(1),
        x.stride(0), x.stride(1),
        shift.stride(0), shift.stride(1),
        new_residual.stride(0), new_residual.stride(1),
        out_fp8.stride(0), out_fp8.stride(1),
        eps=eps, BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return new_residual, out_fp8, fp8_scale.unsqueeze(1)

# Kernel 2b: Fused Residual + Gate (simple, for end-of-block where no next norm)

@triton.jit
def fused_residual_gate_kernel(
    residual_ptr, gate_ptr, x_ptr, out_ptr,
    M, D, L,
    stride_rm, stride_rd,
    stride_gm, stride_gd,
    stride_xm, stride_xd,
    stride_om, stride_od,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    batch_idx = row // L

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    res = tl.load(residual_ptr + row * stride_rm + d_offs * stride_rd, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + batch_idx * stride_gm + d_offs * stride_gd, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + row * stride_xm + d_offs * stride_xd, mask=mask, other=0.0).to(tl.float32)

    out = res + gate * x
    tl.store(out_ptr + row * stride_om + d_offs * stride_od, out.to(tl.bfloat16), mask=mask)


def fused_residual_gate(residual, gate, x, L):
    """Fused residual + gate * x."""
    M, D = residual.shape
    out = torch.empty_like(residual)
    BLOCK_D = triton.next_power_of_2(D)
    fused_residual_gate_kernel[(M,)](
        residual, gate, x, out,
        M, D, L,
        residual.stride(0), residual.stride(1),
        gate.stride(0), gate.stride(1),
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return out

# Kernel 3: Fused RoPE (real-number implementation, no complex overhead)
# Applies rotary embeddings: out = x * cos + rotate(x) * sin
# where rotate interleaves pairs: [-x1, x0, -x3, x2, ...]

@triton.jit
def fused_rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    M, H, D,  # M = B*S, H = num_heads, D = head_dim
    stride_xm, stride_xh, stride_xd,
    stride_cm, stride_cd,  # cos/sin: [S, D//2], need to index by seq position
    stride_om, stride_oh, stride_od,
    S,  # sequence length for computing seq index
    BLOCK_D: tl.constexpr,
):
    """RoPE for [B, S, H, D] tensor with cos/sin of shape [S, D//2]."""
    # Program ID maps to (batch*seq, head)
    row = tl.program_id(0)  # B*S
    head = tl.program_id(1)  # H

    seq_idx = row % S  # position within sequence

    half_d = D // 2
    d_offs = tl.arange(0, BLOCK_D)
    mask_full = d_offs < D
    mask_half = d_offs < half_d

    # Load full x row [D]
    x = tl.load(x_ptr + row * stride_xm + head * stride_xh + d_offs * stride_xd,
                mask=mask_full, other=0.0).to(tl.float32)

    # Load cos, sin [D//2] - same for all heads at this position
    cos = tl.load(cos_ptr + seq_idx * stride_cm + d_offs * stride_cd,
                  mask=mask_half, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + seq_idx * stride_cm + d_offs * stride_cd,
                  mask=mask_half, other=0.0).to(tl.float32)

    # Split x into pairs: x0=x[0::2], x1=x[1::2]
    # x is laid out as [x0, x1, x2, x3, ...] where pairs are (x0,x1), (x2,x3)
    x_even = tl.load(x_ptr + row * stride_xm + head * stride_xh + (d_offs * 2) * stride_xd,
                     mask=mask_half, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + row * stride_xm + head * stride_xh + (d_offs * 2 + 1) * stride_xd,
                    mask=mask_half, other=0.0).to(tl.float32)

    # Rotate: [-x_odd, x_even]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin

    # Interleave back
    tl.store(out_ptr + row * stride_om + head * stride_oh + (d_offs * 2) * stride_od,
             out_even.to(tl.bfloat16), mask=mask_half)
    tl.store(out_ptr + row * stride_om + head * stride_oh + (d_offs * 2 + 1) * stride_od,
             out_odd.to(tl.bfloat16), mask=mask_half)


def fused_rope(x, freqs_cis):
    """
    Fused RoPE application.

    Args:
        x: [B, S, H, D] bf16
        freqs_cis: complex [S, D//2] - will be decomposed to cos/sin
    Returns:
        out: [B, S, H, D] bf16
    """
    B, S, H, D = x.shape
    out = torch.empty_like(x)

    # Decompose complex freqs to cos/sin
    cos = freqs_cis.real.contiguous()  # [S, D//2]
    sin = freqs_cis.imag.contiguous()  # [S, D//2]

    BLOCK_D = triton.next_power_of_2(D // 2)

    grid = (B * S, H)
    fused_rope_kernel[grid](
        x, cos, sin, out,
        B * S, H, D,
        x.stride(0) * x.stride(1) if x.stride(0) == S * H * D else x.stride(1),  # stride for B*S dim
        x.stride(2), x.stride(3),
        cos.stride(0), cos.stride(1),
        out.stride(0) * out.stride(1) if out.stride(0) == S * H * D else out.stride(1),
        out.stride(2), out.stride(3),
        S,
        BLOCK_D=BLOCK_D,
    )
    return out


def fused_rope_simple(x, freqs_cis):
    """
    Fused RoPE - simpler version that handles contiguous [B, S, H, D] tensors.

    Args:
        x: [B, S, H, D] bf16, contiguous
        freqs_cis: complex [S, D//2]
    Returns:
        out: [B, S, H, D] bf16
    """
    B, S, H, D = x.shape

    # Decompose complex freqs
    cos = freqs_cis.real.contiguous()  # [S, D//2]
    sin = freqs_cis.imag.contiguous()  # [S, D//2]

    # Reshape x to work with pairs: [B, S, H, D//2, 2]
    x_pairs = x.view(B, S, H, D // 2, 2).float()
    x_even = x_pairs[..., 0]  # [B, S, H, D//2]
    x_odd = x_pairs[..., 1]   # [B, S, H, D//2]

    # cos/sin: [S, D//2] -> [1, S, 1, D//2]
    cos = cos[None, :, None, :]  # [1, S, 1, D//2]
    sin = sin[None, :, None, :]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin

    # Stack back: [B, S, H, D//2, 2] -> [B, S, H, D]
    out = torch.stack([out_even, out_odd], dim=-1).view(B, S, H, D).to(x.dtype)
    return out

# Kernel 3b: Fast RoPE - processes all heads per (batch, seq) position
# Input: x [B*S, H*D] contiguous (reshaped from [B, S, H, D])
# cos/sin: [S, D//2] float32
# Output: out [B*S, H*D] bf16

@triton.jit
def fused_rope_fast_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    S, H, D, HALF_D: tl.constexpr,
    stride_x_row,    # stride for B*S dim (= H*D)
    stride_cos_row,  # stride for S dim in cos/sin (= D//2)
    BLOCK_HD: tl.constexpr,  # H * HALF_D, process all heads' half-dims at once
):
    """Fast RoPE: one block per (batch*seq) position, processes all heads."""
    row = tl.program_id(0)  # index into B*S
    seq_idx = row % S

    # For each pair (h, d) where h in [0,H) and d in [0, D//2):
    #   x_even = x[row, h, 2*d], x_odd = x[row, h, 2*d+1]
    #   out_even = x_even * cos[seq_idx, d] - x_odd * sin[seq_idx, d]
    #   out_odd  = x_odd * cos[seq_idx, d] + x_even * sin[seq_idx, d]

    hd_offs = tl.arange(0, BLOCK_HD)
    mask = hd_offs < H * HALF_D

    # Decompose hd_offs into head and half_d indices
    h_idx = hd_offs // HALF_D   # which head
    d_idx = hd_offs % HALF_D    # which half-dim

    # Compute offsets into x: x[row, h, 2*d] and x[row, h, 2*d+1]
    # x is contiguous [B*S, H, D], so x[row, h, d] = x_ptr + row*H*D + h*D + d
    base = row * stride_x_row
    even_offset = base + h_idx * D + d_idx * 2
    odd_offset = even_offset + 1

    # Load x pairs
    x_even = tl.load(x_ptr + even_offset, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(x_ptr + odd_offset, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin (broadcast across heads, only depends on seq position and d)
    cs_offset = seq_idx * stride_cos_row + d_idx
    cos = tl.load(cos_ptr + cs_offset, mask=mask, other=1.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_offset, mask=mask, other=0.0).to(tl.float32)

    # Apply rotation
    out_even = x_even * cos - x_odd * sin
    out_odd = x_odd * cos + x_even * sin

    # Store
    tl.store(out_ptr + even_offset, out_even.to(tl.bfloat16), mask=mask)
    tl.store(out_ptr + odd_offset, out_odd.to(tl.bfloat16), mask=mask)


# Kernel 4: Fused GELU (tanh approx) + FP8 Rowwise Quant
# Eliminates memory roundtrip between GELU activation and FP8 quantization.
# Input: x [M, K] bf16 (FC1 GEMM output)
# Output: out_fp8 [M, K] fp8, fp8_scale [M] f32

@triton.jit
def _tanh(x):
    """Manual tanh using exp for Triton compatibility."""
    # tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    x2 = 2.0 * tl.minimum(tl.maximum(x, -20.0), 20.0)
    e = tl.exp(x2)
    return (e - 1.0) / (e + 1.0)


@triton.jit
def _gelu_tanh(x):
    """GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"""
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + _tanh(inner))

@triton.jit
def fused_gelu_fp8_quant_kernel(
    x_ptr, out_fp8_ptr, fp8_scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BLOCK_K: tl.constexpr,
):
    """Single-pass fused GELU + FP8 quant for K <= 8192."""
    row = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)
    mask = k_offs < K

    x = tl.load(x_ptr + row * stride_xm + k_offs * stride_xk, mask=mask, other=0.0).to(tl.float32)

    # GELU tanh approximation
    gelu_out = _gelu_tanh(x)

    # FP8 quantization
    row_amax = tl.max(tl.abs(gelu_out), axis=0)
    fp8_sc = row_amax / 448.0
    fp8_sc = tl.maximum(fp8_sc, 1e-12)
    tl.store(fp8_scale_ptr + row, fp8_sc)

    x_fp8 = (gelu_out / fp8_sc).to(tl.float8e4nv)
    tl.store(out_fp8_ptr + row * stride_om + k_offs * stride_ok, x_fp8, mask=mask)


@triton.jit
def fused_gelu_fp8_quant_large_k_kernel(
    x_ptr, out_fp8_ptr, fp8_scale_ptr,
    M, K,
    stride_xm, stride_xk,
    stride_om, stride_ok,
    BLOCK_K: tl.constexpr,
):
    """Two-pass fused GELU + FP8 quant for K > 8192."""
    row = tl.program_id(0)

    # Pass 1: Apply GELU, compute amax
    row_amax = tl.zeros([1], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_ptr + row * stride_xm + k_offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
        gelu_out = _gelu_tanh(x)
        block_max = tl.max(tl.abs(gelu_out), axis=0)
        row_amax = tl.maximum(row_amax, block_max)

    fp8_sc = tl.sum(row_amax, axis=0) / 448.0
    fp8_sc = tl.maximum(fp8_sc, 1e-12)
    tl.store(fp8_scale_ptr + row, fp8_sc)

    # Pass 2: Apply GELU, quantize with scale
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_ptr + row * stride_xm + k_offs * stride_xk, mask=mask, other=0.0).to(tl.float32)
        gelu_out = _gelu_tanh(x)
        x_fp8 = (gelu_out / fp8_sc).to(tl.float8e4nv)
        tl.store(out_fp8_ptr + row * stride_om + k_offs * stride_ok, x_fp8, mask=mask)


def fused_gelu_fp8_quant(x):
    """
    Fused GELU (tanh approx) + FP8 rowwise quantization.

    Args:
        x: [M, K] bf16 input (FC1 GEMM output)
    Returns:
        out_fp8: [M, K] float8_e4m3fn
        fp8_scale: [M, 1] float32
    """
    M, K = x.shape
    out_fp8 = torch.empty(M, K, device=x.device, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_K = triton.next_power_of_2(K)

    if BLOCK_K <= 16384:
        # Single-pass: load all K elements into registers, apply GELU, hold outputs,
        # compute row max, then quantize — no second pass over the data.
        # For K in (8192, 16384] (e.g. K=12288) use num_warps=8 to keep 80 regs/thread
        # (vs 168 regs for num_warps=4), maintaining 3 blocks/SM occupancy while
        # reducing per-thread register pressure.  Benchmarked: 1.4x vs two-pass.
        fused_gelu_fp8_quant_kernel[(M,)](
            x, out_fp8, fp8_scale,
            M, K,
            x.stride(0), x.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=8 if BLOCK_K > 8192 else 4,
        )
    else:
        fused_gelu_fp8_quant_large_k_kernel[(M,)](
            x, out_fp8, fp8_scale,
            M, K,
            x.stride(0), x.stride(1),
            out_fp8.stride(0), out_fp8.stride(1),
            BLOCK_K=4096,
            num_warps=4,
        )

    return out_fp8, fp8_scale.unsqueeze(1)

# Kernel 5: Fused RMSNorm + RoPE
# Eliminates memory roundtrip between QK RMSNorm and RoPE application.
# Input: x [B, S, H, D] bf16, weight [D] f32, cos [S, D//2] f32, sin [S, D//2] f32
# Output: out [B, S, H, D] bf16

@triton.jit
def fused_rmsnorm_rope_kernel(
    x_ptr, weight_ptr, cos_ptr, sin_ptr, out_ptr,
    S, H, D,
    stride_x_row,    # input row stride (may be > H*D for non-contiguous views)
    stride_out_row,  # output row stride (always H*D for freshly-allocated out)
    stride_cos_row,  # D // 2
    eps: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """Fused RMSNorm + RoPE per (batch*seq, head) position."""
    row = tl.program_id(0)   # B*S
    head = tl.program_id(1)  # H
    seq_idx = row % S

    half_offs = tl.arange(0, HALF_D)

    in_base = row * stride_x_row + head * D
    out_base = row * stride_out_row + head * D

    # Load x as even/odd pairs
    x_even = tl.load(x_ptr + in_base + half_offs * 2).to(tl.float32)
    x_odd = tl.load(x_ptr + in_base + half_offs * 2 + 1).to(tl.float32)

    # RMSNorm: variance = mean(x^2)
    var = (tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)) / D
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load RMSNorm weight
    w_even = tl.load(weight_ptr + half_offs * 2).to(tl.float32)
    w_odd = tl.load(weight_ptr + half_offs * 2 + 1).to(tl.float32)

    # Normalize
    xn_even = x_even * rstd * w_even
    xn_odd = x_odd * rstd * w_odd

    # Load cos/sin
    cos = tl.load(cos_ptr + seq_idx * stride_cos_row + half_offs).to(tl.float32)
    sin = tl.load(sin_ptr + seq_idx * stride_cos_row + half_offs).to(tl.float32)

    # Apply RoPE rotation
    out_even = xn_even * cos - xn_odd * sin
    out_odd = xn_odd * cos + xn_even * sin

    # Store interleaved using output stride (always contiguous)
    tl.store(out_ptr + out_base + half_offs * 2, out_even.to(tl.bfloat16))
    tl.store(out_ptr + out_base + half_offs * 2 + 1, out_odd.to(tl.bfloat16))


def fused_rmsnorm_rope(x, weight, cos, sin, eps=1e-6):
    """
    Fused RMSNorm + RoPE application.

    Args:
        x: [B, S, H, D] bf16, contiguous
        weight: [D] RMSNorm weight parameter
        cos: [S, D//2] float32
        sin: [S, D//2] float32
        eps: RMSNorm epsilon
    Returns:
        out: [B, S, H, D] bf16
    """
    B, S, H, D = x.shape
    HALF_D = D // 2

    # x_flat may be a non-contiguous view (e.g. slice of fused QKV GEMM output
    # with stride > H*D).  We always allocate output as a fresh contiguous tensor
    # and pass separate read/write strides to the kernel to avoid out-of-bounds
    # writes that would occur if the kernel assumed equal strides.
    x_flat = x.reshape(B * S, H * D)         # may be non-contiguous view
    out = torch.empty(B * S, H * D, device=x.device, dtype=x.dtype)  # always contiguous

    grid = (B * S, H)
    fused_rmsnorm_rope_kernel[grid](
        x_flat, weight, cos, sin, out,
        S, H, D,
        x_flat.stride(0),   # read stride  (may be > H*D)
        H * D,              # write stride (always H*D, contiguous out)
        cos.stride(0),
        eps=eps,
        HALF_D=HALF_D,
        num_warps=2,
    )
    return out.view(B, S, H, D)


# Kernel 5b: Fused RMSNorm + RoPE for BOTH Q and K simultaneously
# Saves one kernel launch + shares cos/sin memory loads between Q and K.
# Input: q,k each [B*S, H*D] (may be non-contiguous views from QKV GEMM)
#        q_weight, k_weight [D] float32
#        cos, sin [S, D//2] float32
# Output: q_out, k_out [B*S, H*D] bf16 (always contiguous)

@triton.jit
def fused_rmsnorm_rope_qk_kernel(
    q_ptr, stride_q_row,
    k_ptr, stride_k_row,
    q_weight_ptr, k_weight_ptr,
    cos_ptr, sin_ptr,
    q_out_ptr, k_out_ptr,
    S, H, D,
    stride_out_row,   # = H*D (always contiguous output)
    stride_cos_row,   # = D//2
    eps: tl.constexpr,
    HALF_D: tl.constexpr,
):
    """Fused RMSNorm+RoPE for Q and K in a single kernel, sharing cos/sin loads."""
    row = tl.program_id(0)   # B*S
    head = tl.program_id(1)  # H
    seq_idx = row % S

    half_offs = tl.arange(0, HALF_D)

    in_base_q = row * stride_q_row + head * D
    in_base_k = row * stride_k_row + head * D
    out_base = row * stride_out_row + head * D

    # Load cos/sin ONCE — shared for both Q and K
    cos_base = seq_idx * stride_cos_row
    cos = tl.load(cos_ptr + cos_base + half_offs).to(tl.float32)
    sin = tl.load(sin_ptr + cos_base + half_offs).to(tl.float32)

    # === Process Q ===
    q_even = tl.load(q_ptr + in_base_q + half_offs * 2).to(tl.float32)
    q_odd  = tl.load(q_ptr + in_base_q + half_offs * 2 + 1).to(tl.float32)

    var_q  = (tl.sum(q_even * q_even, axis=0) + tl.sum(q_odd * q_odd, axis=0)) / D
    rstd_q = 1.0 / tl.sqrt(var_q + eps)

    wq_even = tl.load(q_weight_ptr + half_offs * 2).to(tl.float32)
    wq_odd  = tl.load(q_weight_ptr + half_offs * 2 + 1).to(tl.float32)

    qn_even = q_even * rstd_q * wq_even
    qn_odd  = q_odd  * rstd_q * wq_odd

    out_q_even = qn_even * cos - qn_odd * sin
    out_q_odd  = qn_odd  * cos + qn_even * sin

    tl.store(q_out_ptr + out_base + half_offs * 2,     out_q_even.to(tl.bfloat16))
    tl.store(q_out_ptr + out_base + half_offs * 2 + 1, out_q_odd.to(tl.bfloat16))

    # === Process K (reuse cos/sin already in registers) ===
    k_even = tl.load(k_ptr + in_base_k + half_offs * 2).to(tl.float32)
    k_odd  = tl.load(k_ptr + in_base_k + half_offs * 2 + 1).to(tl.float32)

    var_k  = (tl.sum(k_even * k_even, axis=0) + tl.sum(k_odd * k_odd, axis=0)) / D
    rstd_k = 1.0 / tl.sqrt(var_k + eps)

    wk_even = tl.load(k_weight_ptr + half_offs * 2).to(tl.float32)
    wk_odd  = tl.load(k_weight_ptr + half_offs * 2 + 1).to(tl.float32)

    kn_even = k_even * rstd_k * wk_even
    kn_odd  = k_odd  * rstd_k * wk_odd

    out_k_even = kn_even * cos - kn_odd * sin
    out_k_odd  = kn_odd  * cos + kn_even * sin

    tl.store(k_out_ptr + out_base + half_offs * 2,     out_k_even.to(tl.bfloat16))
    tl.store(k_out_ptr + out_base + half_offs * 2 + 1, out_k_odd.to(tl.bfloat16))


def fused_rmsnorm_rope_qk(q, k, q_weight, k_weight, cos, sin, eps=1e-6):
    """
    Fused RMSNorm + RoPE for Q and K simultaneously in one kernel launch.
    Saves: one kernel launch, duplicate cos/sin HBM loads.

    Args:
        q, k: [B, S, H, D] bf16 (may be non-contiguous views from fused QKV GEMM)
        q_weight, k_weight: [D] float32 RMSNorm weights
        cos, sin: [S, D//2] float32
        eps: RMSNorm epsilon
    Returns:
        q_out, k_out: [B, S, H, D] bf16 (always contiguous)
    """
    B, S, H, D = q.shape
    HALF_D = D // 2

    q_flat = q.reshape(B * S, H * D)
    k_flat = k.reshape(B * S, H * D)

    q_out = torch.empty(B * S, H * D, device=q.device, dtype=q.dtype)
    k_out = torch.empty(B * S, H * D, device=k.device, dtype=k.dtype)

    grid = (B * S, H)
    fused_rmsnorm_rope_qk_kernel[grid](
        q_flat, q_flat.stride(0),
        k_flat, k_flat.stride(0),
        q_weight, k_weight,
        cos, sin,
        q_out, k_out,
        S, H, D,
        H * D,          # stride_out_row (contiguous)
        cos.stride(0),  # stride_cos_row = D//2
        eps=eps,
        HALF_D=HALF_D,
        num_warps=4,    # 2x work per CTA vs single kernel → use 4 warps
    )
    return q_out.view(B, S, H, D), k_out.view(B, S, H, D)

# Kernel 5c: Fused RMSNorm + RoPE — all-heads variant
# grid=(B*S,): one CTA per sequence position, loops over all H heads.
# vs kernel 5b grid=(B*S, H): this version:
#   - Loads cos/sin ONCE per row (shared across all H heads) → 1/H the bandwidth
#   - Loads RMSNorm weights ONCE per row and keeps in registers
#   - 24× fewer CTAs → much less scheduling overhead on B200 (148 SMs)

@triton.jit
def fused_rmsnorm_rope_qk_allheads_kernel(
    q_ptr, stride_q_row,
    k_ptr, stride_k_row,
    q_weight_ptr, k_weight_ptr,   # [D] per-head weights (shared across heads)
    cos_ptr, sin_ptr,             # [S, D//2] float32
    q_out_ptr, k_out_ptr,         # [B*S, H*D] contiguous output
    S, H, D,
    stride_out_row,               # = H*D
    stride_cos_row,               # = D//2
    eps: tl.constexpr,
    HALF_D: tl.constexpr,         # D // 2
):
    """One CTA per (batch*seq) row; loop over all H heads, sharing cos/sin."""
    row     = tl.program_id(0)
    seq_idx = row % S

    half_offs = tl.arange(0, HALF_D)
    cos_base  = seq_idx * stride_cos_row

    # Load cos/sin ONCE for this sequence position — reused for all H heads
    cos = tl.load(cos_ptr + cos_base + half_offs).to(tl.float32)
    sin = tl.load(sin_ptr + cos_base + half_offs).to(tl.float32)

    # Load RMSNorm weights ONCE — same weight vector shared by all heads
    wq_even = tl.load(q_weight_ptr + half_offs * 2).to(tl.float32)
    wq_odd  = tl.load(q_weight_ptr + half_offs * 2 + 1).to(tl.float32)
    wk_even = tl.load(k_weight_ptr + half_offs * 2).to(tl.float32)
    wk_odd  = tl.load(k_weight_ptr + half_offs * 2 + 1).to(tl.float32)

    base_q   = row * stride_q_row
    base_k   = row * stride_k_row
    out_base = row * stride_out_row

    for h in range(H):
        head_off = h * D

        # ----- Q -----
        q_even = tl.load(q_ptr + base_q + head_off + half_offs * 2).to(tl.float32)
        q_odd  = tl.load(q_ptr + base_q + head_off + half_offs * 2 + 1).to(tl.float32)

        var_q  = (tl.sum(q_even * q_even, axis=0) + tl.sum(q_odd * q_odd, axis=0)) / D
        rstd_q = 1.0 / tl.sqrt(var_q + eps)

        qn_even = q_even * rstd_q * wq_even
        qn_odd  = q_odd  * rstd_q * wq_odd

        oh = out_base + head_off
        tl.store(q_out_ptr + oh + half_offs * 2,     (qn_even * cos - qn_odd * sin).to(tl.bfloat16))
        tl.store(q_out_ptr + oh + half_offs * 2 + 1, (qn_odd  * cos + qn_even * sin).to(tl.bfloat16))

        # ----- K -----
        k_even = tl.load(k_ptr + base_k + head_off + half_offs * 2).to(tl.float32)
        k_odd  = tl.load(k_ptr + base_k + head_off + half_offs * 2 + 1).to(tl.float32)

        var_k  = (tl.sum(k_even * k_even, axis=0) + tl.sum(k_odd * k_odd, axis=0)) / D
        rstd_k = 1.0 / tl.sqrt(var_k + eps)

        kn_even = k_even * rstd_k * wk_even
        kn_odd  = k_odd  * rstd_k * wk_odd

        tl.store(k_out_ptr + oh + half_offs * 2,     (kn_even * cos - kn_odd * sin).to(tl.bfloat16))
        tl.store(k_out_ptr + oh + half_offs * 2 + 1, (kn_odd  * cos + kn_even * sin).to(tl.bfloat16))


def fused_rmsnorm_rope_qk_allheads(q, k, q_weight, k_weight, cos, sin, eps=1e-6):
    """
    All-heads variant of fused_rmsnorm_rope_qk.
    grid=(B*S,): one CTA per sequence position processes all H heads.
    Loads cos/sin and RMSNorm weights once per row, keeping them in registers.

    Args:
        q, k: [B, S, H, D] bf16 (or non-contiguous views from fused QKV GEMM)
        q_weight, k_weight: [D] float32 RMSNorm weights (shared across heads)
        cos, sin: [S, D//2] float32
        eps: RMSNorm epsilon
    Returns:
        q_out, k_out: [B, S, H, D] bf16 (always contiguous)
    """
    B, S, H, D = q.shape
    HALF_D = D // 2

    q_flat = q.reshape(B * S, H * D)
    k_flat = k.reshape(B * S, H * D)

    q_out = torch.empty(B * S, H * D, device=q.device, dtype=q.dtype)
    k_out = torch.empty(B * S, H * D, device=k.device, dtype=k.dtype)

    grid = (B * S,)
    fused_rmsnorm_rope_qk_allheads_kernel[grid](
        q_flat, q_flat.stride(0),
        k_flat, k_flat.stride(0),
        q_weight, k_weight,
        cos, sin,
        q_out, k_out,
        S, H, D,
        H * D,          # stride_out_row
        cos.stride(0),  # stride_cos_row = D//2
        eps=eps,
        HALF_D=HALF_D,
        num_warps=4,    # 128 threads; 64 elements per load → good occupancy
    )
    return q_out.view(B, S, H, D), k_out.view(B, S, H, D)


def fused_rope_fast(x, cos, sin):
    """
    Fast Triton RoPE for contiguous [B, S, H, D] tensors with pre-decomposed cos/sin.

    Args:
        x: [B, S, H, D] bf16, contiguous
        cos: [S, D//2] float32
        sin: [S, D//2] float32
    Returns:
        out: [B, S, H, D] bf16
    """
    B, S, H, D = x.shape
    HALF_D = D // 2
    out = torch.empty_like(x)

    # Reshape to [B*S, H*D] for contiguous access
    x_flat = x.reshape(B * S, H * D)
    out_flat = out.reshape(B * S, H * D)

    BLOCK_HD = triton.next_power_of_2(H * HALF_D)
    grid = (B * S,)

    fused_rope_fast_kernel[grid](
        x_flat, cos, sin, out_flat,
        S, H, D, HALF_D,
        x_flat.stride(0),
        cos.stride(0),
        BLOCK_HD=BLOCK_HD,
        num_warps=8,
    )
    return out