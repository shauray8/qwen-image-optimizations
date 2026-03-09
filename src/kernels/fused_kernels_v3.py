import torch
import triton
import triton.language as tl

@triton.jit
def fused_qkv_rmsnorm_rope_pack_kernel(
    # QKV GEMM output: [B*seq_len, 3*H*D], stride=(3*H*D, 1)
    qkv_ptr,
    # Joint buffer outputs: [B, joint_seq, H, D], stride=(joint_seq*H*D, H*D, D, 1)
    jq_ptr, jk_ptr, jv_ptr,
    # RMSNorm weights: [D] per-head (same across all heads)
    q_weight_ptr, k_weight_ptr,
    # RoPE tables: [seq_len, HALF_D]
    cos_ptr, sin_ptr,
    # Sizes
    seq_len,        # img_seq or txt_seq
    H, D,           # num_heads, head_dim
    HxD,            # H * D (precomputed to avoid multiply in kernel)
    joint_seq,      # txt_seq + img_seq
    joint_offset,   # 0 for txt, txt_seq for img
    stride_cos_row, # HALF_D
    eps: tl.constexpr,
    HALF_D: tl.constexpr,  # D // 2 = 64
):
    row  = tl.program_id(0)   # index into B * seq_len
    head = tl.program_id(1)   # head index

    batch   = row // seq_len
    seq_idx = row % seq_len

    half_offs = tl.arange(0, HALF_D)  # 0..63

    # Row stride = 3 * H * D; layout: [Q_all_heads | K_all_heads | V_all_heads]
    qkv_row_base = row * 3 * HxD + head * D
    q_src = qkv_row_base           # Q for this head
    k_src = qkv_row_base + HxD    # K for this head
    v_src = qkv_row_base + 2 * HxD  # V for this head

    # Destination addresses in joint buffer [B, joint_seq, H, D] 
    # Stride = (joint_seq*HxD, HxD, D, 1); layout: [txt_tokens | img_tokens]
    dst_seq  = joint_offset + seq_idx
    dst_base = (batch * joint_seq + dst_seq) * HxD + head * D

    # Load cos/sin for this sequence position
    cos_base = seq_idx * stride_cos_row
    cos = tl.load(cos_ptr + cos_base + half_offs).to(tl.float32)
    sin = tl.load(sin_ptr + cos_base + half_offs).to(tl.float32)

    q_even = tl.load(qkv_ptr + q_src + half_offs * 2).to(tl.float32)
    q_odd  = tl.load(qkv_ptr + q_src + half_offs * 2 + 1).to(tl.float32)

    # RMSNorm: variance over all D elements (even + odd)
    var_q  = (tl.sum(q_even * q_even, axis=0) + tl.sum(q_odd * q_odd, axis=0)) / D
    rstd_q = 1.0 / tl.sqrt(var_q + eps)

    wq_even = tl.load(q_weight_ptr + half_offs * 2).to(tl.float32)
    wq_odd  = tl.load(q_weight_ptr + half_offs * 2 + 1).to(tl.float32)

    qn_even = q_even * rstd_q * wq_even
    qn_odd  = q_odd  * rstd_q * wq_odd

    # RoPE rotation: [q_e, q_o] -> [q_e*cos - q_o*sin, q_o*cos + q_e*sin]
    rq_even = qn_even * cos - qn_odd * sin
    rq_odd  = qn_odd  * cos + qn_even * sin

    tl.store(jq_ptr + dst_base + half_offs * 2,     rq_even.to(tl.bfloat16))
    tl.store(jq_ptr + dst_base + half_offs * 2 + 1, rq_odd.to(tl.bfloat16))

    k_even = tl.load(qkv_ptr + k_src + half_offs * 2).to(tl.float32)
    k_odd  = tl.load(qkv_ptr + k_src + half_offs * 2 + 1).to(tl.float32)

    var_k  = (tl.sum(k_even * k_even, axis=0) + tl.sum(k_odd * k_odd, axis=0)) / D
    rstd_k = 1.0 / tl.sqrt(var_k + eps)

    wk_even = tl.load(k_weight_ptr + half_offs * 2).to(tl.float32)
    wk_odd  = tl.load(k_weight_ptr + half_offs * 2 + 1).to(tl.float32)

    kn_even = k_even * rstd_k * wk_even
    kn_odd  = k_odd  * rstd_k * wk_odd

    rk_even = kn_even * cos - kn_odd * sin
    rk_odd  = kn_odd  * cos + kn_even * sin

    tl.store(jk_ptr + dst_base + half_offs * 2,     rk_even.to(tl.bfloat16))
    tl.store(jk_ptr + dst_base + half_offs * 2 + 1, rk_odd.to(tl.bfloat16))

    # Load all D elements at once (unit stride, fully coalesced)
    d_offs = tl.arange(0, HALF_D * 2)  # 0..127 (= D elements)
    v_vals = tl.load(qkv_ptr + v_src + d_offs)  # bfloat16, unit stride
    tl.store(jv_ptr + dst_base + d_offs, v_vals)

@triton.jit
def fused_attn_split_fp8_quant_kernel(
    # Input: joint attention output [B, joint_seq, D], contiguous
    joint_ptr,
    # Outputs:
    img_fp8_ptr,   # [B*img_seq, D] float8_e4m3fn
    img_sc_ptr,    # [B*img_seq]   float32 per-row scale
    txt_fp8_ptr,   # [B*txt_seq, D] float8_e4m3fn
    txt_sc_ptr,    # [B*txt_seq]   float32 per-row scale
    # Sizes
    txt_seq, img_seq, joint_seq, D,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (B * joint_seq,) — one program per (batch, seq) position.
    Each program reads D elements from joint_out, quantizes to FP8, and
    writes to the appropriate (img or txt) output buffer.
    """
    pid = tl.program_id(0)
    batch   = pid // joint_seq
    seq_pos = pid % joint_seq

    d_offs = tl.arange(0, BLOCK_D)
    mask   = d_offs < D

    # Source: joint_out[batch, seq_pos, :]  stride=(joint_seq*D, D, 1)
    src_base = (batch * joint_seq + seq_pos) * D
    x = tl.load(joint_ptr + src_base + d_offs, mask=mask, other=0.0).to(tl.float32)

    # Per-row FP8 quantization
    row_amax = tl.max(tl.abs(x), axis=0)
    fp8_scale = row_amax / 448.0
    fp8_scale = tl.maximum(fp8_scale, 1e-12)
    x_fp8 = (x / fp8_scale).to(tl.float8e4nv)

    is_txt = seq_pos < txt_seq

    if is_txt:
        # Write to txt_fp8[batch * txt_seq + seq_pos, :]
        dst_row = batch * txt_seq + seq_pos
        tl.store(txt_fp8_ptr + dst_row * D + d_offs, x_fp8, mask=mask)
        tl.store(txt_sc_ptr  + dst_row, fp8_scale)
    else:
        # Write to img_fp8[batch * img_seq + (seq_pos - txt_seq), :]
        dst_row = batch * img_seq + (seq_pos - txt_seq)
        tl.store(img_fp8_ptr + dst_row * D + d_offs, x_fp8, mask=mask)
        tl.store(img_sc_ptr  + dst_row, fp8_scale)


def fused_attn_split_fp8_quant(
    joint_out,   # [B, joint_seq, D] contiguous bfloat16
    txt_seq,
    img_seq,
    B,
    joint_seq,
    D,
):
    """
    Fused: split flash-attn output by txt/img + FP8 quantize both streams.

    Replaces per block:
      - 2 × implicit .contiguous() copies (non-contiguous attn slice → FP8 quant input)
      - 2 × fused_fp8_quant_kernel calls
    With 1 efficient Triton kernel with coalesced reads and contiguous writes.

    Returns:
        img_fp8  [B*img_seq, D]  float8_e4m3fn
        img_sc   [B*img_seq, 1] float32  (unsqueezed for _scaled_mm scale_a)
        txt_fp8  [B*txt_seq, D]  float8_e4m3fn
        txt_sc   [B*txt_seq, 1] float32
    """
    assert joint_out.is_contiguous(), "joint_out must be contiguous"
    BLOCK_D = triton.next_power_of_2(D)

    img_fp8 = torch.empty(B * img_seq, D, device=joint_out.device, dtype=torch.float8_e4m3fn)
    img_sc  = torch.empty(B * img_seq, device=joint_out.device, dtype=torch.float32)
    txt_fp8 = torch.empty(B * txt_seq, D, device=joint_out.device, dtype=torch.float8_e4m3fn)
    txt_sc  = torch.empty(B * txt_seq, device=joint_out.device, dtype=torch.float32)

    grid = (B * joint_seq,)
    fused_attn_split_fp8_quant_kernel[grid](
        joint_out,
        img_fp8, img_sc,
        txt_fp8, txt_sc,
        txt_seq, img_seq, joint_seq, D,
        BLOCK_D=BLOCK_D,
        num_warps=2,  # For D=3072 BLOCK_D=4096: 64 threads × 64 elem/thread; benchmarked 2.95× faster than 32 warps
    )

    return img_fp8, img_sc.unsqueeze(1), txt_fp8, txt_sc.unsqueeze(1)


def fused_qkv_rmsnorm_rope_pack(
    qkv,          # [B*seq_len, 3*H*D] contiguous bfloat16, from FusedQKVFP8Linear
    joint_q,      # [B, joint_seq, H, D] pre-allocated bfloat16
    joint_k,      # [B, joint_seq, H, D]
    joint_v,      # [B, joint_seq, H, D]
    q_weight,     # [D] bfloat16 RMSNorm weight for Q (per-head, head_dim=D)
    k_weight,     # [D] bfloat16 RMSNorm weight for K
    cos,          # [seq_len, D//2] float32 RoPE cosines
    sin,          # [seq_len, D//2] float32 RoPE sines
    B,            # batch size
    seq_len,      # img_seq or txt_seq
    H,            # num_heads
    D,            # head_dim
    joint_seq,    # txt_seq + img_seq
    joint_offset, # 0 for txt, txt_seq for img
    eps=1e-6,
):
    """
    Fused: read QKV from contiguous GEMM output → RMSNorm+RoPE(Q,K) → write to joint bufs.

    Eliminates:
      - Non-contiguous view surgery on QKV GEMM output
      - fused_rmsnorm_rope_qk kernel call (strided reads, extra allocation)
      - 2 × .copy_() per stream into joint buffer (non-contiguous dest)

    Replaces all 3 with a single Triton kernel with coalesced reads/writes.
    """
    assert qkv.is_contiguous(), "QKV GEMM output must be contiguous"
    HALF_D = D // 2
    HxD = H * D

    grid = (B * seq_len, H)
    fused_qkv_rmsnorm_rope_pack_kernel[grid](
        qkv,
        joint_q, joint_k, joint_v,
        q_weight, k_weight,
        cos, sin,
        seq_len, H, D, HxD,
        joint_seq, joint_offset,
        cos.stride(0),  # stride_cos_row = HALF_D
        eps=eps,
        HALF_D=HALF_D,
        num_warps=1,   # 32 threads: HALF_D=64 → 2 elem/thread; min waves (19 vs 83 for warps=4)
    )