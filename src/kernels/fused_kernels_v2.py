import torch
import triton
import triton.language as tl

FP8_MAX = 448.0  # max of float8_e4m3fn

def _best_block_d(D: int) -> int:
    """Pick power-of-2 BLOCK_D that fits D, ≤ 8192."""
    b = triton.next_power_of_2(D)
    return min(b, 8192)

@triton.jit
def _dual_adaln_fp8_kernel(
    # img stream
    img_ptr, img_shift_ptr, img_scale_ptr,
    img_fp8_ptr, img_fp8sc_ptr,
    M_img, L_img,                       # M_img = B*L_img
    stride_ix, stride_iy,               # x=row, y=col strides
    stride_is, stride_id,               # shift/scale row/col strides (B, D)
    stride_oy, stride_od,               # out strides
    # txt stream
    txt_ptr, txt_shift_ptr, txt_scale_ptr,
    txt_fp8_ptr, txt_fp8sc_ptr,
    M_txt, L_txt,
    stride_tx, stride_ty,
    stride_ts, stride_td,
    stride_toy, stride_tod,
    # shared
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (M_img + M_txt,)
    Each CTA handles one row: if row_id < M_img → img stream, else txt stream.
    """
    row = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    if row < M_img:
        # ── img stream ────────────────────────────────────────────────────
        batch_idx = row // L_img
        x = tl.load(img_ptr + row * stride_ix + d_offs * stride_iy, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) / D
        xc = x - mean
        var  = tl.sum(xc * xc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        xn = xc * rstd

        shift = tl.load(img_shift_ptr + batch_idx * stride_is + d_offs * stride_id, mask=mask, other=0.0).to(tl.float32)
        sc    = tl.load(img_scale_ptr + batch_idx * stride_is + d_offs * stride_id, mask=mask, other=0.0).to(tl.float32)
        xm = xn * (1.0 + sc) + shift

        amax = tl.max(tl.abs(xm), axis=0)
        fp8_sc = tl.maximum(amax / 448.0, 1e-12)
        tl.store(img_fp8sc_ptr + row, fp8_sc)
        tl.store(img_fp8_ptr + row * stride_oy + d_offs * stride_od, (xm / fp8_sc).to(tl.float8e4nv), mask=mask)
    else:
        # ── txt stream ────────────────────────────────────────────────────
        trow = row - M_img
        batch_idx = trow // L_txt
        x = tl.load(txt_ptr + trow * stride_tx + d_offs * stride_ty, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) / D
        xc = x - mean
        var  = tl.sum(xc * xc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        xn = xc * rstd

        shift = tl.load(txt_shift_ptr + batch_idx * stride_ts + d_offs * stride_td, mask=mask, other=0.0).to(tl.float32)
        sc    = tl.load(txt_scale_ptr + batch_idx * stride_ts + d_offs * stride_td, mask=mask, other=0.0).to(tl.float32)
        xm = xn * (1.0 + sc) + shift

        amax = tl.max(tl.abs(xm), axis=0)
        fp8_sc = tl.maximum(amax / 448.0, 1e-12)
        tl.store(txt_fp8sc_ptr + trow, fp8_sc)
        tl.store(txt_fp8_ptr + trow * stride_toy + d_offs * stride_tod, (xm / fp8_sc).to(tl.float8e4nv), mask=mask)


def fused_dual_adaln_fp8(
    img: torch.Tensor, img_shift: torch.Tensor, img_scale: torch.Tensor, L_img: int,
    txt: torch.Tensor, txt_shift: torch.Tensor, txt_scale: torch.Tensor, L_txt: int,
    eps: float = 1e-6,
):
    """
    Single kernel for both img and txt AdaLN + FP8 quantization.

    img: [B*L_img, D] bfloat16
    txt: [B*L_txt, D] bfloat16
    img_shift, img_scale: [B, D] bfloat16
    txt_shift, txt_scale: [B, D] bfloat16

    Returns: img_fp8, img_scale [B*L_img, D]/[B*L_img], txt_fp8, txt_scale [B*L_txt, D]/[B*L_txt]
    """
    M_img, D = img.shape
    M_txt     = txt.shape[0]
    assert D <= 8192, f"D={D} > 8192, need large-D path"

    img_fp8   = torch.empty(M_img, D, device=img.device, dtype=torch.float8_e4m3fn)
    img_fp8sc = torch.empty(M_img,    device=img.device, dtype=torch.float32)
    txt_fp8   = torch.empty(M_txt, D, device=txt.device, dtype=torch.float8_e4m3fn)
    txt_fp8sc = torch.empty(M_txt,    device=txt.device, dtype=torch.float32)

    BLOCK_D = _best_block_d(D)
    grid = (M_img + M_txt,)

    _dual_adaln_fp8_kernel[grid](
        img, img_shift, img_scale, img_fp8, img_fp8sc,
        M_img, L_img,
        img.stride(0), img.stride(1),
        img_shift.stride(0), img_shift.stride(1),
        img_fp8.stride(0), img_fp8.stride(1),
        txt, txt_shift, txt_scale, txt_fp8, txt_fp8sc,
        M_txt, L_txt,
        txt.stride(0), txt.stride(1),
        txt_shift.stride(0), txt_shift.stride(1),
        txt_fp8.stride(0), txt_fp8.stride(1),
        D=D, eps=eps,
        BLOCK_D=BLOCK_D,
        num_warps=max(1, min(32, BLOCK_D // 2048)),  # BLOCK_D=4096→2 warps; benchmarked best for D=3072
    )
    return img_fp8, img_fp8sc.unsqueeze(1), txt_fp8, txt_fp8sc.unsqueeze(1)

@triton.jit
def _dual_residual_gate_adaln_fp8_kernel(
    # img stream
    img_res_ptr, img_gate_ptr, img_delta_ptr,
    img_shift2_ptr, img_scale2_ptr,
    img_res_out_ptr,                    # updated residual (stored for next block)
    img_fp8_ptr, img_fp8sc_ptr,
    M_img, L_img,
    stride_irx, stride_iry,
    stride_igx, stride_igd,             # gate: [B, D]
    stride_iop, stride_iod,
    stride_i2s, stride_i2d,
    stride_iox, stride_ioy,
    stride_ofpx, stride_ofpy,
    # txt stream
    txt_res_ptr, txt_gate_ptr, txt_delta_ptr,
    txt_shift2_ptr, txt_scale2_ptr,
    txt_res_out_ptr,
    txt_fp8_ptr, txt_fp8sc_ptr,
    M_txt, L_txt,
    stride_trx, stride_try,
    stride_tgx, stride_tgd,
    stride_top, stride_tod,
    stride_t2s, stride_t2d,
    stride_tox, stride_toy,
    stride_tfpx, stride_tfpy,
    # shared
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    if row < M_img:
        batch_idx = row // L_img

        res   = tl.load(img_res_ptr   + row * stride_irx + d_offs * stride_iry,  mask=mask, other=0.0).to(tl.float32)
        gate  = tl.load(img_gate_ptr  + batch_idx * stride_igx + d_offs * stride_igd, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(img_delta_ptr + row * stride_iop + d_offs * stride_iod,  mask=mask, other=0.0).to(tl.float32)

        h = res + gate * delta
        tl.store(img_res_out_ptr + row * stride_iox + d_offs * stride_ioy, h.to(tl.bfloat16), mask=mask)

        # LayerNorm
        mean = tl.sum(h, axis=0) / D
        hc = h - mean
        var  = tl.sum(hc * hc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        hn = hc * rstd

        shift = tl.load(img_shift2_ptr + batch_idx * stride_i2s + d_offs * stride_i2d, mask=mask, other=0.0).to(tl.float32)
        sc    = tl.load(img_scale2_ptr + batch_idx * stride_i2s + d_offs * stride_i2d, mask=mask, other=0.0).to(tl.float32)
        hm = hn * (1.0 + sc) + shift

        amax = tl.max(tl.abs(hm), axis=0)
        fp8_sc = tl.maximum(amax / 448.0, 1e-12)
        tl.store(img_fp8sc_ptr + row, fp8_sc)
        tl.store(img_fp8_ptr + row * stride_ofpx + d_offs * stride_ofpy, (hm / fp8_sc).to(tl.float8e4nv), mask=mask)

    else:
        trow = row - M_img
        batch_idx = trow // L_txt

        res   = tl.load(txt_res_ptr   + trow * stride_trx + d_offs * stride_try, mask=mask, other=0.0).to(tl.float32)
        gate  = tl.load(txt_gate_ptr  + batch_idx * stride_tgx + d_offs * stride_tgd, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(txt_delta_ptr + trow * stride_top + d_offs * stride_tod,  mask=mask, other=0.0).to(tl.float32)

        h = res + gate * delta
        tl.store(txt_res_out_ptr + trow * stride_tox + d_offs * stride_toy, h.to(tl.bfloat16), mask=mask)

        mean = tl.sum(h, axis=0) / D
        hc = h - mean
        var  = tl.sum(hc * hc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        hn = hc * rstd

        shift = tl.load(txt_shift2_ptr + batch_idx * stride_t2s + d_offs * stride_t2d, mask=mask, other=0.0).to(tl.float32)
        sc    = tl.load(txt_scale2_ptr + batch_idx * stride_t2s + d_offs * stride_t2d, mask=mask, other=0.0).to(tl.float32)
        hm = hn * (1.0 + sc) + shift

        amax = tl.max(tl.abs(hm), axis=0)
        fp8_sc = tl.maximum(amax / 448.0, 1e-12)
        tl.store(txt_fp8sc_ptr + trow, fp8_sc)
        tl.store(txt_fp8_ptr + trow * stride_tfpx + d_offs * stride_tfpy, (hm / fp8_sc).to(tl.float8e4nv), mask=mask)


def fused_dual_residual_gate_adaln_fp8(
    img_res: torch.Tensor, img_gate: torch.Tensor, img_delta: torch.Tensor,
    img_shift2: torch.Tensor, img_scale2: torch.Tensor, L_img: int,
    txt_res: torch.Tensor, txt_gate: torch.Tensor, txt_delta: torch.Tensor,
    txt_shift2: torch.Tensor, txt_scale2: torch.Tensor, L_txt: int,
    eps: float = 1e-6,
):
    """
    Single kernel for both img+txt: residual+gate+adaln+fp8.
    Returns: img_res_out, img_fp8, img_fp8sc, txt_res_out, txt_fp8, txt_fp8sc
    """
    M_img, D = img_res.shape
    M_txt     = txt_res.shape[0]

    img_res_out = torch.empty_like(img_res)
    img_fp8     = torch.empty(M_img, D, device=img_res.device, dtype=torch.float8_e4m3fn)
    img_fp8sc   = torch.empty(M_img,   device=img_res.device,  dtype=torch.float32)
    txt_res_out = torch.empty_like(txt_res)
    txt_fp8     = torch.empty(M_txt, D, device=txt_res.device, dtype=torch.float8_e4m3fn)
    txt_fp8sc   = torch.empty(M_txt,   device=txt_res.device,  dtype=torch.float32)

    BLOCK_D = _best_block_d(D)
    grid = (M_img + M_txt,)

    _dual_residual_gate_adaln_fp8_kernel[grid](
        img_res, img_gate, img_delta, img_shift2, img_scale2,
        img_res_out, img_fp8, img_fp8sc,
        M_img, L_img,
        img_res.stride(0), img_res.stride(1),
        img_gate.stride(0), img_gate.stride(1),
        img_delta.stride(0), img_delta.stride(1),
        img_shift2.stride(0), img_shift2.stride(1),
        img_res_out.stride(0), img_res_out.stride(1),
        img_fp8.stride(0), img_fp8.stride(1),
        txt_res, txt_gate, txt_delta, txt_shift2, txt_scale2,
        txt_res_out, txt_fp8, txt_fp8sc,
        M_txt, L_txt,
        txt_res.stride(0), txt_res.stride(1),
        txt_gate.stride(0), txt_gate.stride(1),
        txt_delta.stride(0), txt_delta.stride(1),
        txt_shift2.stride(0), txt_shift2.stride(1),
        txt_res_out.stride(0), txt_res_out.stride(1),
        txt_fp8.stride(0), txt_fp8.stride(1),
        D=D, eps=eps,
        BLOCK_D=BLOCK_D,
        num_warps=max(1, min(32, BLOCK_D // 2048)),  # BLOCK_D=4096→2 warps; benchmarked best for D=3072
    )
    return img_res_out, img_fp8, img_fp8sc.unsqueeze(1), txt_res_out, txt_fp8, txt_fp8sc.unsqueeze(1)


@triton.jit
def _dual_residual_gate_kernel(
    img_res_ptr, img_gate_ptr, img_delta_ptr, img_out_ptr,
    M_img, L_img,
    stride_irx, stride_iry, stride_igx, stride_igd, stride_iop, stride_iod,
    txt_res_ptr, txt_gate_ptr, txt_delta_ptr, txt_out_ptr,
    M_txt, L_txt,
    stride_trx, stride_try, stride_tgx, stride_tgd, stride_top, stride_tod,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    if row < M_img:
        batch_idx = row // L_img
        res   = tl.load(img_res_ptr   + row * stride_irx + d_offs * stride_iry, mask=mask, other=0.0).to(tl.float32)
        gate  = tl.load(img_gate_ptr  + batch_idx * stride_igx + d_offs * stride_igd, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(img_delta_ptr + row * stride_iop + d_offs * stride_iod, mask=mask, other=0.0).to(tl.float32)
        tl.store(img_out_ptr + row * stride_irx + d_offs * stride_iry, (res + gate * delta).to(tl.bfloat16), mask=mask)
    else:
        trow = row - M_img
        batch_idx = trow // L_txt
        res   = tl.load(txt_res_ptr   + trow * stride_trx + d_offs * stride_try, mask=mask, other=0.0).to(tl.float32)
        gate  = tl.load(txt_gate_ptr  + batch_idx * stride_tgx + d_offs * stride_tgd, mask=mask, other=0.0).to(tl.float32)
        delta = tl.load(txt_delta_ptr + trow * stride_top + d_offs * stride_tod, mask=mask, other=0.0).to(tl.float32)
        tl.store(txt_out_ptr + trow * stride_trx + d_offs * stride_try, (res + gate * delta).to(tl.bfloat16), mask=mask)


def fused_dual_residual_gate(
    img_res: torch.Tensor, img_gate: torch.Tensor, img_delta: torch.Tensor, L_img: int,
    txt_res: torch.Tensor, txt_gate: torch.Tensor, txt_delta: torch.Tensor, L_txt: int,
):
    """
    In-place residual+gate for both streams in one kernel launch.
    img_res and txt_res are modified in-place.
    Returns (img_res, txt_res) [modified in-place].
    """
    M_img, D = img_res.shape
    M_txt     = txt_res.shape[0]
    BLOCK_D = _best_block_d(D)

    _dual_residual_gate_kernel[(M_img + M_txt,)](
        img_res, img_gate, img_delta, img_res,
        M_img, L_img,
        img_res.stride(0), img_res.stride(1),
        img_gate.stride(0), img_gate.stride(1),
        img_delta.stride(0), img_delta.stride(1),
        txt_res, txt_gate, txt_delta, txt_res,
        M_txt, L_txt,
        txt_res.stride(0), txt_res.stride(1),
        txt_gate.stride(0), txt_gate.stride(1),
        txt_delta.stride(0), txt_delta.stride(1),
        D=D,
        BLOCK_D=BLOCK_D,
        num_warps=min(16, BLOCK_D // 32),
    )
    return img_res, txt_res

@triton.jit
def _cross_block_residual_gate_adaln_fp8_kernel(
    img_res_ptr, img_gate_ptr, img_mlp_ptr,
    img_shift_next_ptr, img_scale_next_ptr,
    img_res_out_ptr, img_fp8_ptr, img_fp8sc_ptr,
    M_img, L_img,
    stride_irx, stride_iry,
    stride_igx, stride_igd,
    stride_iml, stride_imd,
    stride_ins, stride_ind,
    stride_iox, stride_ioy,
    stride_ifx, stride_ify,
    txt_res_ptr, txt_gate_ptr, txt_mlp_ptr,
    txt_shift_next_ptr, txt_scale_next_ptr,
    txt_res_out_ptr, txt_fp8_ptr, txt_fp8sc_ptr,
    M_txt, L_txt,
    stride_trx, stride_try,
    stride_tgx, stride_tgd,
    stride_tml, stride_tmd,
    stride_tns, stride_tnd,
    stride_tox, stride_toy,
    stride_tfx, stride_tfy,
    D: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (M_img + M_txt,)
    Each CTA processes one row of either img or txt stream.

    For each row:
      1. h = res + gate * mlp_out        (residual + gated MLP output)
      2. Store h (updated residual)
      3. ln = LayerNorm(h)              (no affine)
      4. hm = ln * (1 + scale) + shift  (AdaLN modulation for NEXT block)
      5. Quantize hm to FP8
    """
    row = tl.program_id(0)
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < D

    if row < M_img:
        batch_idx = row // L_img

        res  = tl.load(img_res_ptr  + row * stride_irx + d_offs * stride_iry,  mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(img_gate_ptr + batch_idx * stride_igx + d_offs * stride_igd, mask=mask, other=0.0).to(tl.float32)
        mlp  = tl.load(img_mlp_ptr  + row * stride_iml + d_offs * stride_imd,  mask=mask, other=0.0).to(tl.float32)

        h = res + gate * mlp
        tl.store(img_res_out_ptr + row * stride_iox + d_offs * stride_ioy, h.to(tl.bfloat16), mask=mask)

        mean = tl.sum(h, axis=0) / D
        hc = h - mean
        var  = tl.sum(hc * hc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        hn = hc * rstd

        shift = tl.load(img_shift_next_ptr + batch_idx * stride_ins + d_offs * stride_ind, mask=mask, other=0.0).to(tl.float32)
        sc    = tl.load(img_scale_next_ptr + batch_idx * stride_ins + d_offs * stride_ind, mask=mask, other=0.0).to(tl.float32)
        hm = hn * (1.0 + sc) + shift

        amax = tl.max(tl.abs(hm), axis=0)
        fp8_sc = tl.maximum(amax / 448.0, 1e-12)
        tl.store(img_fp8sc_ptr + row, fp8_sc)
        tl.store(img_fp8_ptr + row * stride_ifx + d_offs * stride_ify, (hm / fp8_sc).to(tl.float8e4nv), mask=mask)
    else:
        trow = row - M_img
        batch_idx = trow // L_txt

        res  = tl.load(txt_res_ptr  + trow * stride_trx + d_offs * stride_try, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(txt_gate_ptr + batch_idx * stride_tgx + d_offs * stride_tgd, mask=mask, other=0.0).to(tl.float32)
        mlp  = tl.load(txt_mlp_ptr  + trow * stride_tml + d_offs * stride_tmd, mask=mask, other=0.0).to(tl.float32)

        h = res + gate * mlp
        tl.store(txt_res_out_ptr + trow * stride_tox + d_offs * stride_toy, h.to(tl.bfloat16), mask=mask)

        mean = tl.sum(h, axis=0) / D
        hc = h - mean
        var  = tl.sum(hc * hc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        hn = hc * rstd

        shift = tl.load(txt_shift_next_ptr + batch_idx * stride_tns + d_offs * stride_tnd, mask=mask, other=0.0).to(tl.float32)
        sc    = tl.load(txt_scale_next_ptr + batch_idx * stride_tns + d_offs * stride_tnd, mask=mask, other=0.0).to(tl.float32)
        hm = hn * (1.0 + sc) + shift

        amax = tl.max(tl.abs(hm), axis=0)
        fp8_sc = tl.maximum(amax / 448.0, 1e-12)
        tl.store(txt_fp8sc_ptr + trow, fp8_sc)
        tl.store(txt_fp8_ptr + trow * stride_tfx + d_offs * stride_tfy, (hm / fp8_sc).to(tl.float8e4nv), mask=mask)


def fused_cross_block_residual_gate_adaln_fp8(
    img_res: torch.Tensor, img_gate2: torch.Tensor, img_mlp: torch.Tensor,
    img_shift_next: torch.Tensor, img_scale_next: torch.Tensor, L_img: int,
    txt_res: torch.Tensor, txt_gate2: torch.Tensor, txt_mlp: torch.Tensor,
    txt_shift_next: torch.Tensor, txt_scale_next: torch.Tensor, L_txt: int,
    eps: float = 1e-6,
):
    """
    Fuses: (residual + gate*mlp) → store → adaln_fp8(next_shift, next_scale)
    for both img and txt in one kernel.

    Returns:
        img_res_out [M_img, D]:  updated h (residual for next next op)
        img_fp8 [M_img, D]:      FP8 quantized and modulated for next block's attn
        img_fp8sc [M_img]:       FP8 scale
        txt_res_out [M_txt, D]
        txt_fp8 [M_txt, D]
        txt_fp8sc [M_txt]
    """
    M_img, D = img_res.shape
    M_txt     = txt_res.shape[0]

    img_res_out = torch.empty_like(img_res)
    img_fp8     = torch.empty(M_img, D, device=img_res.device, dtype=torch.float8_e4m3fn)
    img_fp8sc   = torch.empty(M_img,   device=img_res.device,  dtype=torch.float32)
    txt_res_out = torch.empty_like(txt_res)
    txt_fp8     = torch.empty(M_txt, D, device=txt_res.device, dtype=torch.float8_e4m3fn)
    txt_fp8sc   = torch.empty(M_txt,   device=txt_res.device,  dtype=torch.float32)

    BLOCK_D = _best_block_d(D)
    grid = (M_img + M_txt,)

    _cross_block_residual_gate_adaln_fp8_kernel[grid](
        img_res, img_gate2, img_mlp,
        img_shift_next, img_scale_next,
        img_res_out, img_fp8, img_fp8sc,
        M_img, L_img,
        img_res.stride(0), img_res.stride(1),
        img_gate2.stride(0), img_gate2.stride(1),
        img_mlp.stride(0), img_mlp.stride(1),
        img_shift_next.stride(0), img_shift_next.stride(1),
        img_res_out.stride(0), img_res_out.stride(1),
        img_fp8.stride(0), img_fp8.stride(1),
        txt_res, txt_gate2, txt_mlp,
        txt_shift_next, txt_scale_next,
        txt_res_out, txt_fp8, txt_fp8sc,
        M_txt, L_txt,
        txt_res.stride(0), txt_res.stride(1),
        txt_gate2.stride(0), txt_gate2.stride(1),
        txt_mlp.stride(0), txt_mlp.stride(1),
        txt_shift_next.stride(0), txt_shift_next.stride(1),
        txt_res_out.stride(0), txt_res_out.stride(1),
        txt_fp8.stride(0), txt_fp8.stride(1),
        D=D, eps=eps,
        BLOCK_D=BLOCK_D,
        num_warps=max(1, min(32, BLOCK_D // 2048)),  # BLOCK_D=4096→2 warps; benchmarked best for D=3072
    )
    return img_res_out, img_fp8, img_fp8sc.unsqueeze(1), txt_res_out, txt_fp8, txt_fp8sc.unsqueeze(1)

def _smoke_test():
    B, L_img, L_txt, D = 2, 4096, 218, 3072
    M_img = B * L_img
    M_txt = B * L_txt
    device = 'cuda'
    dtype  = torch.bfloat16

    img  = torch.randn(M_img, D, device=device, dtype=dtype)
    txt  = torch.randn(M_txt, D, device=device, dtype=dtype)
    img_shift = torch.randn(B, D, device=device, dtype=dtype)
    img_scale = torch.randn(B, D, device=device, dtype=dtype)
    txt_shift = torch.randn(B, D, device=device, dtype=dtype)
    txt_scale = torch.randn(B, D, device=device, dtype=dtype)

    # Test 1: dual adaln fp8
    i_fp8, i_sc, t_fp8, t_sc = fused_dual_adaln_fp8(
        img, img_shift, img_scale, L_img,
        txt, txt_shift, txt_scale, L_txt,
    )
    assert i_fp8.shape == (M_img, D) and i_sc.shape == (M_img, 1), f"img shape mismatch: {i_sc.shape}"
    assert t_fp8.shape == (M_txt, D) and t_sc.shape == (M_txt, 1), f"txt shape mismatch: {t_sc.shape}"
    print(f"  dual_adaln_fp8: img_fp8={i_fp8.shape}, txt_fp8={t_fp8.shape} ")

    # Test 2: dual residual_gate_adaln_fp8
    img_gate  = torch.randn(B, D, device=device, dtype=dtype)
    txt_gate  = torch.randn(B, D, device=device, dtype=dtype)
    img_delta = torch.randn(M_img, D, device=device, dtype=dtype)
    txt_delta = torch.randn(M_txt, D, device=device, dtype=dtype)
    img_shift2 = torch.randn(B, D, device=device, dtype=dtype)
    img_scale2 = torch.randn(B, D, device=device, dtype=dtype)
    txt_shift2 = torch.randn(B, D, device=device, dtype=dtype)
    txt_scale2 = torch.randn(B, D, device=device, dtype=dtype)

    ir_out, i2_fp8, i2_sc, tr_out, t2_fp8, t2_sc = fused_dual_residual_gate_adaln_fp8(
        img.clone(), img_gate, img_delta, img_shift2, img_scale2, L_img,
        txt.clone(), txt_gate, txt_delta, txt_shift2, txt_scale2, L_txt,
    )
    assert ir_out.shape == (M_img, D), f"ir_out shape: {ir_out.shape}"
    print(f"  dual_residual_gate_adaln_fp8: img_res_out={ir_out.shape}, fp8={i2_fp8.shape} ")

    # Test 3: dual residual_gate
    ir = img.clone()
    tr = txt.clone()
    ir_out2, tr_out2 = fused_dual_residual_gate(
        ir, img_gate, img_delta, L_img,
        tr, txt_gate, txt_delta, L_txt,
    )
    print(f"  dual_residual_gate: in-place img={ir_out2.shape}, txt={tr_out2.shape} ")

    # Test 4: cross-block kernel
    img_mlp_out = torch.randn(M_img, D, device=device, dtype=dtype)
    txt_mlp_out = torch.randn(M_txt, D, device=device, dtype=dtype)
    img_sn = torch.randn(B, D, device=device, dtype=dtype)
    img_cn = torch.randn(B, D, device=device, dtype=dtype)
    txt_sn = torch.randn(B, D, device=device, dtype=dtype)
    txt_cn = torch.randn(B, D, device=device, dtype=dtype)

    ir_cb, i_cb_fp8, i_cb_sc, tr_cb, t_cb_fp8, t_cb_sc = fused_cross_block_residual_gate_adaln_fp8(
        img.clone(), img_gate, img_mlp_out, img_sn, img_cn, L_img,
        txt.clone(), txt_gate, txt_mlp_out, txt_sn, txt_cn, L_txt,
    )
    assert ir_cb.shape == (M_img, D), f"cross-block img shape: {ir_cb.shape}"
    print(f"  cross_block_residual_gate_adaln_fp8: img_res={ir_cb.shape}, fp8={i_cb_fp8.shape} ")
    print("All smoke tests passed.")


if __name__ == '__main__':
    _smoke_test()