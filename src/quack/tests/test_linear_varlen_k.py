# Copyright (C) 2025, Tri Dao.
import math
import pytest
import torch

from quack.gemm_interface import (
    gemm,
    gemm_ref,
    gemm_add,
    gemm_add_ref,
    gemm_add_inplace,
)


def generate_A_with_gather(m, total_k, device, dtype, gather_A=False):
    """Generate A matrix and optionally A_idx for gather_A case with varlen_k.

    Args:
        m: Number of rows
        total_k: Number of columns needed
        device: Device to create tensors on
        dtype: Data type of tensors
        gather_A: Whether to create gather indices

    Returns:
        A: Matrix of shape (m, larger_k) if gather_A else (m, total_k)
        A_idx: Index tensor of shape (total_k,) if gather_A else None
    """
    if gather_A:
        # Create random indices for gathering from a larger A matrix
        larger_k = total_k * 2  # Make A larger than needed
        A = torch.randn((m, larger_k), device=device, dtype=dtype)
        # Make A m-major
        A = A.T.contiguous().T
        # Create random indices to gather from A
        A_idx = torch.randperm(larger_k, device=device, dtype=torch.int32)[:total_k]
    else:
        A = torch.randn((m, total_k), device=device, dtype=dtype)
        # Make A m-major
        A = A.T.contiguous().T
        A_idx = None
    return A, A_idx


@pytest.mark.parametrize("permute_batch", [False, True])
@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
# @pytest.mark.parametrize("alpha_is_tensor", [False])
@pytest.mark.parametrize("alpha", [1.0, 0.93])
# @pytest.mark.parametrize("alpha", [1.0])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504, 4096])
@pytest.mark.parametrize("m", [2048, 1064, 8192])
# @pytest.mark.parametrize("n", [1024])
# @pytest.mark.parametrize("m", [2048])
@pytest.mark.parametrize("num_groups", [2, 4])
# @pytest.mark.parametrize("num_groups", [2])
def test_gemm_varlen_k(
    num_groups,
    m,
    n,
    input_dtype,
    alpha,
    alpha_is_tensor,
    dynamic_scheduler,
    gather_A,
    permute_batch,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_k = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    if permute_batch:
        batch_idx_permute = torch.randperm(num_groups, device=device).to(torch.int32)
    else:
        batch_idx_permute = None
    out = gemm(
        A,
        B,
        alpha=alpha,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        batch_idx_permute=batch_idx_permute,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert out.shape == (num_groups, m, n)
    out_ref = gemm_ref(
        A.float(),
        B.float(),
        alpha=alpha_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_ref(A, B, alpha=alpha_val, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("m", [2048])
def test_gemm_varlen_k_with_zero_lengths(
    m,
    n,
    input_dtype,
    gather_A,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.tensor([150, 64, 0, 200, 0], device="cpu", dtype=torch.int32)
    num_groups = seq_lens.shape[0]
    total_k = seq_lens.sum().item()
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    out = gemm(
        A,
        B,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        dynamic_scheduler=False,
        tuned=False,
    )
    assert out.shape == (num_groups, m, n)
    out_ref = gemm_ref(
        A.float(),
        B.float(),
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_ref(A, B, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx)
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("C_major", ["m", "n"])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.0, 1.17])
@pytest.mark.parametrize("alpha", [1.0, 0.93])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("m", [2048, 1024])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_add_varlen_k(
    num_groups,
    m,
    n,
    input_dtype,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
    C_major,
    dynamic_scheduler,
    gather_A,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_k = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    # Make A m-major
    A = A.T.contiguous().T
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    C = torch.randn((num_groups, m, n), device=device, dtype=input_dtype)
    if C_major == "m":
        C = C.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    out = gemm_add(
        A,
        B,
        C,
        alpha=alpha,
        beta=beta,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    assert out.shape == (num_groups, m, n)
    out_ref = gemm_add_ref(
        A.float(),
        B.float(),
        C.float(),
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_add_ref(
        A, B, C, alpha=alpha_val, beta=beta_val, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4


@pytest.mark.parametrize("gather_A", [False, True])
# @pytest.mark.parametrize("gather_A", [False])
@pytest.mark.parametrize("dynamic_scheduler", [False, True])
# @pytest.mark.parametrize("dynamic_scheduler", [False])
@pytest.mark.parametrize("alpha_is_tensor", [False, True])
@pytest.mark.parametrize("beta_is_tensor", [False, True])
@pytest.mark.parametrize("beta", [0.0, 1.17])
@pytest.mark.parametrize("alpha", [1.0, 0.93])
@pytest.mark.parametrize("input_dtype", [torch.bfloat16])
@pytest.mark.parametrize("n", [1024, 1504])
@pytest.mark.parametrize("m", [2048, 1024])
@pytest.mark.parametrize("num_groups", [2, 4])
def test_gemm_add_inplace_varlen_k(
    num_groups,
    m,
    n,
    input_dtype,
    alpha,
    beta,
    alpha_is_tensor,
    beta_is_tensor,
    dynamic_scheduler,
    gather_A,
):
    device = "cuda"
    torch.random.manual_seed(42)
    seq_lens = torch.randint(50, 300, (num_groups,), device="cpu")
    total_k = seq_lens.sum().item()
    # Create cumulative sequence lengths (num_groups + 1)
    cu_seqlens_k = torch.cat(
        [torch.zeros(1, dtype=torch.int32), seq_lens.cumsum(0).to(torch.int32)]
    )
    cu_seqlens_k = cu_seqlens_k.to(device)
    A, A_idx = generate_A_with_gather(m, total_k, device, input_dtype, gather_A)
    # Make A m-major
    A = A.T.contiguous().T
    avg_k = total_k / num_groups
    B = torch.randn((total_k, n), device=device, dtype=input_dtype) / math.sqrt(avg_k)
    out = torch.randn((num_groups, m, n), device=device, dtype=input_dtype)
    if alpha_is_tensor:
        alpha = torch.tensor(alpha, device=device, dtype=torch.float32)
    if beta_is_tensor:
        beta = torch.tensor(beta, device=device, dtype=torch.float32)
    # Save original out for reference computation
    out_og = out.clone()
    gemm_add_inplace(
        A,
        B,
        out,
        alpha=alpha,
        beta=beta,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
        dynamic_scheduler=dynamic_scheduler,
        tuned=False,
    )
    alpha_val = alpha.item() if torch.is_tensor(alpha) else alpha
    beta_val = beta.item() if torch.is_tensor(beta) else beta
    out_ref = gemm_add_ref(
        A.float(),
        B.float(),
        out_og.float(),
        out=None,  # Don't use in-place for reference
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    out_pt = gemm_add_ref(
        A,
        B,
        out_og,
        out=None,
        alpha=alpha_val,
        beta=beta_val,
        cu_seqlens_k=cu_seqlens_k,
        A_idx=A_idx,
    )
    assert out.shape == (num_groups, m, n), (
        f"Output shape mismatch: {out.shape} vs expected ({num_groups}, {m}, {n})"
    )
    assert (out - out_ref).abs().max() < 2 * (out_pt - out_ref).abs().max() + 1e-4
