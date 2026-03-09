import torch
import torch.nn as nn
from models.transformer_rowwise_fp8 import OptimizedRowwiseFP8Linear
from kernels.fused_fp8_quant_v5 import fused_rowwise_fp8_quant

FP8_MAX = 448.0

class FusedQKVFP8Linear(nn.Module):
    def __init__(self, q_linear, k_linear, v_linear):
        super().__init__()
        assert isinstance(q_linear, OptimizedRowwiseFP8Linear)
        assert isinstance(k_linear, OptimizedRowwiseFP8Linear)
        assert isinstance(v_linear, OptimizedRowwiseFP8Linear)

        self.in_features = q_linear.in_features
        self.out_features_q = q_linear.out_features
        self.out_features_k = k_linear.out_features
        self.out_features_v = v_linear.out_features
        self.total_out = self.out_features_q + self.out_features_k + self.out_features_v

        # Concatenate FP8 weights: W_q.t() is [K, N_q], W_k.t() is [K, N_k], etc.
        # Concat along N dimension: [K, N_q+N_k+N_v]
        # After cat, must restore col_major layout (stride(0)==1) for _scaled_mm
        fused_w = torch.cat([q_linear.weight_fp8_t, k_linear.weight_fp8_t, v_linear.weight_fp8_t], dim=1)
        self.register_buffer('weight_fp8_t', fused_w.t().contiguous().t())

        # Concatenate weight scales: each is [1, N_i]
        self.register_buffer('w_scale',
            torch.cat([q_linear.w_scale, k_linear.w_scale, v_linear.w_scale], dim=1))

        # Concatenate biases if present
        has_bias = q_linear.bias is not None
        if has_bias:
            self.register_buffer('bias',
                torch.cat([q_linear.bias, k_linear.bias, v_linear.bias], dim=0))
        else:
            self.bias = None

    def forward(self, x):
        x_fp8, a_scale = fused_rowwise_fp8_quant(x)
        # bias= fuses the add into the CUTLASS epilogue, saving a separate kernel launch
        output = torch._scaled_mm(
            x_fp8,
            self.weight_fp8_t,
            scale_a=a_scale,
            scale_b=self.w_scale,
            out_dtype=torch.bfloat16,
            bias=self.bias,
        )

        q = output[:, :self.out_features_q]
        k = output[:, self.out_features_q:self.out_features_q + self.out_features_k]
        v = output[:, self.out_features_q + self.out_features_k:]

        return q, k, v

    def forward_with_fp8_input(self, x_fp8, a_scale):
        """Forward pass with pre-quantized FP8 input (for fused AdaLN+FP8)."""
        # bias= fuses the add into the CUTLASS epilogue, saving a separate kernel launch
        output = torch._scaled_mm(
            x_fp8,
            self.weight_fp8_t,
            scale_a=a_scale,
            scale_b=self.w_scale,
            out_dtype=torch.bfloat16,
            bias=self.bias,
        )

        q = output[:, :self.out_features_q]
        k = output[:, self.out_features_q:self.out_features_q + self.out_features_k]
        v = output[:, self.out_features_q + self.out_features_k:]

        return q, k, v

def fuse_qkv_in_attention(attn_module):
    # Image stream
    if (isinstance(attn_module.to_q, OptimizedRowwiseFP8Linear) and
        isinstance(attn_module.to_k, OptimizedRowwiseFP8Linear) and
        isinstance(attn_module.to_v, OptimizedRowwiseFP8Linear)):
        img_qkv = FusedQKVFP8Linear(attn_module.to_q, attn_module.to_k, attn_module.to_v)
    else:
        img_qkv = None

    # Text stream
    if (hasattr(attn_module, 'add_q_proj') and
        isinstance(attn_module.add_q_proj, OptimizedRowwiseFP8Linear) and
        isinstance(attn_module.add_k_proj, OptimizedRowwiseFP8Linear) and
        isinstance(attn_module.add_v_proj, OptimizedRowwiseFP8Linear)):
        txt_qkv = FusedQKVFP8Linear(attn_module.add_q_proj, attn_module.add_k_proj, attn_module.add_v_proj)
    else:
        txt_qkv = None
    return img_qkv, txt_qkv