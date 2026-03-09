"""
V15: Fused attention-output split + FP8-quantize kernel.
"""
import torch
import torch.nn as nn
from typing import Optional

from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention

from models.transformer import flash_attn_wrapper
from models.transformer_rowwise_fp8 import OptimizedRowwiseFP8Linear
from kernels.fused_fp8_quant_v5 import fused_rowwise_fp8_quant
from kernels.fused_kernels import fused_gelu_fp8_quant
from kernels.fused_kernels_v2 import (
    fused_dual_adaln_fp8,
    fused_dual_residual_gate_adaln_fp8,
    fused_cross_block_residual_gate_adaln_fp8,
    fused_dual_residual_gate,
)
from kernels.fused_kernels_v3 import fused_qkv_rmsnorm_rope_pack, fused_attn_split_fp8_quant
from models.optimized_block_v14 import (
    FusedQKVFP8LinearRaw,
    _fp8_gemm,
    _fuse_qkv_raw,
    _patch_rope_decomposition,
)

@maybe_allow_in_graph
class FusedQwenImageTransformerBlockV15(nn.Module):
    def __init__(self, dim, num_attention_heads, attention_head_dim, zero_cond_t=False):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.zero_cond_t = zero_cond_t
        self.eps = 1e-6
        self.attn = None
        self.img_mlp = None
        self.txt_mlp = None
        self.img_qkv = None
        self.txt_qkv = None
        # Cached flags for to_out FP8 support
        self._img_out_is_fp8 = None
        self._txt_out_is_fp8 = None

    def forward(
        self,
        img_fp8_in, img_fp8sc_in,
        txt_fp8_in, txt_fp8sc_in,
        hidden_states_2d, enc_states_2d,
        img_gate1, img_shift2, img_scale2, img_gate2,
        txt_gate1, txt_shift2, txt_scale2, txt_gate2,
        img_shift1_next, img_scale1_next,
        txt_shift1_next, txt_scale1_next,
        img_cos, img_sin, txt_cos, txt_sin,
        B, img_seq, txt_seq, D,
        joint_q_buf, joint_k_buf, joint_v_buf,
    ):
        heads    = self.num_attention_heads
        head_dim = self.attention_head_dim
        attn     = self.attn
        joint_seq = txt_seq + img_seq

        # QKV GEMMs → raw [M, 3*H*D] contiguous
        img_qkv_raw = self.img_qkv.forward_with_fp8_input_raw(img_fp8_in, img_fp8sc_in)
        txt_qkv_raw = self.txt_qkv.forward_with_fp8_input_raw(txt_fp8_in, txt_fp8sc_in)

        # Fused: split QKV + RMSNorm(Q,K) + RoPE(Q,K) + write to joint buffers
        fused_qkv_rmsnorm_rope_pack(
            img_qkv_raw, joint_q_buf, joint_k_buf, joint_v_buf,
            attn.norm_q.weight, attn.norm_k.weight,
            img_cos, img_sin,
            B, img_seq, heads, head_dim,
            joint_seq, txt_seq,
            eps=self.eps,
        )
        fused_qkv_rmsnorm_rope_pack(
            txt_qkv_raw, joint_q_buf, joint_k_buf, joint_v_buf,
            attn.norm_added_q.weight, attn.norm_added_k.weight,
            txt_cos, txt_sin,
            B, txt_seq, heads, head_dim,
            joint_seq, 0,
            eps=self.eps,
        )

        # Flash attention
        joint_out = flash_attn_wrapper(
            joint_q_buf, joint_k_buf, joint_v_buf,
            softmax_scale=1.0 / 11.313708498984761,
        )[0]
        # Flatten heads into D: [B, joint_seq, H, head_dim] → [B, joint_seq, D]
        joint_out_flat = joint_out.reshape(B, joint_seq, heads * head_dim)

        # Eliminates: 2 × implicit .contiguous() + 2 × fused_fp8_quant_kernel
        img_fp8_attn, img_fp8sc_attn, txt_fp8_attn, txt_fp8sc_attn = \
            fused_attn_split_fp8_quant(joint_out_flat, txt_seq, img_seq, B, joint_seq, D)

        # Attention output projections (using pre-quantized FP8 from above)
        img_to_out = attn.to_out[0]
        txt_to_out = attn.to_add_out
        img_attn = _fp8_gemm(img_fp8_attn, img_fp8sc_attn, img_to_out).view(B, img_seq, D)
        txt_attn = _fp8_gemm(txt_fp8_attn, txt_fp8sc_attn, txt_to_out).view(B, txt_seq, D)
        # Note: to_out[1] (if present) is Dropout which is identity in eval mode
        img_attn_2d = img_attn.reshape(B * img_seq, D)
        txt_attn_2d = txt_attn.reshape(B * txt_seq, D)

        (hidden_states_2d, img_mlp_fp8, img_mlp_sc,
         enc_states_2d,    txt_mlp_fp8, txt_mlp_sc) = fused_dual_residual_gate_adaln_fp8(
            hidden_states_2d, img_gate1, img_attn_2d, img_shift2, img_scale2, img_seq,
            enc_states_2d,    txt_gate1, txt_attn_2d, txt_shift2, txt_scale2, txt_seq,
            eps=self.eps,
        )

        img_mlp_out = self._fused_mlp(img_mlp_fp8, img_mlp_sc, self.img_mlp, B, img_seq, D)
        txt_mlp_out = self._fused_mlp(txt_mlp_fp8, txt_mlp_sc, self.txt_mlp, B, txt_seq, D)

        img_mlp_2d = img_mlp_out.reshape(B * img_seq, D)
        txt_mlp_2d = txt_mlp_out.reshape(B * txt_seq, D)

        if img_shift1_next is not None:
            (hidden_states_2d, img_fp8_next, img_fp8sc_next,
             enc_states_2d,    txt_fp8_next, txt_fp8sc_next) = fused_cross_block_residual_gate_adaln_fp8(
                hidden_states_2d, img_gate2, img_mlp_2d, img_shift1_next, img_scale1_next, img_seq,
                enc_states_2d,    txt_gate2, txt_mlp_2d, txt_shift1_next, txt_scale1_next, txt_seq,
                eps=self.eps,
            )
            return (hidden_states_2d, enc_states_2d,
                    img_fp8_next, img_fp8sc_next,
                    txt_fp8_next, txt_fp8sc_next)
        else:
            hidden_states_2d, enc_states_2d = fused_dual_residual_gate(
                hidden_states_2d, img_gate2, img_mlp_2d, img_seq,
                enc_states_2d,    txt_gate2, txt_mlp_2d, txt_seq,
            )
            return (hidden_states_2d, enc_states_2d, None, None, None, None)

    def _fused_mlp(self, x_fp8, x_scale, mlp, B, seq_len, D):
        gelu_module = mlp.net[0]
        fc2 = mlp.net[2]
        if not isinstance(gelu_module.proj, OptimizedRowwiseFP8Linear):
            raise RuntimeError("Expected FP8 linear for MLP FC1")
        fc1_out = _fp8_gemm(x_fp8, x_scale, gelu_module.proj)
        fc1_fp8, fc1_scale = fused_gelu_fp8_quant(fc1_out)
        if isinstance(fc2, OptimizedRowwiseFP8Linear):
            return _fp8_gemm(fc1_fp8, fc1_scale, fc2).view(B, seq_len, D)
        else:
            fc1_bf16 = fc1_fp8.to(torch.bfloat16) * fc1_scale
            return fc2(fc1_bf16.view(B, seq_len, -1))

def _v15_forward_corrected(
    self,
    hidden_states,
    encoder_hidden_states=None,
    encoder_hidden_states_mask=None,
    timestep=None,
    img_shapes=None,
    txt_seq_lens=None,
    guidance=None,
    attention_kwargs=None,
    controlnet_block_samples=None,
    additional_t_cond=None,
    return_dict=True,
):
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from models.transformer import compute_text_seq_len_from_mask
    from kernels.fused_kernels_v2 import fused_dual_adaln_fp8

    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        attention_kwargs.pop("scale", None)

    hidden_states = self.img_in(hidden_states)
    timestep = timestep.to(hidden_states.dtype)

    if self.zero_cond_t:
        timestep = torch.cat([timestep, timestep * 0], dim=0)

    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
        encoder_hidden_states, encoder_hidden_states_mask
    )

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states, additional_t_cond)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
    )

    image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)
    img_cos, img_sin, txt_cos, txt_sin = image_rotary_emb

    B, img_seq, D = hidden_states.shape
    txt_seq = encoder_hidden_states.shape[1]
    eps = 1e-6
    all_mods = []
    for block in self.transformer_blocks:
        temb_txt = temb.chunk(2, dim=0)[0] if block.zero_cond_t else temb
        img_mp = block._img_mod(temb)
        txt_mp = block._txt_mod(temb_txt)
        img_mod1, img_mod2 = img_mp.chunk(2, dim=-1)
        txt_mod1, txt_mod2 = txt_mp.chunk(2, dim=-1)
        img_s1, img_c1, img_g1 = img_mod1.chunk(3, dim=-1)
        img_s2, img_c2, img_g2 = img_mod2.chunk(3, dim=-1)
        txt_s1, txt_c1, txt_g1 = txt_mod1.chunk(3, dim=-1)
        txt_s2, txt_c2, txt_g2 = txt_mod2.chunk(3, dim=-1)
        all_mods.append((img_s1, img_c1, img_g1, img_s2, img_c2, img_g2,
                         txt_s1, txt_c1, txt_g1, txt_s2, txt_c2, txt_g2))

    img_2d = hidden_states.reshape(B * img_seq, D)
    txt_2d = encoder_hidden_states.reshape(B * txt_seq, D)

    img_fp8, img_fp8sc, txt_fp8, txt_fp8sc = fused_dual_adaln_fp8(
        img_2d, all_mods[0][0], all_mods[0][1], img_seq,
        txt_2d, all_mods[0][6], all_mods[0][7], txt_seq,
        eps=eps,
    )

    heads    = self.transformer_blocks[0].num_attention_heads
    head_dim = self.transformer_blocks[0].attention_head_dim
    joint_seq = txt_seq + img_seq

    joint_q_buf = torch.empty(B, joint_seq, heads, head_dim,
                              device=hidden_states.device, dtype=hidden_states.dtype)
    joint_k_buf = torch.empty_like(joint_q_buf)
    joint_v_buf = torch.empty_like(joint_q_buf)

    for i, block in enumerate(self.transformer_blocks):
        mods = all_mods[i]
        (img_s1, img_c1, img_g1, img_s2, img_c2, img_g2,
         txt_s1, txt_c1, txt_g1, txt_s2, txt_c2, txt_g2) = mods

        is_last = (i == len(self.transformer_blocks) - 1)
        if not is_last:
            nm = all_mods[i + 1]
            img_s1_next, img_c1_next = nm[0], nm[1]
            txt_s1_next, txt_c1_next = nm[6], nm[7]
        else:
            img_s1_next = img_c1_next = txt_s1_next = txt_c1_next = None

        result = block(
            img_fp8, img_fp8sc, txt_fp8, txt_fp8sc,
            img_2d, txt_2d,
            img_g1, img_s2, img_c2, img_g2,
            txt_g1, txt_s2, txt_c2, txt_g2,
            img_s1_next, img_c1_next, txt_s1_next, txt_c1_next,
            img_cos, img_sin, txt_cos, txt_sin,
            B, img_seq, txt_seq, D,
            joint_q_buf, joint_k_buf, joint_v_buf,
        )

        img_2d, txt_2d = result[0], result[1]
        img_fp8, img_fp8sc = result[2], result[3]
        txt_fp8, txt_fp8sc = result[4], result[5]

        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(__import__('math').ceil(interval_control))
            hidden_s_view = img_2d.view(B, img_seq, D)
            hidden_s_view = hidden_s_view + controlnet_block_samples[i // interval_control]
            img_2d = hidden_s_view.reshape(B * img_seq, D)

    hidden_states = img_2d.view(B, img_seq, D)

    if self.zero_cond_t:
        temb = temb.chunk(2, dim=0)[0]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

def apply_v15_optimizations(model, verbose=True):
    import types
    from models.transformer import QwenImageTransformerBlock

    if verbose:
        print("V15: V14 + Fused attention-split + FP8-quantize kernel")
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, QwenImageTransformerBlock):
            new_block = FusedQwenImageTransformerBlockV15(
                dim=module.dim,
                num_attention_heads=module.num_attention_heads,
                attention_head_dim=module.attention_head_dim,
                zero_cond_t=module.zero_cond_t,
            )

            new_block.attn     = module.attn
            new_block.img_mlp  = module.img_mlp
            new_block.txt_mlp  = module.txt_mlp
            new_block._img_mod = module.img_mod
            new_block._txt_mod = module.txt_mod

            img_qkv, txt_qkv = _fuse_qkv_raw(module.attn)
            if img_qkv is None or txt_qkv is None:
                raise RuntimeError(
                    f"Block {name}: Q/K/V not OptimizedRowwiseFP8Linear. "
                    "Run apply_optimized_rowwise_fp8 first."
                )
            new_block.img_qkv = img_qkv
            new_block.txt_qkv = txt_qkv

            # Verify to_out projections are FP8 (needed for _fp8_gemm bypass)
            if not isinstance(module.attn.to_out[0], OptimizedRowwiseFP8Linear):
                raise RuntimeError(
                    f"Block {name}: attn.to_out[0] is not OptimizedRowwiseFP8Linear. "
                    "V15 requires FP8 attention output projections."
                )
            if not isinstance(module.attn.to_add_out, OptimizedRowwiseFP8Linear):
                raise RuntimeError(
                    f"Block {name}: attn.to_add_out is not OptimizedRowwiseFP8Linear."
                )

            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model
                for p in parent_name.split('.'):
                    parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                if attr_name.isdigit():
                    parent[int(attr_name)] = new_block
                else:
                    setattr(parent, attr_name, new_block)
            replaced += 1

    if verbose:
        print(f"  Replaced {replaced} blocks with V15 blocks")

    _patch_rope_decomposition(model)
    model.forward = types.MethodType(_v15_forward_corrected, model)
    return replaced