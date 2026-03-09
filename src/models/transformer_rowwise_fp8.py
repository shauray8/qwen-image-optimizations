import torch
import torch.nn as nn
from kernels.fused_fp8_quant_v5 import fused_rowwise_fp8_quant

FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max

class OptimizedRowwiseFP8Linear(nn.Module):
    def __init__(self, original_linear: nn.Linear):
        super().__init__()

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        weight = original_linear.weight.data  # [N, K]

        # Per-row weight quantization
        row_amax = weight.abs().amax(dim=1)  # [N]
        w_scale = (row_amax / FP8_MAX).clamp(min=1e-12)  # [N]

        # Quantize weight to FP8
        weight_fp8 = (weight / w_scale.unsqueeze(1)).to(torch.float8_e4m3fn)  # [N, K]

        # Store as transpose for _scaled_mm (requires stride(0)==1)
        # weight_fp8 is [N, K], .t() gives [K, N] with stride(1, N) so stride(0)==1
        self.register_buffer('weight_fp8_t', weight_fp8.t())
        self.register_buffer('w_scale', w_scale.unsqueeze(0).float())  # [1, N]

        if original_linear.bias is not None:
            self.register_buffer('bias', original_linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_2d = x.reshape(-1, self.in_features).contiguous()  # [M, K]

        # Fused scale computation (Triton) + quantization (PyTorch)
        # This exactly matches the reference implementation
        x_fp8, a_scale = fused_rowwise_fp8_quant(x_2d)

        # FP8 GEMM via cuBLAS; bias= fuses the add into CUTLASS epilogue
        output = torch._scaled_mm(
            x_fp8,
            self.weight_fp8_t,
            scale_a=a_scale,
            scale_b=self.w_scale,
            out_dtype=torch.bfloat16,
            bias=self.bias,
        )

        return output.view(*original_shape[:-1], self.out_features)

def _replace_norms_with_quack(model, verbose=False):
    try:
        import quack
        import torch.nn as nn

        class QuackNormWrapper(nn.Module):
            def __init__(self, norm):
                super().__init__()
                self.weight = norm.weight if hasattr(norm, 'weight') else None
                self.bias   = norm.bias   if hasattr(norm, 'bias')   else None
                self.eps    = getattr(norm, 'eps', 1e-6)
                self.norm_type = type(norm).__name__

            def forward(self, x):
                return quack.rmsnorm(x, self.weight, eps=self.eps)

        replaced = 0
        for name, module in list(model.named_modules()):
            cls = type(module).__name__
            if cls in ('RMSNorm', 'LlamaRMSNorm') or \
               (cls == 'LayerNorm' and not getattr(module, 'elementwise_affine', True)):
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = model
                    for p in parts[0].split('.'):
                        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
                    setattr(parent, parts[1], QuackNormWrapper(module))
                    replaced += 1
        if verbose:
            print(f"  Replaced {replaced} norms with QuackNorm")
        return replaced
    except ImportError:
        if verbose:
            print("  quack not available, skipping norm replacement")
        return 0

def apply_optimized_rowwise_fp8(model, verbose=True):
    replace_norms_with_optimized = _replace_norms_with_quack

    if verbose:
        print("APPLYING OPTIMIZED ROWWISE FP8 (Fused Triton Quant)")
    num_norms = replace_norms_with_optimized(model, verbose=verbose)
    converted = 0
    skipped = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, OptimizedRowwiseFP8Linear):
            # Skip if dimensions not divisible by 128 (FP8 alignment)
            if module.in_features % 128 != 0 or module.out_features % 128 != 0:
                skipped += 1
                continue
            
            # Skip input/output projection layers that may have dimension mismatches
            skip_patterns = ['txt_in', 'img_in', 'final_layer', 'time_', 'guidance_']
            should_skip = any(p in name for p in skip_patterns)
            if should_skip:
                skipped += 1
                continue

            try:
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model
                    for p in parent_name.split('.'):
                        if p.isdigit():
                            parent = parent[int(p)]
                        else:
                            parent = getattr(parent, p)
                    setattr(parent, attr_name, OptimizedRowwiseFP8Linear(module))
                    converted += 1
                elif len(parts) == 1:
                    setattr(model, parts[0], OptimizedRowwiseFP8Linear(module))
                    converted += 1
            except Exception as e:
                if verbose:
                    print(f"  Skip {name}: {e}")
                skipped += 1

    if verbose:
        print(f"\n  Converted: {converted}, Skipped: {skipped}")
    return num_norms + converted