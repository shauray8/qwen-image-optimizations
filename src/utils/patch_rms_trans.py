# modify transformer on the fly as to keep things clean 
import torch
import torch.nn as nn
from transformer import * 
from quack.rmsnorm import rmsnorm, RMSNorm as QuackRMSNorm

class QuackRMSNormWrapper(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.dim = dim
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))
        else:
            self.register_parameter('weight', None)
    
    def forward(self, x):
        # (M, N)
        orig_shape = x.shape
        orig_dtype = x.dtype
        x_2d = x.reshape(-1, self.dim)
        
        # Convert weight to float32 if needed (quack expects float32 weights)
        weight = self.weight if self.weight is not None else None
        if weight is not None and weight.dtype != torch.float32:
            weight = weight.float()
        out = rmsnorm(x_2d, weight=weight, eps=self.eps)
        out = out.reshape(orig_shape)
        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        
        return out

def replace_norms_with_optimized(model, verbose=True):
    replacements = 0
    
    for name, module in list(model.named_children()):
        is_layer_norm = isinstance(module, nn.LayerNorm)
        is_rms_norm = type(module).__name__ == 'RMSNorm' or isinstance(module, RMSNorm)
        
        if is_layer_norm or is_rms_norm:
            if hasattr(module, 'normalized_shape'):
                if isinstance(module.normalized_shape, (list, tuple)):
                    dim = module.normalized_shape[0]
                else:
                    dim = module.normalized_shape
            elif hasattr(module, 'weight') and module.weight is not None:
                dim = module.weight.shape[0]
            else:
                continue
            
            eps = getattr(module, 'eps', 1e-6)
            elementwise_affine = hasattr(module, 'weight') and module.weight is not None
            new_module = QuackRMSNormWrapper(
                dim, 
                eps=eps, 
                elementwise_affine=elementwise_affine,
                device=module.weight.device if hasattr(module, 'weight') and module.weight is not None else None,
                dtype=torch.float32  # quack works with float32 weights
            )
            
            if elementwise_affine and hasattr(module, 'weight') and module.weight is not None:
                with torch.no_grad():
                    new_module.weight.copy_(module.weight.data.float())
            
            setattr(model, name, new_module)
            replacements += 1
            if verbose:
                print(f"Replaced {name} ({module.__class__.__name__}) with QuackRMSNorm")
        else:
            child_replacements = replace_norms_with_optimized(module, verbose=False)
            replacements += child_replacements
    
    if verbose:
        print(f"\nTotal replacements: {replacements}")
    
    return replacements
