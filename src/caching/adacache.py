"""
AdaCache: Adaptive Caching for Diffusion Transformers
arXiv:2411.02397
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field

@dataclass
class AdaCacheState:
    cached_residuals: Dict[str, torch.Tensor] = field(default_factory=dict)
    cache_countdown: Dict[str, int] = field(default_factory=dict)
    prev_features: Optional[torch.Tensor] = None

    total_computes: int = 0
    total_cached: int = 0
    
    def reset(self):
        self.cached_residuals.clear()
        self.cache_countdown.clear()
        self.prev_features = None
        self.total_computes = 0
        self.total_cached = 0

_adacache_state: Optional[AdaCacheState] = None

def get_adacache_state() -> AdaCacheState:
    global _adacache_state
    if _adacache_state is None:
        _adacache_state = AdaCacheState()
    return _adacache_state

def reset_adacache():
    global _adacache_state
    if _adacache_state is not None:
        _adacache_state.reset()

# tuned for Qwen
DEFAULT_CODEBOOK_28 = {
    0.05: 5,
    0.10: 4,
    0.15: 3,
    0.20: 2,
    0.30: 1,
    1.00: 1,
}

DEFAULT_CODEBOOK_50 = {
    0.03: 8,
    0.06: 6,
    0.09: 4,
    0.12: 3,
    0.15: 2,
    1.00: 1,
}

def compute_cache_duration(distance: float, codebook: Dict[float, int]) -> int:
    for threshold, duration in sorted(codebook.items()):
        if distance < threshold:
            return duration
    return 1  # recompute next step

@torch.compiler.disable
def compute_distance(current: torch.Tensor, previous: torch.Tensor) -> float:
    if previous is None:
        return 1.0
    diff = (current - previous).abs().mean()
    magnitude = current.abs().mean() + 1e-8
    return (diff / magnitude).item()

class AdaCacheTransformerWrapper(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        codebook: Optional[Dict[float, int]] = None,
        enabled: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        self.codebook = codebook or DEFAULT_CODEBOOK_28
        self.enabled = enabled
        self.verbose = verbose
        self._step_idx = 0
        self._cache_until_step = -1
        
    def set_codebook(self, codebook: Dict[float, int]):
        self.codebook = codebook
        
    def reset(self):
        self._step_idx = 0
        self._cache_until_step = -1
        reset_adacache()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        block_attention_kwargs: Dict,
        modulate_index: Optional[torch.Tensor] = None,
        controlnet_block_samples=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not self.enabled:
            return self._forward_full(
                hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                temb, image_rotary_emb, block_attention_kwargs, modulate_index,
                controlnet_block_samples
            )
        
        state = get_adacache_state()
        should_compute = self._step_idx >= self._cache_until_step
        
        if should_compute:
            enc_out, hidden_out = self._forward_full(
                hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                temb, image_rotary_emb, block_attention_kwargs, modulate_index,
                controlnet_block_samples
            )
            
            hidden_residual = hidden_out - hidden_states
            encoder_residual = enc_out - encoder_hidden_states
            
            distance = compute_distance(hidden_residual, state.prev_features)
            cache_steps = compute_cache_duration(distance, self.codebook)
            self._cache_until_step = self._step_idx + cache_steps
            
            state.prev_features = hidden_residual.clone()
            state.cached_residuals['hidden'] = hidden_residual
            state.cached_residuals['encoder'] = encoder_residual
            state.total_computes += 1
            
            if self.verbose:
                print(f"[AdaCache] Step {self._step_idx}: COMPUTE, dist={distance:.4f}, cache_for={cache_steps}")
            
        else:
            hidden_residual = state.cached_residuals.get('hidden')
            encoder_residual = state.cached_residuals.get('encoder')
            
            if hidden_residual is not None and encoder_residual is not None:
                hidden_out = hidden_states + hidden_residual
                enc_out = encoder_hidden_states + encoder_residual
                state.total_cached += 1
                
                if self.verbose:
                    print(f"[AdaCache] Step {self._step_idx}: CACHED (until step {self._cache_until_step})")
            else:
                enc_out, hidden_out = self._forward_full(
                    hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                    temb, image_rotary_emb, block_attention_kwargs, modulate_index,
                    controlnet_block_samples
                )
                state.total_computes += 1
        
        self._step_idx += 1
        return enc_out, hidden_out

    def _forward_full(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        block_attention_kwargs: Dict,
        modulate_index: Optional[torch.Tensor] = None,
        controlnet_block_samples=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for idx, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=modulate_index,
            )
            
            if controlnet_block_samples is not None:
                interval = len(self.transformer_blocks) / len(controlnet_block_samples)
                hidden_states = hidden_states + controlnet_block_samples[int(idx // interval)]
        
        return encoder_hidden_states, hidden_states

def apply_adacache(
    transformer: nn.Module,
    codebook: Optional[Dict[float, int]] = None,
    enabled: bool = True,
    verbose: bool = False,
) -> nn.Module:
    wrapper = AdaCacheTransformerWrapper(
        transformer=transformer,
        codebook=codebook,
        enabled=enabled,
        verbose=verbose,
    )
    transformer._adacache_wrapper = wrapper
    transformer._adacache_enabled = enabled
    
    return transformer

def get_adacache_stats() -> Dict:
    state = get_adacache_state()
    total = state.total_computes + state.total_cached
    hit_rate = state.total_cached / total if total > 0 else 0
    return {
        'computes': state.total_computes,
        'cached': state.total_cached,
        'total': total,
        'hit_rate': hit_rate,
    }
