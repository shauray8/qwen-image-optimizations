"""
H2-Cache: Hierarchical Dual-Stage Cache for Qwen-Image Transformer
- 60 double-stream blocks
- Each block processes both image and text streams jointly
Caching strategy:
- Stage 1 (Structure): Early blocks (0-29) - defines global layout
- Stage 2 (Detail): Later blocks (30-59) - refines high-frequency details
(Does not work very well as seen on tests)

Uses Pooled Feature Summarization (PFS) for efficient similarity checking.
"""
import contextlib
import dataclasses
from collections import defaultdict
from typing import DefaultDict, Dict, Optional, Tuple, Callable
import functools
import torch
from torch import nn
import torch.nn.functional as F


NUM_TRANSFORMER_BLOCKS = 60
STAGE1_END = 30  # First 30 blocks for structure
STAGE2_START = 30  # Last 30 for detail

# Pool sizes for PFS
STAGE1_POOL_SIZE = 2048
STAGE2_POOL_SIZE = 384

@dataclasses.dataclass
class QwenCacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))
    stage1_hits: int = 0
    stage2_hits: int = 0
    total_steps: int = 0

    def get_buffer(self, name: str) -> Optional[torch.Tensor]:
        return self.buffers.get(name)

    def set_buffer(self, name: str, buffer: torch.Tensor):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()
        self.stage1_hits = 0
        self.stage2_hits = 0
        self.total_steps = 0

_current_cache_context: Optional[QwenCacheContext] = None

def create_cache_context() -> QwenCacheContext:
    return QwenCacheContext()

def get_current_cache_context() -> Optional[QwenCacheContext]:
    return _current_cache_context

@contextlib.contextmanager
def cache_context(ctx: QwenCacheContext):
    global _current_cache_context
    old_ctx = _current_cache_context
    _current_cache_context = ctx
    try:
        yield
    finally:
        _current_cache_context = old_ctx

@torch.compiler.disable
def get_buffer(name: str) -> Optional[torch.Tensor]:
    ctx = get_current_cache_context()
    if ctx is None:
        return None
    return ctx.get_buffer(name)

@torch.compiler.disable
def set_buffer(name: str, buffer: torch.Tensor):
    ctx = get_current_cache_context()
    if ctx is not None:
        ctx.set_buffer(name, buffer)

@torch.compiler.disable
def are_features_similar(
    t1: torch.Tensor,
    t2: torch.Tensor,
    threshold: float,
    pool_size: int = 2048,
) -> Tuple[bool, float]:
    """
    Check if two feature tensors are similar using Pooled Feature Summarization (PFS).
    Uses average pooling to create compact "thumbnails" of features,
    then compares them for efficient similarity estimation.
    """
    if t1 is None or t2 is None:
        return False, float('inf')
    
    if t2.ndim == 3:
        # [B, S, D] -> [B, 1, S, D]
        t1 = t1.unsqueeze(1)
        t2 = t2.unsqueeze(1)
    
    B, C, H, W = t2.shape
    stride = max(1, H // pool_size)
    
    thumb1 = F.avg_pool2d(t1.float(), kernel_size=stride, stride=stride)
    thumb2 = F.avg_pool2d(t2.float(), kernel_size=stride, stride=stride)
    
    mean_diff = (thumb1 - thumb2).abs().mean()
    mean_t1 = thumb1.abs().mean()
    diff = (mean_diff / (mean_t1 + 1e-8)).item()
    
    return diff < threshold, diff

@torch.compiler.disable
def check_can_use_cache(
    current_residual: torch.Tensor,
    buffer_name: str,
    threshold: float,
    pool_size: int,
) -> Tuple[bool, float]:
    prev_residual = get_buffer(buffer_name)
    if prev_residual is None:
        return False, threshold
    
    is_similar, diff = are_features_similar(
        prev_residual, current_residual, threshold, pool_size
    )
    return is_similar, diff

class QwenCachedTransformerBlocks(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        threshold_stage1: float = 0.12,
        threshold_stage2: float = 0.09,
        use_h2cache: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer.transformer_blocks
        
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2
        self.use_h2cache = use_h2cache
        self.verbose = verbose
        
        self._current_threshold_s1 = threshold_stage1
        self._current_threshold_s2 = threshold_stage2

    def update_thresholds(
        self,
        threshold_stage1: Optional[float] = None,
        threshold_stage2: Optional[float] = None,
        use_h2cache: Optional[bool] = None,
    ):
        if threshold_stage1 is not None:
            self.threshold_stage1 = threshold_stage1
            self._current_threshold_s1 = threshold_stage1
        if threshold_stage2 is not None:
            self.threshold_stage2 = threshold_stage2
            self._current_threshold_s2 = threshold_stage2
        if use_h2cache is not None:
            self.use_h2cache = use_h2cache

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
        if not self.use_h2cache:
            return self._forward_no_cache(
                hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                temb, image_rotary_emb, block_attention_kwargs, modulate_index,
                controlnet_block_samples
            )
        # Stage 1: Structure blocks (0 to STAGE1_END-1)
        hidden_states, encoder_hidden_states, stage1_hit = self._forward_stage1(
            hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
            temb, image_rotary_emb, block_attention_kwargs, modulate_index,
            controlnet_block_samples
        )
        # Stage 2: Detail blocks (STAGE2_START to end)
        hidden_states, encoder_hidden_states = self._forward_stage2(
            hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
            temb, image_rotary_emb, block_attention_kwargs, modulate_index,
            controlnet_block_samples, force_hit=stage1_hit
        )
        
        return encoder_hidden_states, hidden_states

    def _forward_no_cache(
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
        """Forward pass without caching."""
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

    def _forward_stage1(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        block_attention_kwargs: Dict,
        modulate_index: Optional[torch.Tensor] = None,
        controlnet_block_samples=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Forward through Stage 1 (structure-defining blocks)."""
        original_hidden = hidden_states.clone()
        original_encoder = encoder_hidden_states.clone()
       
        encoder_hidden_states, hidden_states = self.transformer_blocks[0](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=block_attention_kwargs,
            modulate_index=modulate_index,
        )
        
        first_residual = hidden_states - original_hidden
        torch._dynamo.graph_break()
        can_use_cache, diff = check_can_use_cache(
            first_residual, "stage1_first_residual", 
            self._current_threshold_s1, STAGE1_POOL_SIZE
        )
        
        ctx = get_current_cache_context()
        
        if can_use_cache:
            if self.verbose:
                print(f"[STAGE1] Cache HIT diff={diff:.4f}")
            
            hidden_residual = get_buffer("stage1_hidden_residual")
            encoder_residual = get_buffer("stage1_encoder_residual")
            
            hidden_states = original_hidden + hidden_residual
            encoder_hidden_states = original_encoder + encoder_residual
            
            if ctx:
                ctx.stage1_hits += 1
            
            return hidden_states.contiguous(), encoder_hidden_states.contiguous(), True
        
        if self.verbose:
            print(f"[STAGE1] Cache MISS. diff={diff:.4f}, threshold={self._current_threshold_s1:.4f}")
        
        set_buffer("stage1_first_residual", first_residual)
        
        for idx in range(1, STAGE1_END):
            encoder_hidden_states, hidden_states = self.transformer_blocks[idx](
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
        
        # Store Stage 1 residuals for future cache hits
        hidden_residual = hidden_states - original_hidden
        encoder_residual = encoder_hidden_states - original_encoder
        set_buffer("stage1_hidden_residual", hidden_residual)
        set_buffer("stage1_encoder_residual", encoder_residual)
        
        if ctx:
            ctx.total_steps += 1
        
        return hidden_states.contiguous(), encoder_hidden_states.contiguous(), False

    def _forward_stage2(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        block_attention_kwargs: Dict,
        modulate_index: Optional[torch.Tensor] = None,
        controlnet_block_samples=None,
        force_hit: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through Stage 2 (detail-refining blocks)."""
        
        original_hidden = hidden_states.clone()
        original_encoder = encoder_hidden_states.clone()
       
        encoder_hidden_states, hidden_states = self.transformer_blocks[STAGE2_START](
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=block_attention_kwargs,
            modulate_index=modulate_index,
        )
        
        first_residual = hidden_states - original_hidden
        
        if force_hit:
            can_use_cache = True
            diff = self._current_threshold_s2
        else:
            torch._dynamo.graph_break()
            can_use_cache, diff = check_can_use_cache(
                first_residual, "stage2_first_residual",
                self._current_threshold_s2, STAGE2_POOL_SIZE
            )
        
        ctx = get_current_cache_context()
        
        if can_use_cache:
            if self.verbose:
                print(f"[STAGE2] Cache HIT! diff={diff:.4f}")
            
            hidden_residual = get_buffer("stage2_hidden_residual")
            encoder_residual = get_buffer("stage2_encoder_residual")
            
            if hidden_residual is not None and encoder_residual is not None:
                hidden_states = original_hidden + hidden_residual
                encoder_hidden_states = original_encoder + encoder_residual
                
                if ctx:
                    ctx.stage2_hits += 1
                
                return hidden_states.contiguous(), encoder_hidden_states.contiguous()
        
        if self.verbose:
            print(f"[STAGE2] Cache MISS. diff={diff:.4f}, threshold={self._current_threshold_s2:.4f}")
        
        set_buffer("stage2_first_residual", first_residual)
        
        for idx in range(STAGE2_START + 1, NUM_TRANSFORMER_BLOCKS):
            encoder_hidden_states, hidden_states = self.transformer_blocks[idx](
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
        
        hidden_residual = hidden_states - original_hidden
        encoder_residual = encoder_hidden_states - original_encoder
        set_buffer("stage2_hidden_residual", hidden_residual)
        set_buffer("stage2_encoder_residual", encoder_residual)
        
        return hidden_states.contiguous(), encoder_hidden_states.contiguous()


def apply_h2cache_to_transformer(
    transformer: nn.Module,
    threshold_stage1: float = 0.12,
    threshold_stage2: float = 0.09,
    use_h2cache: bool = True,
    verbose: bool = False,
) -> nn.Module:

    actual_transformer = transformer
    if hasattr(transformer, '_orig_mod'):
        actual_transformer = transformer._orig_mod
    
    if getattr(actual_transformer, "_h2cache_applied", False):
        actual_transformer._h2cache_wrapper.update_thresholds(
            threshold_stage1, threshold_stage2, use_h2cache
        )
        return transformer
    
    wrapper = QwenCachedTransformerBlocks(
        transformer=actual_transformer,
        threshold_stage1=threshold_stage1,
        threshold_stage2=threshold_stage2,
        use_h2cache=use_h2cache,
        verbose=verbose,
    )
    
    actual_transformer._h2cache_wrapper = wrapper
    actual_transformer._h2cache_applied = True
    
    return transformer


def apply_h2cache_to_pipeline(
    pipe,
    threshold_stage1: float = 0.12,
    threshold_stage2: float = 0.09,
    use_h2cache: bool = True,
    verbose: bool = False,
):
    apply_h2cache_to_transformer(
        pipe.transformer,
        threshold_stage1=threshold_stage1,
        threshold_stage2=threshold_stage2,
        use_h2cache=use_h2cache,
        verbose=verbose,
    )
    pipe._h2cache_enabled = use_h2cache
    pipe._h2cache_threshold_s1 = threshold_stage1
    pipe._h2cache_threshold_s2 = threshold_stage2
    pipe._h2cache_verbose = verbose
    
    return pipe


def set_h2cache_enabled(pipe_or_transformer, enabled: bool):
    if hasattr(pipe_or_transformer, 'transformer'):
        transformer = pipe_or_transformer.transformer
    else:
        transformer = pipe_or_transformer
    if hasattr(transformer, '_h2cache_wrapper'):
        transformer._h2cache_wrapper.use_h2cache = enabled


def get_cache_stats(pipe_or_transformer) -> Dict:
    ctx = get_current_cache_context()
    if ctx is None:
        return {"stage1_hits": 0, "stage2_hits": 0, "total_steps": 0}
    return {
        "stage1_hits": ctx.stage1_hits,
        "stage2_hits": ctx.stage2_hits,
        "total_steps": ctx.total_steps,
    }
