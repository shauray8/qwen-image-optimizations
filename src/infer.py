import os
import sys
import time
import argparse

sys.path.insert(0, '/workspace/triton/python')
import torch

sys.path.append('/workspace')

os.environ['PATH'] = '/opt/nvidia/nsight-systems/2026.1.1/bin:' + os.environ.get('PATH', '')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')

p = argparse.ArgumentParser()
p.add_argument('--v14', action='store_true', help='V14: fused QKV-pack kernel only')
p.add_argument('--euler', action='store_true', help='Use Euler scheduler (default: DPM-Solver++)')
p.add_argument('--profile-nsys', action='store_true', help='Minimal run for nsys capture')
p.add_argument('--steps', type=int, default=None, help='Denoising steps (default: 14 for DPM2, 28 for Euler)')
p.add_argument('--height', type=int, default=1024)
p.add_argument('--width', type=int, default=1024)
args = p.parse_args()

use_euler = args.euler or args.v10_baseline or args.v14
if args.steps is None:
    args.steps = 28 if use_euler else 14

PROMPT = (
    'Black-and-white vintage-style photography, high contrast lighting, '
    'frontal close-up portrait framing. A young woman is shown from the upper '
    'torso upward. Her skin appears uniformly pale in the monochrome image. '
    'Her hair is long, straight, and extremely smooth, rendered in near-white '
    'tones. A blunt, perfectly horizontal fringe falls straight across her '
    'face, fully covering both eyes. The fringe is dense, opaque, and uniform, '
    'forming a sharp horizontal edge. The remaining hair falls symmetrically '
    'on both sides of her head in straight vertical sheets extending past the '
    'shoulders. Individual strands are fine and tightly aligned, producing a '
    'sleek surface with minimal volume. Soft linear highlights run vertically '
    'along the hair, confirming its straight alignment and smooth texture. A '
    'narrow shadow beneath the fringe confirms its thickness and separation '
    'from the face. matte black lipstick on her lips. A narrow matte black '
    'ribbon is wrapped tightly around her eyes, positioned over the fringe. '
    'the word "EYES" is written on the ribbon in bright red ink in a repeating '
    'pattern. The background is a uniform black field with no visible texture'
)
NEG = (
    '低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，'
    '过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。'
)

sched_label = 'Euler' if use_euler else f'DPM2'
if args.v14:
    label = 'V14'
    desc  = f'V12 + fused QKV-pack kernel + {sched_label}-{args.steps}'
else:
    label = f'V15-DPM2-{args.steps}' if not use_euler else 'V15-Euler'
    desc  = f'V15 kernels + {sched_label}-{args.steps} scheduler'
print(f'{label}: {desc}')

from models.transformer import QwenImageTransformer2DModel
from models.transformer_rowwise_fp8 import apply_optimized_rowwise_fp8
from pipeline import QwenImagePipeline
trans = QwenImageTransformer2DModel.from_pretrained(
    'Qwen/Qwen-Image-2512', subfolder='transformer', torch_dtype=torch.bfloat16
).to('cuda')
apply_optimized_rowwise_fp8(trans, verbose=False)

if args.v14:
    from models.optimized_block_v14 import apply_v14_optimizations
    apply_v14_optimizations(trans, verbose=True)
else:
    from models.optimized_block_v15 import apply_v15_optimizations
    apply_v15_optimizations(trans, verbose=True)
pipe = QwenImagePipeline.from_pretrained(
    'Qwen/Qwen-Image-2512', transformer=trans, torch_dtype=torch.bfloat16
).to('cuda')

if not use_euler:
    import numpy as np
    from diffusers import DPMSolverMultistepScheduler
    from pipeline import calculate_shift
    _packed_h = 2 * (args.height // (pipe.vae_scale_factor * 2))
    _packed_w = 2 * (args.width // (pipe.vae_scale_factor * 2))
    _image_seq_len = (_packed_h // 2) * (_packed_w // 2)
    _mu = calculate_shift(
        _image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler = DPMSolverMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        flow_shift=np.exp(_mu),
        solver_order=2,
        algorithm_type='dpmsolver++',
        final_sigmas_type='zero',
        lower_order_final=True,
    )
    print(f'Scheduler: DPM-Solver++ order=2, {args.steps} steps, mu={_mu:.4f}')

WARMUP = 3 if args.profile_nsys else 6
BENCH  = 2 if args.profile_nsys else 8

print(f'\nWarmup ({WARMUP} runs)...')
for i in range(WARMUP):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        img = pipe(
            prompt=PROMPT, negative_prompt=NEG,
            num_inference_steps=args.steps,
            height=args.height, width=args.width,
            true_cfg_scale=4.0,
        ).images[0]
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f'  warmup {i+1}: {elapsed:.3f}s  ({args.steps/elapsed:.1f} it/s)')

if args.profile_nsys:
    import torch.cuda.profiler as profiler
    import torch.cuda.nvtx as nvtx
    print('\nNVTX profiling range started...')
    profiler.start()

print(f'\nBenchmark ({BENCH} runs)...')
times = []
for i in range(BENCH):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        if args.profile_nsys:
            nvtx.range_push(f'inference_{i}')
        img = pipe(
            prompt=PROMPT, negative_prompt=NEG,
            num_inference_steps=args.steps,
            height=args.height, width=args.width,
            true_cfg_scale=4.0,
        ).images[0]
        if args.profile_nsys:
            nvtx.range_pop()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    print(f'  run {i+1}: {elapsed:.3f}s  ({args.steps/elapsed:.1f} it/s)')

if args.profile_nsys:
    profiler.stop()

eff  = times[min(2, len(times) - 1):]
avg  = sum(eff) / len(eff)
best = min(eff)

img.save(f'{label}_result.png')

print(f'RESULTS — {label}')
print(f'Config:   {args.steps} steps, {args.height}×{args.width}')
print(f'Average:  {avg:.3f}s   ({args.steps/avg:.2f} it/s)')
print(f'Best:     {best:.3f}s   ({args.steps/best:.2f} it/s)')
print(f'GPU mem:  {torch.cuda.max_memory_allocated()/1e9:.2f} GB')
print(f'Saved:    {label}_result.png')
