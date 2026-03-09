import torch
from torch.utils.cpp_extension import load
import os
ext_dir = os.path.dirname(os.path.abspath(__file__))
mxfp8_ext = load(
    name="mxfp8_ext",
    sources=[
        os.path.join(ext_dir, "mxfp8_ext_bindings.cpp"),
        os.path.join(ext_dir, "mxfp8_ext.cu"),
    ],
    extra_cuda_cflags=[
        "-gencode", "arch=compute_100a,code=sm_100a",
        "-O3",
        "-std=c++17",
        "--use_fast_math",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=True,
)
print(f"setup_gemm_config: {mxfp8_ext.setup_gemm_config}")
print(f"torch.ops.mxfp8v4.linear: {torch.ops.mxfp8v4.linear}")
