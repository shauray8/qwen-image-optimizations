#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cstring>

// Forward declarations from mxfp8_ext.cu
namespace mxfp8 {
cudaError_t launch_quant_fp8_e1(
    const __nv_bfloat16* x, uint8_t* out_fp8, uint8_t* sf_mma,
    int M_pad, int K, int sf_k, cudaStream_t stream);
cudaError_t launch_metadata_e1(
    int64_t A_ptr, int64_t A_row_bytes,
    int64_t B_ptr, int64_t B_expert_bytes,
    int64_t C_ptr, int64_t C_row_bytes,
    int64_t SFA_ptr, int64_t SFA_row_bytes,
    int64_t SFB_ptr, int64_t SFB_expert_bytes,
    int32_t A_stride0, int32_t A_stride1,
    int32_t B_stride0, int32_t B_stride1,
    int32_t C_stride0, int32_t C_stride1,
    int32_t M_pad, int32_t N, int32_t K,
    int32_t* sizes_mnkl, int32_t* strides_abc,
    int64_t* ptrs_abc, int64_t* ptrs_sfasfb,
    cudaStream_t stream);
cudaError_t launch_rowwise_fp8_quant(
    const __nv_bfloat16* input, uint8_t* output, float* scales,
    int M, int K, cudaStream_t stream);
}

struct QuantBufs {
    torch::Tensor A_q;       // [M_pad, K] uint8 (FP8 E4M3)
    torch::Tensor SFA;       // [sf_size] uint8 (E8M0 tcgen05 swizzled)
    torch::Tensor x_pad;     // [M_pad, K] bfloat16 (padded input)
};

static std::unordered_map<int64_t, QuantBufs> g_quant_bufs;
static std::mutex g_bufs_mutex;

static int64_t make_buf_key(int device_idx, int M_pad, int K) {
    return (static_cast<int64_t>(device_idx) << 48) |
           (static_cast<int64_t>(M_pad) << 24) |
           static_cast<int64_t>(K);
}

static QuantBufs& get_quant_bufs(int device_idx, int M_pad, int K) {
    int64_t key = make_buf_key(device_idx, M_pad, K);
    std::lock_guard<std::mutex> lock(g_bufs_mutex);
    auto it = g_quant_bufs.find(key);
    if (it != g_quant_bufs.end()) {
        return it->second;
    }

    auto device = torch::Device(torch::kCUDA, device_idx);
    auto opts_u8 = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    int sf_k = K / 32;
    int n_atoms_m = M_pad / 128;
    int n_atoms_k = sf_k / 4;
    int sf_size = n_atoms_m * n_atoms_k * 512;

    QuantBufs bufs;
    bufs.A_q = torch::empty({M_pad, K}, opts_u8);
    bufs.SFA = torch::empty({sf_size}, opts_u8);
    bufs.x_pad = torch::zeros({M_pad, K}, opts_bf16);

    g_quant_bufs[key] = std::move(bufs);
    return g_quant_bufs[key];
}

static int64_t ensure_quant_bufs(int device_idx, int M_pad, int K) {
    auto& bufs = get_quant_bufs(device_idx, M_pad, K);
    return reinterpret_cast<int64_t>(bufs.A_q.data_ptr());
}

struct GEMMConfig {
    using capi_func_t = int32_t (*)(void**);
    capi_func_t capi_func = nullptr;

    // Pre-packed args buffer (void* array).
    // IMPORTANT: CuTeDSL uses pointer-to-value semantics.
    // packed_args[i] is a POINTER TO where arg i's value lives.
    std::vector<void*> packed_args;
    int num_user_args = 0;

    // Indices for runtime-updated args
    int stream_arg_idx = -1;
    int a_ptr_arg_idx = -1;

    // Persistent storage for values that packed_args entries point to.
    // packed_args[stream_arg_idx] = &stream_value
    // packed_args[a_ptr_arg_idx] = &a_ptr_value
    cudaStream_t stream_value = nullptr;
    int64_t a_ptr_value = 0;

    // Constant strides for metadata builder
    int32_t A_stride0 = 0, A_stride1 = 0;
    int32_t B_stride0 = 0, B_stride1 = 0;
    int32_t C_stride0 = 0, C_stride1 = 0;
    int64_t A_row_bytes = 0, B_expert_bytes = 0;
    int64_t SFA_row_bytes = 0, SFB_expert_bytes = 0;

    // Workspace tensor pointers (for metadata builder)
    int32_t* sizes_mnkl_ptr = nullptr;
    int32_t* strides_abc_ptr = nullptr;
    int64_t* ptrs_abc_ptr = nullptr;
    int64_t* ptrs_sfasfb_ptr = nullptr;

    int M_pad = 0, K = 0, N = 0;
    bool a_ptr_patched = false;
    bool ready = false;
};

static std::unordered_map<int64_t, GEMMConfig> g_gemm_configs;
static std::mutex g_gemm_mutex;

static int64_t make_gemm_key(int device_idx, int M_pad, int K, int N, int64_t w_ptr) {
    int64_t key = device_idx;
    key = key * 31 + M_pad;
    key = key * 31 + K;
    key = key * 31 + N;
    key = key * 31 + (w_ptr >> 8);
    return key;
}

void setup_gemm_config(
    int device_idx, int M_pad, int K, int N, int64_t w_ptr,
    uintptr_t capi_func_ptr,
    std::vector<int64_t> packed_arg_values,
    int num_user_args,
    int stream_arg_idx,
    int a_ptr_arg_idx,
    int32_t A_stride0, int32_t A_stride1,
    int32_t B_stride0, int32_t B_stride1,
    int32_t C_stride0, int32_t C_stride1,
    int64_t A_row_bytes, int64_t B_expert_bytes,
    int64_t SFA_row_bytes, int64_t SFB_expert_bytes,
    uintptr_t sizes_mnkl_ptr, uintptr_t strides_abc_ptr,
    uintptr_t ptrs_abc_ptr, uintptr_t ptrs_sfasfb_ptr)
{
    int64_t key = make_gemm_key(device_idx, M_pad, K, N, w_ptr);
    std::lock_guard<std::mutex> lock(g_gemm_mutex);

    GEMMConfig& cfg = g_gemm_configs[key];
    cfg.capi_func = reinterpret_cast<GEMMConfig::capi_func_t>(capi_func_ptr);
    cfg.num_user_args = num_user_args;
    cfg.stream_arg_idx = stream_arg_idx;
    cfg.a_ptr_arg_idx = a_ptr_arg_idx;

    cfg.packed_args.resize(packed_arg_values.size());
    for (size_t i = 0; i < packed_arg_values.size(); i++) {
        cfg.packed_args[i] = reinterpret_cast<void*>(packed_arg_values[i]);
    }

    cfg.A_stride0 = A_stride0; cfg.A_stride1 = A_stride1;
    cfg.B_stride0 = B_stride0; cfg.B_stride1 = B_stride1;
    cfg.C_stride0 = C_stride0; cfg.C_stride1 = C_stride1;
    cfg.A_row_bytes = A_row_bytes;
    cfg.B_expert_bytes = B_expert_bytes;
    cfg.SFA_row_bytes = SFA_row_bytes;
    cfg.SFB_expert_bytes = SFB_expert_bytes;

    cfg.sizes_mnkl_ptr = reinterpret_cast<int32_t*>(sizes_mnkl_ptr);
    cfg.strides_abc_ptr = reinterpret_cast<int32_t*>(strides_abc_ptr);
    cfg.ptrs_abc_ptr = reinterpret_cast<int64_t*>(ptrs_abc_ptr);
    cfg.ptrs_sfasfb_ptr = reinterpret_cast<int64_t*>(ptrs_sfasfb_ptr);

    cfg.M_pad = M_pad; cfg.K = K; cfg.N = N;
    cfg.a_ptr_patched = false;
    cfg.ready = true;
}

static torch::Tensor mxfp8v4_linear_cuda(
    const torch::Tensor& x_2d,
    const torch::Tensor& W_q,
    const torch::Tensor& W_sf_mma,
    int64_t K,
    int64_t N)
{
    const int M = x_2d.size(0);
    const int M_pad = ((M + 127) / 128) * 128;
    const int sf_k = K / 32;

    auto device = x_2d.device();
    const int device_idx = device.index();

    auto& bufs = get_quant_bufs(device_idx, M_pad, K);
    auto stream = at::cuda::getCurrentCUDAStream(device_idx).stream();

    // Step 1: Pad input if needed
    if (M_pad != M) {
        bufs.x_pad.narrow(0, 0, M).copy_(x_2d, /*non_blocking=*/true);
    }

    const __nv_bfloat16* x_ptr = (M_pad != M)
        ? reinterpret_cast<const __nv_bfloat16*>(bufs.x_pad.data_ptr())
        : reinterpret_cast<const __nv_bfloat16*>(x_2d.data_ptr());

    // Step 2: Launch FP8 quant kernel
    auto err = mxfp8::launch_quant_fp8_e1(
        x_ptr,
        reinterpret_cast<uint8_t*>(bufs.A_q.data_ptr()),
        reinterpret_cast<uint8_t*>(bufs.SFA.data_ptr()),
        M_pad, K, sf_k, stream);
    TORCH_CHECK(err == cudaSuccess, "quant kernel failed: ", cudaGetErrorString(err));

    // Step 3: Allocate output
    auto C = torch::empty({M_pad, N, 1},
        torch::TensorOptions().dtype(torch::kBFloat16).device(device));

    // Step 4: Look up GEMM config
    int64_t gemm_key = make_gemm_key(device_idx, M_pad, K, N,
        reinterpret_cast<int64_t>(W_q.data_ptr()));
    GEMMConfig* cfg = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_gemm_mutex);
        auto it = g_gemm_configs.find(gemm_key);
        if (it != g_gemm_configs.end() && it->second.ready) {
            cfg = &it->second;
        }
    }

    if (cfg == nullptr) {
        return C.narrow(0, 0, M).select(2, 0);
    }

    // Step 5: Patch A_ptr (once) — pointer-to-value semantics
    if (!cfg->a_ptr_patched && cfg->a_ptr_arg_idx >= 0) {
        cfg->a_ptr_value = reinterpret_cast<int64_t>(bufs.A_q.data_ptr());
        cfg->packed_args[cfg->a_ptr_arg_idx] = &cfg->a_ptr_value;
        cfg->a_ptr_patched = true;
    }

    // Step 6: Build metadata
    err = mxfp8::launch_metadata_e1(
        reinterpret_cast<int64_t>(bufs.A_q.data_ptr()),
        cfg->A_row_bytes,
        reinterpret_cast<int64_t>(W_q.data_ptr()),
        cfg->B_expert_bytes,
        reinterpret_cast<int64_t>(C.data_ptr()),
        C.stride(0) * 2,
        reinterpret_cast<int64_t>(bufs.SFA.data_ptr()),
        cfg->SFA_row_bytes,
        reinterpret_cast<int64_t>(W_sf_mma.data_ptr()),
        cfg->SFB_expert_bytes,
        cfg->A_stride0, cfg->A_stride1,
        cfg->B_stride0, cfg->B_stride1,
        static_cast<int32_t>(C.stride(0)), static_cast<int32_t>(C.stride(1)),
        M_pad, N, K,
        cfg->sizes_mnkl_ptr, cfg->strides_abc_ptr,
        cfg->ptrs_abc_ptr, cfg->ptrs_sfasfb_ptr,
        stream);
    TORCH_CHECK(err == cudaSuccess, "metadata kernel failed: ", cudaGetErrorString(err));

    // Step 7: Update stream (pointer-to-value) and launch GEMM
    cfg->stream_value = stream;
    cfg->packed_args[cfg->stream_arg_idx] = &cfg->stream_value;
    int32_t ret = cfg->capi_func(cfg->packed_args.data());
    TORCH_CHECK(ret == 0, "GEMM capi_func returned error: ", ret);

    return C.narrow(0, 0, M).select(2, 0);
}

// Meta implementation for torch.compile
static torch::Tensor mxfp8v4_linear_meta(
    const torch::Tensor& x_2d,
    const torch::Tensor& W_q,
    const torch::Tensor& W_sf_mma,
    int64_t K,
    int64_t N)
{
    int M = x_2d.size(0);
    return torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x_2d.device()));
}

static std::tuple<torch::Tensor, torch::Tensor> rowwise_fp8_quant_cuda(
    const torch::Tensor& x)
{
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBFloat16, "Input must be BFloat16");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D [M, K]");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    const int M = x.size(0);
    const int K = x.size(1);
    TORCH_CHECK(K % 128 == 0, "K must be divisible by 128");

    auto device = x.device();
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();

    // Allocate output tensors (CUDA graph trees manage memory reuse)
    auto output_u8 = torch::empty({M, K},
        torch::TensorOptions().dtype(torch::kUInt8).device(device));
    auto scales = torch::empty({M, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));

    auto err = mxfp8::launch_rowwise_fp8_quant(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
        reinterpret_cast<uint8_t*>(output_u8.data_ptr()),
        scales.data_ptr<float>(),
        M, K, stream);
    TORCH_CHECK(err == cudaSuccess, "rowwise_fp8_quant failed: ", cudaGetErrorString(err));

    // Reinterpret uint8 as float8_e4m3fn
    auto output_fp8 = output_u8.view(torch::kFloat8_e4m3fn);
    return std::make_tuple(output_fp8, scales);
}

static std::tuple<torch::Tensor, torch::Tensor> rowwise_fp8_quant_meta(
    const torch::Tensor& x)
{
    const int M = x.size(0);
    const int K = x.size(1);
    auto fp8_out = torch::empty({M, K},
        torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(x.device()));
    auto scales = torch::empty({M, 1},
        torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));
    return std::make_tuple(fp8_out, scales);
}

struct V6Config {
    using capi_func_t = int32_t (*)(void**);
    capi_func_t capi_func = nullptr;

    std::vector<void*> packed_args;
    int stream_arg_idx = -1;

    // Persistent storage for stream (packed_args points here)
    cudaStream_t stream_value = nullptr;

    // Pre-allocated buffer raw pointers (owned by Python tensors)
    uint8_t* A_q_ptr = nullptr;
    uint8_t* SFA_ptr = nullptr;
    __nv_bfloat16* x_pad_ptr = nullptr;

    // Metadata workspace pointers (for updating C_ptr before GEMM)
    int64_t* ptrs_abc_ptr = nullptr;  // workspace ptrs_abc tensor
    int64_t* ptrs_sfasfb_ptr = nullptr;
    int32_t* sizes_mnkl_ptr = nullptr;
    int32_t* strides_abc_ptr = nullptr;

    int M_pad = 0, K = 0, N = 0, sf_k = 0;

    // Pre-computed metadata values (written to workspace before GEMM)
    int64_t fixed_A_ptr = 0, fixed_SFA_ptr = 0;
    int64_t fixed_B_ptr = 0, fixed_SFB_ptr = 0;
    int64_t A_row_bytes = 0, B_expert_bytes = 0;
    int64_t SFA_row_bytes = 0, SFB_expert_bytes = 0;
    int32_t A_stride0 = 0, A_stride1 = 0;
    int32_t B_stride0 = 0, B_stride1 = 0;
    int32_t C_stride0 = 0, C_stride1 = 0;

    bool ready = false;
};

static constexpr int MAX_V6_CONFIGS = 2048;
static V6Config g_v6_configs[MAX_V6_CONFIGS];
static int g_v6_next_id = 0;
static std::mutex g_v6_init_mutex;

static int64_t v6_create_config(
    int M_pad, int K, int N,
    uintptr_t capi_func_ptr,
    std::vector<int64_t> packed_arg_values,
    int stream_arg_idx,
    int64_t A_q_ptr, int64_t SFA_ptr, int64_t x_pad_ptr,
    int64_t B_ptr, int64_t SFB_ptr,
    int64_t A_row_bytes, int64_t B_expert_bytes,
    int64_t SFA_row_bytes, int64_t SFB_expert_bytes,
    int32_t A_stride0, int32_t A_stride1,
    int32_t B_stride0, int32_t B_stride1,
    int32_t C_stride0, int32_t C_stride1,
    uintptr_t sizes_mnkl_ptr, uintptr_t strides_abc_ptr,
    uintptr_t ptrs_abc_ptr, uintptr_t ptrs_sfasfb_ptr)
{
    std::lock_guard<std::mutex> lock(g_v6_init_mutex);
    int id = g_v6_next_id++;
    TORCH_CHECK(id < MAX_V6_CONFIGS, "Too many V6 configs");

    V6Config& cfg = g_v6_configs[id];
    cfg.capi_func = reinterpret_cast<V6Config::capi_func_t>(capi_func_ptr);
    cfg.stream_arg_idx = stream_arg_idx;

    cfg.packed_args.resize(packed_arg_values.size());
    for (size_t i = 0; i < packed_arg_values.size(); i++) {
        cfg.packed_args[i] = reinterpret_cast<void*>(packed_arg_values[i]);
    }
    // Patch stream to point to our persistent storage
    if (stream_arg_idx >= 0) {
        cfg.packed_args[stream_arg_idx] = &cfg.stream_value;
    }

    cfg.A_q_ptr = reinterpret_cast<uint8_t*>(A_q_ptr);
    cfg.SFA_ptr = reinterpret_cast<uint8_t*>(SFA_ptr);
    cfg.x_pad_ptr = reinterpret_cast<__nv_bfloat16*>(x_pad_ptr);

    cfg.ptrs_abc_ptr = reinterpret_cast<int64_t*>(ptrs_abc_ptr);
    cfg.ptrs_sfasfb_ptr = reinterpret_cast<int64_t*>(ptrs_sfasfb_ptr);
    cfg.sizes_mnkl_ptr = reinterpret_cast<int32_t*>(sizes_mnkl_ptr);
    cfg.strides_abc_ptr = reinterpret_cast<int32_t*>(strides_abc_ptr);

    cfg.M_pad = M_pad; cfg.K = K; cfg.N = N;
    cfg.sf_k = K / 32;

    cfg.fixed_A_ptr = A_q_ptr;
    cfg.fixed_SFA_ptr = SFA_ptr;
    cfg.fixed_B_ptr = B_ptr;
    cfg.fixed_SFB_ptr = SFB_ptr;
    cfg.A_row_bytes = A_row_bytes;
    cfg.B_expert_bytes = B_expert_bytes;
    cfg.SFA_row_bytes = SFA_row_bytes;
    cfg.SFB_expert_bytes = SFB_expert_bytes;
    cfg.A_stride0 = A_stride0; cfg.A_stride1 = A_stride1;
    cfg.B_stride0 = B_stride0; cfg.B_stride1 = B_stride1;
    cfg.C_stride0 = C_stride0; cfg.C_stride1 = C_stride1;

    cfg.ready = true;
    return static_cast<int64_t>(id);
}

// V6 hot path: quant + metadata + GEMM with pre-allocated output
static torch::Tensor v6_linear_cuda(
    const torch::Tensor& x_2d,
    const torch::Tensor& C_buf,
    int64_t config_id)
{
    V6Config& cfg = g_v6_configs[static_cast<int>(config_id)];
    const int M = x_2d.size(0);
    const int M_pad = cfg.M_pad;
    const int K = cfg.K;
    const int N = cfg.N;
    auto stream = at::cuda::getCurrentCUDAStream(x_2d.device().index()).stream();

    // Step 1: Input pointer (skip copy if already aligned)
    const __nv_bfloat16* x_ptr;
    if (M < M_pad) {
        cudaMemcpy2DAsync(
            cfg.x_pad_ptr, K * sizeof(__nv_bfloat16),
            x_2d.data_ptr(), K * sizeof(__nv_bfloat16),
            K * sizeof(__nv_bfloat16), M,
            cudaMemcpyDeviceToDevice, stream);
        x_ptr = cfg.x_pad_ptr;
    } else {
        x_ptr = reinterpret_cast<const __nv_bfloat16*>(x_2d.data_ptr());
    }

    // Step 2: Blockscale FP8 quantization
    mxfp8::launch_quant_fp8_e1(
        x_ptr, cfg.A_q_ptr, cfg.SFA_ptr,
        M_pad, K, cfg.sf_k, stream);

    // Step 3: Build metadata (A_q, SFA are pre-allocated so ptrs are fixed;
    // only C_ptr changes if different output buffer)
    auto C_ptr_val = reinterpret_cast<int64_t>(C_buf.data_ptr());
    mxfp8::launch_metadata_e1(
        cfg.fixed_A_ptr, cfg.A_row_bytes,
        cfg.fixed_B_ptr, cfg.B_expert_bytes,
        C_ptr_val, cfg.C_stride0 * 2,
        cfg.fixed_SFA_ptr, cfg.SFA_row_bytes,
        cfg.fixed_SFB_ptr, cfg.SFB_expert_bytes,
        cfg.A_stride0, cfg.A_stride1,
        cfg.B_stride0, cfg.B_stride1,
        cfg.C_stride0, cfg.C_stride1,
        M_pad, N, K,
        cfg.sizes_mnkl_ptr, cfg.strides_abc_ptr,
        cfg.ptrs_abc_ptr, cfg.ptrs_sfasfb_ptr,
        stream);

    // Step 4: Update stream and launch GEMM
    cfg.stream_value = stream;
    cfg.capi_func(cfg.packed_args.data());

    // Return view of pre-allocated output (zero allocation)
    if (M < M_pad) {
        return C_buf.narrow(0, 0, M);
    }
    return C_buf;
}

static torch::Tensor v6_linear_meta(
    const torch::Tensor& x_2d,
    const torch::Tensor& C_buf,
    int64_t config_id)
{
    int M = x_2d.size(0);
    int N = C_buf.size(1);
    return torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(x_2d.device()));
}

TORCH_LIBRARY(mxfp8v4, m) {
    m.def("linear(Tensor x, Tensor W_q, Tensor W_sf, int K, int N) -> Tensor");
    m.def("rowwise_fp8_quant(Tensor x) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(mxfp8v4, CUDA, m) {
    m.impl("linear", &mxfp8v4_linear_cuda);
    m.impl("rowwise_fp8_quant", &rowwise_fp8_quant_cuda);
}

TORCH_LIBRARY_IMPL(mxfp8v4, Meta, m) {
    m.impl("linear", &mxfp8v4_linear_meta);
    m.impl("rowwise_fp8_quant", &rowwise_fp8_quant_meta);
}

TORCH_LIBRARY(mxfp8v6, m) {
    m.def("linear(Tensor x, Tensor C_buf, int config_id) -> Tensor");
}

TORCH_LIBRARY_IMPL(mxfp8v6, CUDA, m) {
    m.impl("linear", &v6_linear_cuda);
}

TORCH_LIBRARY_IMPL(mxfp8v6, Meta, m) {
    m.impl("linear", &v6_linear_meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "MXFP8 fused inference extension (B200)";

    m.def("ensure_quant_bufs", &ensure_quant_bufs,
        "Create quant buffers for (device, M_pad, K) and return A_q data_ptr",
        py::arg("device_idx"), py::arg("M_pad"), py::arg("K"));

    m.def("setup_gemm_config", &setup_gemm_config,
        "Register GEMM configuration for a specific layer",
        py::arg("device_idx"), py::arg("M_pad"), py::arg("K"), py::arg("N"),
        py::arg("w_ptr"), py::arg("capi_func_ptr"),
        py::arg("packed_arg_values"), py::arg("num_user_args"),
        py::arg("stream_arg_idx"), py::arg("a_ptr_arg_idx"),
        py::arg("A_stride0"), py::arg("A_stride1"),
        py::arg("B_stride0"), py::arg("B_stride1"),
        py::arg("C_stride0"), py::arg("C_stride1"),
        py::arg("A_row_bytes"), py::arg("B_expert_bytes"),
        py::arg("SFA_row_bytes"), py::arg("SFB_expert_bytes"),
        py::arg("sizes_mnkl_ptr"), py::arg("strides_abc_ptr"),
        py::arg("ptrs_abc_ptr"), py::arg("ptrs_sfasfb_ptr"));

    m.def("v6_create_config", &v6_create_config,
        "Create V6 layer config with pre-allocated buffers",
        py::arg("M_pad"), py::arg("K"), py::arg("N"),
        py::arg("capi_func_ptr"),
        py::arg("packed_arg_values"),
        py::arg("stream_arg_idx"),
        py::arg("A_q_ptr"), py::arg("SFA_ptr"), py::arg("x_pad_ptr"),
        py::arg("B_ptr"), py::arg("SFB_ptr"),
        py::arg("A_row_bytes"), py::arg("B_expert_bytes"),
        py::arg("SFA_row_bytes"), py::arg("SFB_expert_bytes"),
        py::arg("A_stride0"), py::arg("A_stride1"),
        py::arg("B_stride0"), py::arg("B_stride1"),
        py::arg("C_stride0"), py::arg("C_stride1"),
        py::arg("sizes_mnkl_ptr"), py::arg("strides_abc_ptr"),
        py::arg("ptrs_abc_ptr"), py::arg("ptrs_sfasfb_ptr"));
}
