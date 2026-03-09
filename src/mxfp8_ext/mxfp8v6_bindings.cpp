#include <torch/extension.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <mutex>

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
}

struct V6Config {
    using capi_func_t = int32_t (*)(void**);
    capi_func_t capi_func = nullptr;

    std::vector<void*> packed_args;
    int stream_arg_idx = -1;

    // Persistent storage for mutable values
    cudaStream_t stream_value = nullptr;

    // Pre-allocated buffers (owned by Python, stored here for fast access)
    uint8_t* A_q_ptr = nullptr;
    uint8_t* SFA_ptr = nullptr;
    __nv_bfloat16* x_pad_ptr = nullptr;
    __nv_bfloat16* C_ptr = nullptr;

    int M_pad = 0;
    int K = 0;
    int N = 0;
    int sf_k = 0;
    bool ready = false;
};

// Simple array-based config storage (no hash, no mutex in hot path)
static constexpr int MAX_V6_CONFIGS = 2048;
static V6Config g_v6_configs[MAX_V6_CONFIGS];
static int g_v6_next_id = 0;
static std::mutex g_v6_init_mutex;  // only used during setup, not hot path

static int64_t v6_create_config(
    int M_pad, int K, int N,
    // capi_func and packed args from CuTeDSL
    uintptr_t capi_func_ptr,
    std::vector<int64_t> packed_arg_values,
    int stream_arg_idx,
    // Pre-allocated buffer pointers
    int64_t A_q_ptr, int64_t SFA_ptr, int64_t x_pad_ptr, int64_t C_ptr)
{
    std::lock_guard<std::mutex> lock(g_v6_init_mutex);
    int id = g_v6_next_id++;
    TORCH_CHECK(id < MAX_V6_CONFIGS, "Too many V6 configs (max ", MAX_V6_CONFIGS, ")");

    V6Config& cfg = g_v6_configs[id];
    cfg.capi_func = reinterpret_cast<V6Config::capi_func_t>(capi_func_ptr);
    cfg.stream_arg_idx = stream_arg_idx;

    cfg.packed_args.resize(packed_arg_values.size());
    for (size_t i = 0; i < packed_arg_values.size(); i++) {
        cfg.packed_args[i] = reinterpret_cast<void*>(packed_arg_values[i]);
    }

    cfg.A_q_ptr = reinterpret_cast<uint8_t*>(A_q_ptr);
    cfg.SFA_ptr = reinterpret_cast<uint8_t*>(SFA_ptr);
    cfg.x_pad_ptr = reinterpret_cast<__nv_bfloat16*>(x_pad_ptr);
    cfg.C_ptr = reinterpret_cast<__nv_bfloat16*>(C_ptr);

    cfg.M_pad = M_pad;
    cfg.K = K;
    cfg.N = N;
    cfg.sf_k = K / 32;
    cfg.ready = true;

    return static_cast<int64_t>(id);
}

// Update the stream pointer in packed_args to point to our persistent storage
static void v6_patch_stream(int64_t config_id) {
    V6Config& cfg = g_v6_configs[static_cast<int>(config_id)];
    if (cfg.stream_arg_idx >= 0) {
        cfg.packed_args[cfg.stream_arg_idx] = &cfg.stream_value;
    }
}

static torch::Tensor v6_linear_cuda(
    const torch::Tensor& x_2d,    // [M, K] bfloat16
    const torch::Tensor& C_buf,   // [M_pad, N] bfloat16 (pre-allocated output)
    int64_t config_id)
{
    const V6Config& cfg = g_v6_configs[static_cast<int>(config_id)];
    const int M = x_2d.size(0);
    const int M_pad = cfg.M_pad;
    const int K = cfg.K;
    auto stream = at::cuda::getCurrentCUDAStream(x_2d.device().index()).stream();

    // Step 1: Get input pointer (pad if needed)
    const __nv_bfloat16* x_ptr;
    if (M < M_pad) {
        // Copy to padded buffer (async)
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

    // Step 3: Update stream and launch GEMM
    // const_cast is safe: we're updating our own persistent storage
    const_cast<V6Config&>(cfg).stream_value = stream;
    const_cast<V6Config&>(cfg).capi_func(
        const_cast<V6Config&>(cfg).packed_args.data());

    // Return view of pre-allocated output (no allocation)
    return C_buf.narrow(0, 0, M);
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
    m.doc() = "MXFP8 V6 optimized inference extension";

    m.def("create_config", &v6_create_config,
        "Create a pre-configured layer config, returns config_id",
        py::arg("M_pad"), py::arg("K"), py::arg("N"),
        py::arg("capi_func_ptr"),
        py::arg("packed_arg_values"),
        py::arg("stream_arg_idx"),
        py::arg("A_q_ptr"), py::arg("SFA_ptr"),
        py::arg("x_pad_ptr"), py::arg("C_ptr"));

    m.def("patch_stream", &v6_patch_stream,
        "Patch stream pointer in packed args",
        py::arg("config_id"));
}
