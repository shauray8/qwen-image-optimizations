#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace mxfp8 {
static constexpr float FP8_MAX = 448.0f;     // E4M3 max finite value
static constexpr int SF_VEC = 32;             // Scale factor granularity

__host__ __device__ __forceinline__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Encode positive float to E8M0 (ceil power-of-two): byte = clamp(exponent + (mantissa > 0), 0, 254)
__device__ __forceinline__ uint8_t e8m0_encode_from_pos_f32(float scale) {
    uint32_t bits;
    asm volatile("mov.b32 %0, %1;" : "=r"(bits) : "f"(scale));
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    uint32_t e8m0 = (mant > 0) ? (exp + 1) : exp;
    return static_cast<uint8_t>(min(e8m0, 254u));
}

// Decode E8M0 to inverse scale: 2^(127 + (127 - e8m0)) = 2^(254 - e8m0)
__device__ __forceinline__ float e8m0_inv_decode_to_f32(uint8_t e8m0) {
    uint32_t inv_exp = 127u + (127u - static_cast<uint32_t>(e8m0));
    inv_exp = min(max(inv_exp, 0u), 254u);
    uint32_t bits = inv_exp << 23;
    float result;
    asm volatile("mov.b32 %0, %1;" : "=f"(result) : "r"(bits));
    return result;
}

// Convert float to FP8 E4M3 byte via PTX
__device__ __forceinline__ uint8_t f32_to_e4m3_byte(float v) {
    uint16_t result;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(result) : "f"(v));
    return static_cast<uint8_t>(result & 0xFF);
}

// tcgen05 MMA scale factor swizzle offset
// Matches nmoe/csrc/swizzle.cuh exactly.
// atom shape = (128, 4), atom size = 512 bytes
// Within atom: m_32 * 16 + m_4 * 4 + k_4

__device__ __forceinline__ int sf_swizzle_offset(int m, int k_sf, int sf_k) {
    const int m_32 = m & 31;
    const int m_4 = (m >> 5) & 3;
    const int m_rest = m >> 7;
    const int k_4 = k_sf & 3;
    const int k_rest = k_sf >> 2;
    const int rest_k = sf_k >> 2;
    const int atom_offset = m_32 * 16 + m_4 * 4 + k_4;
    const int atom_idx = m_rest * rest_k + k_rest;
    return atom_idx * 512 + atom_offset;
}

// Optimized for E=1 inference (no expert lookup).
// One warp (32 threads) processes one row × 32 columns.
// Grid: (K/32, M_pad / WARPS_PER_BLOCK)
//
// Output: out_fp8 [M_pad, K] uint8 (unpacked FP8 E4M3)
//         sf_mma  [sf_size] uint8 (E8M0 in tcgen05 MMA layout)

static constexpr int WARPS_PER_BLOCK = 8;
static constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK;

__global__ void k_quant_fp8_e1(
    const __nv_bfloat16* __restrict__ x,    // [M_pad, K] input
    uint8_t* __restrict__ out_fp8,           // [M_pad, K] unpacked FP8
    uint8_t* __restrict__ sf_mma,            // [sf_size] swizzled E8M0
    int M_pad, int K, int sf_k)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Each warp handles one row × one SF group (32 columns)
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int m = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    const int k_sf = blockIdx.x;  // SF column index
    const int k_base = k_sf * 32;

    if (m >= M_pad) return;
    if (k_base + lane >= K) return;

    // Load one BF16 element per thread
    const int idx = m * K + k_base + lane;
    float v = __bfloat162float(x[idx]);

    // Warp-level reduction for amax across 32 elements
    float amax = fabsf(v);
    const unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(mask, amax, off));
    }

    // Lane 0 computes and stores E8M0 scale factor
    int scale_i = 0;
    if (lane == 0) {
        float scale = amax / FP8_MAX;
        if (!(scale > 0.0f)) scale = 1.0f;
        const uint8_t scale_byte = e8m0_encode_from_pos_f32(scale);
        // Write to tcgen05 swizzled position
        const int offset = sf_swizzle_offset(m, k_sf, sf_k);
        sf_mma[offset] = scale_byte;
        scale_i = static_cast<int>(scale_byte);
    }

    // Broadcast scale to all lanes
    scale_i = __shfl_sync(mask, scale_i, 0);
    const float inv_scale = e8m0_inv_decode_to_f32(static_cast<uint8_t>(scale_i));

    // Quantize to FP8 E4M3
    const uint8_t fp8_byte = f32_to_e4m3_byte(v * inv_scale);

    // Write unpacked FP8 byte
    out_fp8[idx] = fp8_byte;
#endif
}

// For single-expert (E=1), metadata is trivial:
//   sizes = (M_pad, N, K, 1)
//   ptrs = (A_ptr, B_ptr, C_ptr)
//   ptrs_sfasfb = (SFA_ptr, SFB_ptr)
//   strides = same for all "experts"
//
// This is a single-thread kernel (E=1 → 1 thread).

__global__ void k_build_metadata_e1(
    int64_t A_ptr, int64_t A_row_bytes,
    int64_t B_ptr, int64_t B_expert_bytes,
    int64_t C_ptr, int64_t C_row_bytes,
    int64_t SFA_ptr, int64_t SFA_row_bytes,
    int64_t SFB_ptr, int64_t SFB_expert_bytes,
    int32_t A_stride0, int32_t A_stride1,
    int32_t B_stride0, int32_t B_stride1,
    int32_t C_stride0, int32_t C_stride1,
    int32_t M_pad, int32_t N, int32_t K,
    int32_t* __restrict__ sizes_mnkl,
    int32_t* __restrict__ strides_abc,
    int64_t* __restrict__ ptrs_abc,
    int64_t* __restrict__ ptrs_sfasfb)
{
    // E=1, single expert
    sizes_mnkl[0] = M_pad;
    sizes_mnkl[1] = N;
    sizes_mnkl[2] = K;
    sizes_mnkl[3] = 1;

    strides_abc[0] = A_stride0;
    strides_abc[1] = A_stride1;
    strides_abc[2] = B_stride0;
    strides_abc[3] = B_stride1;
    strides_abc[4] = C_stride0;
    strides_abc[5] = C_stride1;

    ptrs_abc[0] = A_ptr;
    ptrs_abc[1] = B_ptr;
    ptrs_abc[2] = C_ptr;

    ptrs_sfasfb[0] = SFA_ptr;
    ptrs_sfasfb[1] = SFB_ptr;
}

// For cuBLAS _scaled_mm with per-row activation scaling.
// Single-pass: read input → register buffer → warp+block reduction for row amax
//   → scale + cast to FP8 → write output.
//
// Each thread block processes ROWS_PER_BLOCK rows.
// Each row uses BLOCK_DIM threads, where each thread handles K/BLOCK_DIM elements.
// K must be divisible by 128 (guaranteed by caller).
//
// Memory bandwidth: read 48MB + write 24MB + 32KB = 72MB for M=8192, K=3072
// At 1.8TB/s HBM3e → ~40us theoretical. Target: <80us.

static constexpr int RFP8_BLOCK_DIM = 256;   // threads per block
static constexpr int RFP8_ROWS_PER_BLOCK = 1; // one row per block for large K
static constexpr int RFP8_MAX_ELEMS_PER_THREAD = 48; // max K/BLOCK_DIM elements buffered

__global__ void k_rowwise_fp8_quant(
    const __nv_bfloat16* __restrict__ input,   // [M, K]
    uint8_t* __restrict__ output,              // [M, K] FP8 E4M3 as uint8
    float* __restrict__ scales,                // [M]
    int M, int K)
{
    const int row = blockIdx.x * RFP8_ROWS_PER_BLOCK + threadIdx.y;
    if (row >= M) return;

    const int tid = threadIdx.x;
    const int elems_per_thread = K / RFP8_BLOCK_DIM;
    const __nv_bfloat16* row_ptr = input + row * K;
    uint8_t* out_ptr = output + row * K;

    // Phase 1: Load elements into registers and compute thread-local abs max
    float local_buf[RFP8_MAX_ELEMS_PER_THREAD];
    float thread_max = 0.0f;

    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int col = tid + i * RFP8_BLOCK_DIM;
        float v = __bfloat162float(row_ptr[col]);
        local_buf[i] = v;
        float av = fabsf(v);
        thread_max = fmaxf(thread_max, av);
    }

    // Phase 2: Warp-level reduction
    const unsigned FULL_MASK = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(FULL_MASK, thread_max, off));
    }

    // Phase 3: Inter-warp reduction via shared memory
    __shared__ float warp_max[32];  // up to 32 warps
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = RFP8_BLOCK_DIM >> 5;

    if (lane == 0) {
        warp_max[warp_id] = thread_max;
    }
    __syncthreads();

    // Final reduction in first warp
    float row_amax;
    if (warp_id == 0) {
        float val = (lane < num_warps) ? warp_max[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            val = fmaxf(val, __shfl_xor_sync(FULL_MASK, val, off));
        }
        row_amax = val;
        if (lane == 0) {
            float scale = row_amax / FP8_MAX;
            scale = fmaxf(scale, 1e-12f);
            scales[row] = scale;
            // Store inverse scale for broadcast
            warp_max[0] = 1.0f / scale;
        }
    }
    __syncthreads();
    float inv_scale = warp_max[0];

    // Phase 4: Scale and convert to FP8, write output
    #pragma unroll 4
    for (int i = 0; i < elems_per_thread; i++) {
        int col = tid + i * RFP8_BLOCK_DIM;
        float scaled = local_buf[i] * inv_scale;
        // Use PTX for precise FP8 E4M3 conversion with saturation
        uint16_t fp8x2;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(fp8x2) : "f"(scaled));
        out_ptr[col] = static_cast<uint8_t>(fp8x2 & 0xFF);
    }
}

// Variant for small K (< 3072) — uses multiple rows per block
__global__ void k_rowwise_fp8_quant_small(
    const __nv_bfloat16* __restrict__ input,
    uint8_t* __restrict__ output,
    float* __restrict__ scales,
    int M, int K)
{
    // For small K, use one warp per row
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + warp_id;
    if (row >= M) return;

    const __nv_bfloat16* row_ptr = input + row * K;
    uint8_t* out_ptr = output + row * K;

    // Each lane handles K/32 elements (K guaranteed divisible by 128)
    const int elems_per_lane = K / 32;
    float local_buf[48];  // max 48 elements per lane (K up to 1536)
    float thread_max = 0.0f;

    for (int i = 0; i < elems_per_lane; i++) {
        int col = lane + i * 32;
        float v = __bfloat162float(row_ptr[col]);
        local_buf[i] = v;
        thread_max = fmaxf(thread_max, fabsf(v));
    }

    // Warp reduction
    const unsigned FULL_MASK = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        thread_max = fmaxf(thread_max, __shfl_xor_sync(FULL_MASK, thread_max, off));
    }

    float scale = thread_max / FP8_MAX;
    scale = fmaxf(scale, 1e-12f);
    if (lane == 0) scales[row] = scale;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < elems_per_lane; i++) {
        int col = lane + i * 32;
        float scaled = local_buf[i] * inv_scale;
        uint16_t fp8x2;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;" : "=h"(fp8x2) : "f"(scaled));
        out_ptr[col] = static_cast<uint8_t>(fp8x2 & 0xFF);
    }
}

cudaError_t launch_rowwise_fp8_quant(
    const __nv_bfloat16* input, uint8_t* output, float* scales,
    int M, int K, cudaStream_t stream)
{
    if (K >= 1536) {
        // Large K: one row per block, 256 threads
        // K/256 elements per thread (e.g., K=3072 → 12 elems/thread)
        dim3 grid(M);
        dim3 block(RFP8_BLOCK_DIM);
        k_rowwise_fp8_quant<<<grid, block, 0, stream>>>(
            input, output, scales, M, K);
    } else {
        // Small K: multiple rows per block with one warp per row
        int warps_per_block = 8;
        dim3 grid(ceil_div(M, warps_per_block));
        dim3 block(warps_per_block * 32);
        k_rowwise_fp8_quant_small<<<grid, block, 0, stream>>>(
            input, output, scales, M, K);
    }
    return cudaGetLastError();
}


cudaError_t launch_quant_fp8_e1(
    const __nv_bfloat16* x, uint8_t* out_fp8, uint8_t* sf_mma,
    int M_pad, int K, int sf_k, cudaStream_t stream)
{
    dim3 grid(sf_k, ceil_div(M_pad, WARPS_PER_BLOCK));
    dim3 block(THREADS_PER_BLOCK);
    k_quant_fp8_e1<<<grid, block, 0, stream>>>(x, out_fp8, sf_mma, M_pad, K, sf_k);
    return cudaGetLastError();
}

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
    cudaStream_t stream)
{
    k_build_metadata_e1<<<1, 1, 0, stream>>>(
        A_ptr, A_row_bytes, B_ptr, B_expert_bytes,
        C_ptr, C_row_bytes, SFA_ptr, SFA_row_bytes,
        SFB_ptr, SFB_expert_bytes,
        A_stride0, A_stride1, B_stride0, B_stride1,
        C_stride0, C_stride1,
        M_pad, N, K,
        sizes_mnkl, strides_abc, ptrs_abc, ptrs_sfasfb);
    return cudaGetLastError();
}

}  // namespace mxfp8
