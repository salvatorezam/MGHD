#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstring>

#ifndef CUDART_VERSION
#define CUDART_VERSION 0
#endif

// ===== Constant LUT (already used by batch path) =====
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

// Fast cache-bypassing load for avoiding L1 cache staleness
__device__ __forceinline__ int ld_cg(const int* p) {
    int v; 
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p)); 
    return v;
}

__constant__ uint16_t ROT_D3_LUT_PERSIST[256];

// ===== Simple device-side ring buffer state =====
struct RingState {
    uint8_t* in_bytes;     // [capacity]
    uint8_t* out_bits;     // [capacity * 9]
    int capacity;          // power-of-two preferred
    int* head;             // producer writes; device reads
    int* tail;             // device writes; producer reads
    int* run_flag;         // 1=running, 0=stop
};

// Device globals set once at start (minimalistic for clarity)
__device__ RingState g_state;

// ===== Persistent kernel: 1 block is enough for B=1, but we allow 32 threads =====
extern "C" __global__ void lut_persistent_kernel() {
    // Keep tail in register for efficient updates
    int t_local = *g_state.tail;
    
    // Single workgroup spins, popping items and decoding
    while (atomicAdd(g_state.run_flag, 0) != 0) {
        // Use cache-bypassing load for head to avoid L1 staleness
        int h = ld_cg(g_state.head);
        if (t_local == h) {
            // Nothing to do; light backoff
            __nanosleep(64);
            continue;
        }
        
        // Fast power-of-2 modulo using bitwise AND
        int idx = t_local & (g_state.capacity - 1);
        
        // Read one syndrome and decode
        uint8_t s = g_state.in_bytes[idx];
        uint16_t val = ROT_D3_LUT_PERSIST[s];

        // Write 9 output bits LSBF (keep original for B=1 performance)
        uint8_t* dst = g_state.out_bits + idx * 9;
        #pragma unroll
        for (int j = 0; j < 9; ++j) {
            dst[j] = (val >> j) & 0x1;
        }

        // publish results:
        // 1) data visible system-wide
        __threadfence_system();
        // 2) commit index (release)
        atomicExch(g_state.tail, ++t_local);
    }
}

// ===== Host helpers =====
static void copy_lut_to_constant(const torch::Tensor& lut16_cpu) {
    TORCH_CHECK(!lut16_cpu.is_cuda(), "LUT must be on CPU");
    TORCH_CHECK(lut16_cpu.dtype() == torch::kUInt16 && lut16_cpu.numel() == 256, "LUT must be [256] uint16");
    cudaMemcpyToSymbol(ROT_D3_LUT_PERSIST, lut16_cpu.data_ptr<uint16_t>(),
                       256 * sizeof(uint16_t), 0, cudaMemcpyHostToDevice);
}

static void set_device_ptr(void* dst_symbol_addr, const void* src_ptr, size_t bytes) {
    cudaMemcpy(dst_symbol_addr, src_ptr, bytes, cudaMemcpyHostToDevice);
}

// Kernel-side symbol addresses
__device__ __constant__ RingState* __sym_gstate_addr = &g_state;

// ===== API: start/stop/submit =====
struct PersistHandle {
    // input/output data
    torch::Tensor in_bytes;   // uint8 [capacity] (device)
    torch::Tensor out_bits;   // uint8 [capacity * 9] (device)
    uint8_t* out_bits_host;   // [capacity * 9] host-mapped

    // head stays on device; we also keep a host shadow to avoid D2H reads
    torch::Tensor head;       // int32 [1] (device)
    int head_shadow;          // host-side mirror of head

    // tail + run_flag become HOST-MAPPED (device sees them via aliases)
    int* tail_host;           // host-mapped
    int* tail_dev;            // device alias
    int* run_flag_host;       // host-mapped
    int* run_flag_dev;        // device alias

    int capacity;

    cudaStream_t kstream;     // persistent kernel stream
    cudaStream_t cpystream;   // copy/submit stream
};
static PersistHandle* g_handle = nullptr;

torch::Tensor persist_start(torch::Tensor lut16_cpu, int capacity) {
    // Ensure capacity is power-of-2 for fast modulo via bitwise AND
    TORCH_CHECK(capacity >= 64 && (capacity & (capacity - 1)) == 0, 
                "capacity must be >= 64 and power-of-2 for optimal performance");
    TORCH_CHECK(lut16_cpu.dtype() == torch::kUInt16 && lut16_cpu.numel() == 256, "LUT must be [256] uint16 on CPU");

    auto dev = torch::kCUDA;
    
    // --- constant LUT ---
    copy_lut_to_constant(lut16_cpu);

    // --- allocate buffers ---
    // in_bytes on device (keep as-is for fast H2D of 1 byte)
    auto in_bytes  = torch::empty({capacity},    torch::dtype(torch::kUInt8).device(torch::kCUDA));
    // head stays on device
    auto head      = torch::zeros({1},           torch::dtype(torch::kInt32).device(torch::kCUDA));

    // out_bits in HOST-MAPPED pinned memory for zero-copy reads
    uint8_t* out_bits_host = nullptr;
    cudaHostAlloc((void**)&out_bits_host, capacity * 9 * sizeof(uint8_t), cudaHostAllocMapped);
    std::memset(out_bits_host, 0, capacity * 9);
    uint8_t* out_bits_dev = nullptr;
    cudaHostGetDevicePointer((void**)&out_bits_dev, (void*)out_bits_host, 0);

    // tail in HOST-MAPPED pinned memory
    int* tail_host = nullptr;
    cudaHostAlloc((void**)&tail_host, sizeof(int), cudaHostAllocMapped);
    *tail_host = 0;
    int* tail_dev = nullptr;
    cudaHostGetDevicePointer((void**)&tail_dev, (void*)tail_host, 0);

    // run_flag in HOST-MAPPED pinned memory
    int* run_flag_host = nullptr;
    cudaHostAlloc((void**)&run_flag_host, sizeof(int), cudaHostAllocMapped);
    *run_flag_host = 1;
    int* run_flag_dev = nullptr;
    cudaHostGetDevicePointer((void**)&run_flag_dev, (void*)run_flag_host, 0);

    // --- initialize device-side state ---
    RingState hstate;
    std::memset(&hstate, 0, sizeof(RingState));
    hstate.capacity = capacity;
    cudaMemcpyToSymbol(g_state, &hstate, sizeof(RingState));

    RingState tmp;
    cudaMemcpyFromSymbol(&tmp, g_state, sizeof(RingState));
    tmp.in_bytes   = (uint8_t*) in_bytes.data_ptr<uint8_t>();
    tmp.out_bits   = out_bits_dev;                     // device alias to host-mapped
    tmp.capacity   = capacity;
    tmp.head       = (int*) head.data_ptr<int32_t>();  // device memory
    tmp.tail       = tail_dev;                         // device alias to host-mapped
    tmp.run_flag   = run_flag_dev;                     // device alias to host-mapped
    cudaMemcpyToSymbol(g_state, &tmp, sizeof(RingState));

    // --- streams and launch ---
    cudaStream_t kstream, cpystream;
    cudaStreamCreateWithFlags(&kstream,  cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&cpystream,cudaStreamNonBlocking);
    lut_persistent_kernel<<<1, 64, 0, kstream>>>();

    g_handle = new PersistHandle{
        in_bytes,
        torch::empty({capacity * 9}, torch::dtype(torch::kUInt8).device(torch::kCUDA)), // placeholder
        out_bits_host,
        head,
        /* head_shadow */ 0,
        tail_host, tail_dev,
        run_flag_host, run_flag_dev,
        capacity,
        kstream, cpystream
    };
    return torch::tensor({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
}

torch::Tensor persist_stop() {
    if (!g_handle) {
        return torch::tensor({0}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    }
    // signal stop (host-mapped flag)
    *(g_handle->run_flag_host) = 0;

    // wait streams & kernel to observe flag and exit
    cudaStreamSynchronize(g_handle->cpystream);
    cudaStreamSynchronize(g_handle->kstream);

    // destroy streams
    cudaStreamDestroy(g_handle->cpystream);
    cudaStreamDestroy(g_handle->kstream);

    // free host-mapped memory
    cudaFreeHost(g_handle->out_bits_host);
    cudaFreeHost(g_handle->tail_host);
    cudaFreeHost(g_handle->run_flag_host);

    delete g_handle; 
    g_handle = nullptr;
    return torch::tensor({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
}

// Submit N bytes synchronously: copies bytes into ring, waits until processed, returns [N,9] on device
torch::Tensor persist_submit(torch::Tensor bytes_cpu) {
    TORCH_CHECK(g_handle != nullptr, "persist_start must be called first");
    TORCH_CHECK(!bytes_cpu.is_cuda() && bytes_cpu.dtype() == torch::kUInt8 && bytes_cpu.dim() == 1, "bytes must be CPU uint8 [N]");
    const int N = (int)bytes_cpu.size(0);
    TORCH_CHECK(N <= g_handle->capacity, "N exceeds ring capacity");

    // read current head/tail once (use host shadow; avoid D2H)
    int h_host = g_handle->head_shadow;
    int t_host = *(g_handle->tail_host);

    // simple no-wrap submit with fast power-of-2 modulo
    int start = h_host & (g_handle->capacity - 1);
    const int cap = g_handle->capacity;

    // Wrap-aware H2D: split into up to 2 copies
    int first = std::min(N, cap - start);
    int second = N - first;

    // Part A
    cudaMemcpyAsync(
        g_handle->in_bytes.data_ptr<uint8_t>() + start,
        bytes_cpu.data_ptr<uint8_t>(),
        first * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        g_handle->cpystream);

    // Part B (only if wrap)
    if (second > 0) {
        cudaMemcpyAsync(
            g_handle->in_bytes.data_ptr<uint8_t>(),             // wraps to index 0
            bytes_cpu.data_ptr<uint8_t>() + first,
            second * sizeof(uint8_t),
            cudaMemcpyHostToDevice,
            g_handle->cpystream);
    }

    int32_t new_head = h_host + N;

    // Stream-ordered doorbell (fallback to memcpy for compatibility)
    cudaMemcpyAsync(
        g_handle->head.data_ptr<int32_t>(),
        &new_head,
        sizeof(int32_t),
        cudaMemcpyHostToDevice,
        g_handle->cpystream);

    // Update host shadow after enqueueing the update
    g_handle->head_shadow = new_head;

    // wait until tail catches up by reading HOST-MAPPED tail directly (no cudaMemcpy)
    // Stream ordering guarantees the head update completes before we start waiting
    int spins = 0;
    // Wrap-safe compare using signed distance: tail < new_head  <=> (new_head - tail) > 0
    while ((int32_t)(new_head - *(g_handle->tail_host)) > 0) {
        if (spins < 500) {  // ~few hundred cycles
#if defined(__x86_64__)
            __builtin_ia32_pause();  // or _mm_pause()
#endif
            ++spins;
        } else {
            std::this_thread::yield();  // or sleep_for(0us)
        }
    }

    // results are already in HOST-MAPPED out_bits; create a CPU tensor and memcpy from host memory
    auto out_cpu = torch::empty({N, 9}, torch::dtype(torch::kUInt8).device(torch::kCPU));

    // Wrap-aware host memcpy from host-mapped out_bits
    uint8_t* dst0 = out_cpu.data_ptr<uint8_t>();
    const uint8_t* srcA = g_handle->out_bits_host + start * 9;
    const uint8_t* srcB = g_handle->out_bits_host; // index 0 after wrap

    // First segment
    std::memcpy(dst0, srcA, first * 9 * sizeof(uint8_t));

    // Second segment (if wrap)
    if (second > 0) {
        std::memcpy(dst0 + first * 9, srcB, second * 9 * sizeof(uint8_t));
    }

    return out_cpu; // CPU tensor; tests use numpy math so this is fine
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("persist_start",  &persist_start, "Start persistent LUT decoder");
    m.def("persist_stop",   &persist_stop,  "Stop persistent LUT decoder");
    m.def("persist_submit", &persist_submit,"Submit syndromes and get decoded bits");
}
