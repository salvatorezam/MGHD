#include <cuda_runtime.h>
#include <cstdint>

// Device-side constant LUT in constant memory (faster than global)
__constant__ uint16_t ROT_D3_LUT_PERSIST[256];

// Device-side ring buffer state structure
struct RingState {
    uint8_t* in_bytes;     // [capacity]
    uint8_t* out_bits;     // [capacity * 9] 
    int capacity;          // power-of-two
    int* head;             // producer writes; device reads
    int* tail;             // device writes; producer reads
    int* run_flag;         // 1=running, 0=stop
};

// Device global state (will be set by host)
__device__ RingState g_state;

// Fast cache-bypassing load for avoiding L1 cache staleness
__device__ __forceinline__ int ld_cg(const int* p) {
    int v; 
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p)); 
    return v;
}

// Persistent kernel that runs forever until stop signal
__global__ void lut_persistent_kernel() {
    // Keep tail in register for efficient updates
    int t_local = *g_state.tail;
    
    // Single workgroup spins, popping items and decoding
    while (atomicAdd(g_state.run_flag, 0) != 0) {
        // Use cache-bypassing load for head to avoid L1 staleness
        int h = ld_cg(g_state.head);
        if (t_local == h) {
            // Nothing to do; light backoff (using CUDA clock for brief delay)
            clock_t start = clock();
            while (clock() - start < 100) { /* brief delay */ }
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

// Host-callable functions to interface with the kernel

extern "C" {

// Function to copy LUT to device constant memory
cudaError_t copy_lut_to_device(const uint16_t* lut_data) {
    return cudaMemcpyToSymbol(ROT_D3_LUT_PERSIST, lut_data, 
                              256 * sizeof(uint16_t), 0, cudaMemcpyHostToDevice);
}

// Function to setup device state
cudaError_t setup_device_ring_state(const RingState* state) {
    return cudaMemcpyToSymbol(g_state, state, sizeof(RingState), 0, cudaMemcpyHostToDevice);
}

// Function to launch persistent kernel
cudaError_t launch_persistent_kernel(cudaStream_t stream) {
    lut_persistent_kernel<<<1, 64, 0, stream>>>();
    return cudaGetLastError();
}

}
