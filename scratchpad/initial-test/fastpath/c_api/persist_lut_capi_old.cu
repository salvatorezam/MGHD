#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cstring>

// Fast cache-bypassing load for avoiding L1 cache staleness
__device__ __forceinline__ int ld_cg(const int* p) {
    int v; 
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(p)); 
    return v;
}

__constant__ uint16_t ROT_D3_LUT_PERSIST[256];

// Device-side ring buffer state
struct RingState {
    uint8_t* in_bytes;     // [capacity]
    uint8_t* out_bits;     // [capacity * 9]
    int capacity;          // power-of-two preferred
    int* head;             // producer writes; device reads
    int* tail;             // device writes; producer reads
    int* run_flag;         // 1=running, 0=stop
};

// Device global state
__device__ RingState g_state;

// Persistent kernel: 1 block with 64 threads
extern "C" __global__ void lut_persistent_kernel() {
    // Keep tail in register for efficient updates
    int t_local = *g_state.tail;
    
    // Single workgroup spins, popping items and decoding
    while (atomicAdd(g_state.run_flag, 0) != 0) {
        // Use cache-bypassing load for head to avoid L1 staleness
        int h = ld_cg(g_state.head);
        if (t_local == h) {
            // Nothing to do; light backoff (using standard CUDA sleep)
            __nanosleep(64U);
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
