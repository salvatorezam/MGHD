#include "fastpath_capi.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

// Default rotated d=3 LUT (256 entries)
const uint16_t ROTATED_D3_LUT_256[256] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    256, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 4, 0, 0, 0, 0, 0, 0,
    0, 32, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 64, 0, 0, 0, 8, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    128, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 16, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
};

// Device-side ring buffer state structure
struct RingState {
    uint8_t* in_bytes;     // [capacity]
    uint8_t* out_bits;     // [capacity * 9] 
    int capacity;          // power-of-two
    int* head;             // producer writes; device reads
    int* tail;             // device writes; producer reads
    int* run_flag;         // 1=running, 0=stop
};

// CUDA kernel and setup functions
extern "C" {
    // Function to copy LUT to device constant memory
    cudaError_t copy_lut_to_device(const uint16_t* lut_data);
    
    // Function to setup device state
    cudaError_t setup_device_ring_state(const RingState* state);
    
    // Function to launch persistent kernel
    cudaError_t launch_persistent_kernel(cudaStream_t stream);
}

// Global state for the C API
static struct FastpathCState {
    // Device buffers
    uint8_t* d_in_bytes;      // device input ring [capacity]
    uint8_t* d_out_bits_dev;  // device pointer to host-mapped output [capacity*9]
    int* d_head;              // device head pointer
    int* d_tail_dev;          // device pointer to host-mapped tail
    int* d_run_flag_dev;      // device pointer to host-mapped run flag
    
    // Host-mapped memory for zero-copy
    uint8_t* h_out_bits_host; // host-mapped output [capacity*9]
    int* h_tail_host;         // host-mapped tail
    int* h_run_flag_host;     // host-mapped run flag
    
    // Ring buffer parameters
    uint32_t capacity;
    uint32_t capacity_mask;
    
    // Head shadow for performance (avoid D2H reads)
    uint32_t head_shadow;
    
    // CUDA streams
    cudaStream_t k_stream;    // kernel stream
    cudaStream_t copy_stream; // copy stream
    
    // State tracking
    bool initialized;
    
    FastpathCState() : d_in_bytes(nullptr), d_out_bits_dev(nullptr), d_head(nullptr),
                      d_tail_dev(nullptr), d_run_flag_dev(nullptr), h_out_bits_host(nullptr),
                      h_tail_host(nullptr), h_run_flag_host(nullptr), capacity(0),
                      capacity_mask(0), head_shadow(0), k_stream(0), copy_stream(0),
                      initialized(false) {}
} g_cstate;

// Helper function to check if value is power of 2
static bool is_power_of_2(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

// Error handling - catch all exceptions and return error codes
template<typename F>
static int safe_call(F&& func) {
    try {
        return func();
    } catch (...) {
        return -1; // Generic error
    }
}

extern "C" int fastpath_init(const uint16_t* lut16, int capacity) {
    return safe_call([=]() -> int {
        if (g_cstate.initialized) {
            return -2; // Already initialized
        }
        
        if (capacity < 64 || !is_power_of_2(capacity)) {
            return -3; // Invalid capacity
        }
        
        g_cstate.capacity = capacity;
        g_cstate.capacity_mask = capacity - 1;
        
        // Create CUDA streams
        if (cudaStreamCreateWithFlags(&g_cstate.k_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&g_cstate.copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
            return -4; // Stream creation failed
        }
        
        // Copy LUT to constant memory
        const uint16_t* lut_data = lut16 ? lut16 : ROTATED_D3_LUT_256;
        if (copy_lut_to_device(lut_data) != cudaSuccess) {
            return -5; // LUT copy failed
        }
        
        // Allocate device memory for input ring
        if (cudaMalloc(&g_cstate.d_in_bytes, capacity * sizeof(uint8_t)) != cudaSuccess) {
            return -6; // Input allocation failed
        }
        
        // Allocate device memory for head
        if (cudaMalloc(&g_cstate.d_head, sizeof(int)) != cudaSuccess) {
            return -7; // Head allocation failed
        }
        
        // Allocate host-mapped memory for output ring (zero-copy)
        if (cudaHostAlloc(&g_cstate.h_out_bits_host, capacity * 9 * sizeof(uint8_t), 
                         cudaHostAllocMapped) != cudaSuccess) {
            return -8; // Output allocation failed
        }
        std::memset(g_cstate.h_out_bits_host, 0, capacity * 9);
        
        // Get device pointer for host-mapped output
        if (cudaHostGetDevicePointer((void**)&g_cstate.d_out_bits_dev, 
                                    g_cstate.h_out_bits_host, 0) != cudaSuccess) {
            return -9; // Device pointer failed
        }
        
        // Allocate host-mapped memory for tail
        if (cudaHostAlloc(&g_cstate.h_tail_host, sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
            return -10; // Tail allocation failed
        }
        *g_cstate.h_tail_host = 0;
        
        // Get device pointer for host-mapped tail
        if (cudaHostGetDevicePointer((void**)&g_cstate.d_tail_dev, 
                                    g_cstate.h_tail_host, 0) != cudaSuccess) {
            return -11; // Tail device pointer failed
        }
        
        // Allocate host-mapped memory for run flag
        if (cudaHostAlloc(&g_cstate.h_run_flag_host, sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
            return -12; // Run flag allocation failed
        }
        *g_cstate.h_run_flag_host = 1;
        
        // Get device pointer for host-mapped run flag
        if (cudaHostGetDevicePointer((void**)&g_cstate.d_run_flag_dev, 
                                    g_cstate.h_run_flag_host, 0) != cudaSuccess) {
            return -13; // Run flag device pointer failed
        }
        
        // Initialize device head to 0
        int zero = 0;
        if (cudaMemcpy(g_cstate.d_head, &zero, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
            return -14; // Head initialization failed
        }
        
        g_cstate.head_shadow = 0;
        
        // Set up device-side global state
        RingState ring_state;
        ring_state.in_bytes = g_cstate.d_in_bytes;
        ring_state.out_bits = g_cstate.d_out_bits_dev;
        ring_state.capacity = g_cstate.capacity;
        ring_state.head = g_cstate.d_head;
        ring_state.tail = g_cstate.d_tail_dev;
        ring_state.run_flag = g_cstate.d_run_flag_dev;
        
        // Setup device global state
        if (setup_device_ring_state(&ring_state) != cudaSuccess) {
            return -15; // State setup failed
        }
        
        // Launch persistent kernel
        if (launch_persistent_kernel(g_cstate.k_stream) != cudaSuccess) {
            return -16; // Kernel launch failed
        }
        
        g_cstate.initialized = true;
        return 0;
    });
}

extern "C" int fastpath_decode(const uint8_t* synd, int N, uint8_t* out9xN) {
    return safe_call([=]() -> int {
        if (!g_cstate.initialized) {
            return -1; // Not initialized
        }
        
        if (!synd || !out9xN || N <= 0) {
            return -2; // Invalid arguments
        }
        
        if (N > (int)g_cstate.capacity) {
            return -3; // Batch too large
        }
        
        uint32_t start_head = g_cstate.head_shadow;
        uint32_t end_head = start_head + N;
        
        // Check if we have space (conservative check)
        uint32_t tail = *g_cstate.h_tail_host;
        int32_t available = (int32_t)(tail + g_cstate.capacity - start_head);
        if (available < N) {
            return -4; // Ring buffer full
        }
        
        // Handle wrap-around for input
        uint32_t start_idx = start_head & g_cstate.capacity_mask;
        
        if (start_idx + N <= g_cstate.capacity) {
            // No wrap - single copy
            if (cudaMemcpyAsync(g_cstate.d_in_bytes + start_idx, synd, N,
                               cudaMemcpyHostToDevice, g_cstate.copy_stream) != cudaSuccess) {
                return -5; // Copy failed
            }
        } else {
            // Wrap-around - split copy
            uint32_t first_chunk = g_cstate.capacity - start_idx;
            uint32_t second_chunk = N - first_chunk;
            
            if (cudaMemcpyAsync(g_cstate.d_in_bytes + start_idx, synd, first_chunk,
                               cudaMemcpyHostToDevice, g_cstate.copy_stream) != cudaSuccess ||
                cudaMemcpyAsync(g_cstate.d_in_bytes, synd + first_chunk, second_chunk,
                               cudaMemcpyHostToDevice, g_cstate.copy_stream) != cudaSuccess) {
                return -5; // Copy failed
            }
        }
        
        // Stream-ordered doorbell update (head shadow)
        g_cstate.head_shadow = end_head;
        if (cudaMemcpyAsync(g_cstate.d_head, &end_head, sizeof(uint32_t),
                           cudaMemcpyHostToDevice, g_cstate.copy_stream) != cudaSuccess) {
            return -6; // Doorbell update failed
        }
        
        // Wait for results with wrap-safe comparison
        int spins = 0;
        while (true) {
            uint32_t current_tail = *g_cstate.h_tail_host;
            int32_t diff = (int32_t)(current_tail - end_head);
            if (diff >= 0) break;
            
            // Spin with backoff
            if (spins < 500) {
#if defined(__x86_64__)
                __builtin_ia32_pause();
#endif
                ++spins;
            } else {
                std::this_thread::yield();
            }
        }
        
        // Copy results from host-mapped memory (handle wrap-around)
        if (start_idx + N <= g_cstate.capacity) {
            // No wrap - single copy
            std::memcpy(out9xN, g_cstate.h_out_bits_host + start_idx * 9, N * 9);
        } else {
            // Wrap-around - split copy
            uint32_t first_chunk = g_cstate.capacity - start_idx;
            uint32_t second_chunk = N - first_chunk;
            
            std::memcpy(out9xN, g_cstate.h_out_bits_host + start_idx * 9, first_chunk * 9);
            std::memcpy(out9xN + first_chunk * 9, g_cstate.h_out_bits_host, second_chunk * 9);
        }
        
        return N; // Success
    });
}

extern "C" void fastpath_stop(void) {
    if (!g_cstate.initialized) return;
    
    try {
        // Signal kernel to stop
        if (g_cstate.h_run_flag_host) {
            *g_cstate.h_run_flag_host = 0;
        }
        
        // Synchronize streams
        if (g_cstate.k_stream) {
            cudaStreamSynchronize(g_cstate.k_stream);
            cudaStreamDestroy(g_cstate.k_stream);
        }
        if (g_cstate.copy_stream) {
            cudaStreamSynchronize(g_cstate.copy_stream);
            cudaStreamDestroy(g_cstate.copy_stream);
        }
        
        // Free device memory
        if (g_cstate.d_in_bytes) cudaFree(g_cstate.d_in_bytes);
        if (g_cstate.d_head) cudaFree(g_cstate.d_head);
        
        // Free host-mapped memory
        if (g_cstate.h_out_bits_host) cudaFreeHost(g_cstate.h_out_bits_host);
        if (g_cstate.h_tail_host) cudaFreeHost(g_cstate.h_tail_host);
        if (g_cstate.h_run_flag_host) cudaFreeHost(g_cstate.h_run_flag_host);
        
        // Reset state
        memset(&g_cstate, 0, sizeof(g_cstate));
    } catch (...) {
        // Ignore exceptions during cleanup
    }
}

extern "C" int fastpath_submit(const uint8_t* synd, int N) {
    // For now, just use blocking decode - can optimize later
    static thread_local std::vector<uint8_t> temp_buffer;
    temp_buffer.resize(N * 9);
    
    int result = fastpath_decode(synd, N, temp_buffer.data());
    return (result == N) ? 0 : result;
}

extern "C" int fastpath_poll(uint8_t* out9xN, int maxN) {
    // Placeholder - would need separate queue for true nonblocking
    return 0; // No results ready
}
