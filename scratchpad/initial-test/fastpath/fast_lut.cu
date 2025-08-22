#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

__constant__ uint16_t ROT_D3_LUT[256]; // 9-bit answer in LSBs

// Kernel: one thread per sample, reading [B] byte syndromes -> writing [B,9] uint8
__global__ void lut_decode_kernel(const uint8_t* __restrict__ synd_bytes,
                                  uint8_t* __restrict__ out_bits,
                                  int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;
    uint8_t s = synd_bytes[idx]; // LSB-first syndrome byte
    uint16_t val = ROT_D3_LUT[s];

    // Write 9 bits to out_bits[idx, :]
    uint8_t* dst = out_bits + idx * 9;
    #pragma unroll
    for (int j = 0; j < 9; ++j) {
        dst[j] = (val >> j) & 0x1;
    }
}

static void load_constant_lut(torch::Tensor lut16) {
    TORCH_CHECK(lut16.device().is_cpu(), "LUT must be a CPU tensor");
    TORCH_CHECK(lut16.dtype() == torch::kUInt16, "LUT must be uint16");
    TORCH_CHECK(lut16.numel() == 256, "LUT must have 256 entries");
    cudaMemcpyToSymbol(ROT_D3_LUT, lut16.data_ptr<uint16_t>(),
                       256 * sizeof(uint16_t), 0, cudaMemcpyHostToDevice);
}

torch::Tensor fast_decode(torch::Tensor synd_bytes, torch::Tensor lut16) {
    TORCH_CHECK(synd_bytes.is_cuda(), "synd_bytes must be CUDA uint8 tensor [B]");
    TORCH_CHECK(synd_bytes.dtype() == torch::kUInt8 && synd_bytes.dim() == 1,
                "synd_bytes must be [B] uint8");
    TORCH_CHECK(lut16.dtype() == torch::kUInt16 && lut16.numel() == 256,
                "lut16 must be [256] uint16 on CPU");

    // Load LUT into constant memory (idempotent if same buffer)
    load_constant_lut(lut16);

    int B = synd_bytes.size(0);
    auto out = torch::empty({B, 9}, torch::dtype(torch::kUInt8).device(synd_bytes.device()));

    dim3 block(256);
    dim3 grid((B + block.x - 1) / block.x);
    lut_decode_kernel<<<grid, block>>>(synd_bytes.data_ptr<uint8_t>(),
                                       out.data_ptr<uint8_t>(),
                                       B);
    cudaDeviceSynchronize(); // simple correctness; we will remove in persistent/graph step
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_decode", &fast_decode, "Rotated d=3 LUT fast decode");
}
