#include "../fastpath/c_api/fastpath_capi.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

int main() {
    printf("Fastpath C API HAL Demo\n");
    printf("=======================\n");
    
    // Initialize with default LUT and 1024 capacity
    printf("Initializing fastpath decoder (capacity=1024)...\n");
    int result = fastpath_init(NULL, 1024);
    if (result != 0) {
        fprintf(stderr, "❌ fastpath_init failed with error %d\n", result);
        return 1;
    }
    printf("✅ Fastpath decoder initialized\n");
    
    // Generate random syndrome data
    srand(42); // Reproducible demo
    const int N = 64;
    uint8_t syndromes[N];
    
    printf("\nGenerating %d random syndromes...\n", N);
    for (int i = 0; i < N; i++) {
        syndromes[i] = (uint8_t)(rand() % 256);
    }
    
    // Allocate output buffer
    uint8_t corrections[N * 9];
    
    // Decode syndromes
    printf("Decoding %d syndromes...\n", N);
    clock_t start = clock();
    int decoded = fastpath_decode(syndromes, N, corrections);
    clock_t end = clock();
    
    if (decoded != N) {
        fprintf(stderr, "❌ Decode failed: expected %d, got %d\n", N, decoded);
        fastpath_stop();
        return 2;
    }
    
    double decode_time_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
    double latency_us = (decode_time_ms * 1000.0) / N;
    
    printf("✅ Successfully decoded %d syndromes\n", decoded);
    printf("⏱️  Total time: %.2f ms\n", decode_time_ms);
    printf("⏱️  Avg latency: %.1f µs per syndrome\n", latency_us);
    
    // Print first 4 corrections
    printf("\nFirst 4 syndrome→correction mappings:\n");
    for (int i = 0; i < 4; i++) {
        printf("syndrome[%d] = %3u → correction = ", i, (unsigned)syndromes[i]);
        for (int j = 0; j < 9; j++) {
            printf("%u", (unsigned)corrections[i * 9 + j]);
        }
        printf("\n");
    }
    
    // Verify output format
    printf("\nValidating output format...\n");
    bool format_ok = true;
    for (int i = 0; i < N * 9; i++) {
        if (corrections[i] != 0 && corrections[i] != 1) {
            printf("❌ Invalid correction bit at position %d: %u\n", i, corrections[i]);
            format_ok = false;
            break;
        }
    }
    
    if (format_ok) {
        printf("✅ All correction bits are valid (0 or 1)\n");
    }
    
    // Clean shutdown
    printf("\nShutting down fastpath decoder...\n");
    fastpath_stop();
    printf("✅ Fastpath decoder stopped\n");
    
    printf("\n🎉 HAL demo completed successfully!\n");
    printf("📊 Performance: %.1f µs/syndrome on average\n", latency_us);
    
    return 0;
}
