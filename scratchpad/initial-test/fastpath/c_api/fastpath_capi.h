#ifndef FASTPATH_CAPI_H
#define FASTPATH_CAPI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the fastpath decoder with LUT and ring buffer capacity.
 * 
 * @param lut16 Pointer to 256-entry uint16 LUT (can be NULL to auto-load rotated d=3)
 * @param capacity Ring buffer capacity (must be power of 2, >= 64)
 * @return 0 on success, negative error code on failure
 */
int fastpath_init(const uint16_t* lut16, int capacity);

/**
 * Blocking decode: decode N syndromes and return corrections.
 * 
 * @param synd Input syndromes [N] each 8-bit LSB-first syndrome
 * @param N Number of syndromes to decode
 * @param out9xN Output buffer [N*9] for correction bits (row-major, one byte per bit)
 * @return Number of successfully decoded syndromes on success, negative on error
 */
int fastpath_decode(const uint8_t* synd, int N, uint8_t* out9xN);

/**
 * Clean shutdown of fastpath decoder. Safe to call multiple times.
 */
void fastpath_stop(void);

/**
 * Nonblocking submit: queue N syndromes for decoding.
 * 
 * @param synd Input syndromes [N] each 8-bit LSB-first syndrome  
 * @param N Number of syndromes to submit
 * @return 0 on success, negative on error (queue full, etc.)
 */
int fastpath_submit(const uint8_t* synd, int N);

/**
 * Nonblocking poll: retrieve up to maxN decoded corrections.
 * 
 * @param out9xN Output buffer [maxN*9] for correction bits
 * @param maxN Maximum number of corrections to retrieve
 * @return Number of corrections retrieved (0 if none ready), negative on error
 */
int fastpath_poll(uint8_t* out9xN, int maxN);

#ifdef __cplusplus
}
#endif

#endif // FASTPATH_CAPI_H
