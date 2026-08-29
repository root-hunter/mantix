#ifndef MANTIX_LIMB_H
#define MANTIX_LIMB_H

#include "mantix/mantix.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum mtx_cpu_feature {
    MTX_CPU_FEATURE_NONE = 0U,
    MTX_CPU_FEATURE_BMI2 = 1U << 0U,
    MTX_CPU_FEATURE_ADX = 1U << 1U,
    MTX_CPU_FEATURE_AVX2 = 1U << 2U
} mtx_cpu_feature;

/* Runtime CPU feature mask. Unsupported architectures return NONE. */
uint32_t mtx_cpu_features(void);

/* Name of the limb backend selected for this build. */
const char *mtx_limb_backend(void);

/*
 * Add count little-endian limbs and return the carry (zero or one).
 * result may equal left or right. Other partial overlaps are not supported.
 */
mtx_limb mtx_limb_add_n(mtx_limb *result, const mtx_limb *left,
                        const mtx_limb *right, size_t count);

/*
 * Subtract right from left and return the borrow (zero or one).
 * result may equal left or right. Other partial overlaps are not supported.
 */
mtx_limb mtx_limb_sub_n(mtx_limb *result, const mtx_limb *left,
                        const mtx_limb *right, size_t count);

/*
 * Multiply two count-limb, little-endian integers. The complete 2 * count
 * limb product is written to result. Unlike add_n/sub_n, result must not
 * overlap either input. A zero count performs no memory access.
 */
void mtx_limb_mul_n(mtx_limb *result, const mtx_limb *left,
                    const mtx_limb *right, size_t count);

/*
 * Shift count limbs left by shift bits (0 <= shift < 64).
 * Returns the carry out from the most significant limb.
 */
mtx_limb mtx_limb_lshift(mtx_limb *result, const mtx_limb *src,
                         size_t count, unsigned shift);

/*
 * Shift count limbs right by shift bits (0 <= shift < 64).
 * Returns the bits shifted out from the least significant limb.
 */
mtx_limb mtx_limb_rshift(mtx_limb *result, const mtx_limb *src,
                         size_t count, unsigned shift);

/*
 * Compare two count-limb integers: returns -1 if left < right, 0 if equal, +1 if left > right.
 */
int mtx_limb_cmp_n(const mtx_limb *left, const mtx_limb *right, size_t count);

#ifdef __cplusplus
}
#endif

#endif
