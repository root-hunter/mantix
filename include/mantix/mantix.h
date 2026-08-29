#ifndef MANTIX_MANTIX_H
#define MANTIX_MANTIX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MANTIX_VERSION_MAJOR 0
#define MANTIX_VERSION_MINOR 1
#define MANTIX_VERSION_PATCH 0
#define MANTIX_VERSION_STRING "0.1.0"

/*
 * A finite Mantix number represents:
 *
 *   (-1)^negative * significand * 2^exponent
 *
 * Limbs are stored least-significant first. A non-zero value is normalized:
 * its most-significant used limb is non-zero and its significand is odd.
 */
typedef uint64_t mtx_limb;

typedef enum mtx_status {
    MTX_OK = 0,
    MTX_ERROR_INVALID_ARGUMENT,
    MTX_ERROR_OUT_OF_MEMORY,
    MTX_ERROR_EXPONENT_OVERFLOW
} mtx_status;

typedef enum mtx_rounding {
    MTX_ROUND_TO_NEAREST_EVEN = 0,
    MTX_ROUND_TOWARD_ZERO,
    MTX_ROUND_TOWARD_POSITIVE,
    MTX_ROUND_TOWARD_NEGATIVE
} mtx_rounding;

/*
 * An initialized value owns its storage and must stay at a stable address.
 * Do not copy or move it with struct assignment; use mtx_set instead.
 */
typedef struct mtx_float {
    mtx_limb *limbs;
    size_t used;
    size_t capacity;
    size_t precision;
    int64_t exponent;
    bool negative;
    mtx_limb inline_limb;
} mtx_float;

/* Initialize x with a target precision in bits. precision must be non-zero. */
mtx_status mtx_init(mtx_float *x, size_t precision);

/* Release owned storage. It is safe to clear an initialized value repeatedly. */
void mtx_clear(mtx_float *x);

/* Set x to canonical positive zero without changing its target precision. */
void mtx_set_zero(mtx_float *x);

/* Set x from an unsigned machine integer. */
mtx_status mtx_set_u64(mtx_float *x, uint64_t value);

/* Set x from a signed machine integer. */
mtx_status mtx_set_i64(mtx_float *x, int64_t value);

/* Set x from a 32-bit IEEE 754 single-precision float. */
mtx_status mtx_set_f32(mtx_float *x, float value);

/* Extract a 32-bit IEEE 754 single-precision float with rounding. */
float mtx_get_f32(const mtx_float *x, mtx_rounding rnd);

/* Set x from a 64-bit IEEE 754 double-precision float. */
mtx_status mtx_set_f64(mtx_float *x, double value);

/* Extract a 64-bit IEEE 754 double-precision float with rounding. */
double mtx_get_f64(const mtx_float *x, mtx_rounding rnd);

/* Aliases for double-precision */
mtx_status mtx_set_d(mtx_float *x, double value);
double mtx_get_d(const mtx_float *x, mtx_rounding rnd);

/* Copy src into dst, preserving dst's target precision. */
mtx_status mtx_set(mtx_float *dst, const mtx_float *src);

/* Negate a: r = -a */
mtx_status mtx_neg(mtx_float *r, const mtx_float *a);

/* Absolute value of a: r = |a| */
mtx_status mtx_abs(mtx_float *r, const mtx_float *a);

/* Compare a and b: returns -1 if a < b, 0 if a == b, +1 if a > b. */
int mtx_cmp(const mtx_float *a, const mtx_float *b);

/* Floating-point addition: r = a + b with rounding. */
mtx_status mtx_add(mtx_float *r, const mtx_float *a, const mtx_float *b,
                   mtx_rounding rnd);

/* Floating-point subtraction: r = a - b with rounding. */
mtx_status mtx_sub(mtx_float *r, const mtx_float *a, const mtx_float *b,
                   mtx_rounding rnd);

/* Floating-point multiplication: r = a * b with rounding. */
mtx_status mtx_mul(mtx_float *r, const mtx_float *a, const mtx_float *b,
                   mtx_rounding rnd);

bool mtx_is_zero(const mtx_float *x);
bool mtx_is_normalized(const mtx_float *x);
const char *mtx_status_string(mtx_status status);

#ifdef __cplusplus
}
#endif

#endif
