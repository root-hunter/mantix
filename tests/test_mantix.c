#include "mantix/mantix.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static void test_initialization(void)
{
    mtx_float value;

    assert(mtx_init(NULL, 128U) == MTX_ERROR_INVALID_ARGUMENT);
    assert(mtx_init(&value, 0U) == MTX_ERROR_INVALID_ARGUMENT);
    assert(mtx_init(&value, 128U) == MTX_OK);
    assert(value.precision == 128U);
    assert(value.capacity == 1U);
    assert(value.limbs == &value.inline_limb);
    assert(mtx_is_zero(&value));
    assert(mtx_is_normalized(&value));
    mtx_clear(&value);
    mtx_clear(&value);
}

static void test_machine_integer(void)
{
    mtx_float value;

    assert(mtx_init(&value, 256U) == MTX_OK);
    assert(mtx_set_u64(&value, UINT64_C(40)) == MTX_OK);
    assert(value.used == 1U);
    assert(value.limbs == &value.inline_limb);
    assert(value.limbs[0] == UINT64_C(5));
    assert(value.exponent == 3);
    assert(!value.negative);
    assert(mtx_is_normalized(&value));

    assert(mtx_set_u64(&value, 0U) == MTX_OK);
    assert(mtx_is_zero(&value));
    assert(mtx_is_normalized(&value));
    mtx_clear(&value);
}

static void test_copy(void)
{
    mtx_float source;
    mtx_float destination;

    assert(mtx_init(&source, 64U) == MTX_OK);
    assert(mtx_init(&destination, 512U) == MTX_OK);
    assert(mtx_set_u64(&source, UINT64_C(123456)) == MTX_OK);
    assert(mtx_set(&destination, &source) == MTX_OK);

    assert(destination.precision == 512U);
    assert(destination.used == source.used);
    assert(destination.exponent == source.exponent);
    assert(memcmp(destination.limbs, source.limbs,
                  source.used * sizeof(*source.limbs)) == 0);

    mtx_clear(&destination);
    mtx_clear(&source);
}

static void test_copy_grows_inline_storage(void)
{
    mtx_limb limbs[] = {UINT64_C(3), UINT64_C(5), UINT64_C(7)};
    mtx_float source = {
        .limbs = limbs,
        .used = 3U,
        .capacity = 3U,
        .precision = 192U,
    };
    mtx_float destination;

    assert(mtx_is_normalized(&source));
    assert(mtx_init(&destination, 192U) == MTX_OK);
    assert(mtx_set_u64(&destination, UINT64_C(9)) == MTX_OK);
    assert(destination.limbs == &destination.inline_limb);

    assert(mtx_set(&destination, &source) == MTX_OK);
    assert(destination.capacity == 3U);
    assert(destination.limbs != &destination.inline_limb);
    assert(memcmp(destination.limbs, limbs, sizeof(limbs)) == 0);
    assert(mtx_is_normalized(&destination));
    mtx_clear(&destination);
}

static void test_signed_integer(void)
{
    mtx_float val;
    assert(mtx_init(&val, 64U) == MTX_OK);
    assert(mtx_set_i64(&val, -42) == MTX_OK);
    assert(val.negative);
    assert(val.limbs[0] == 21U);
    assert(val.exponent == 1);
    assert(mtx_is_normalized(&val));

    assert(mtx_set_i64(&val, 42) == MTX_OK);
    assert(!val.negative);
    assert(val.limbs[0] == 21U);
    assert(val.exponent == 1);

    assert(mtx_set_i64(&val, 0) == MTX_OK);
    assert(mtx_is_zero(&val));

    mtx_clear(&val);
}

static void test_float_conversions(void)
{
    mtx_float val;
    assert(mtx_init(&val, 64U) == MTX_OK);

    assert(mtx_set_f32(&val, 0.0f) == MTX_OK);
    assert(mtx_is_zero(&val));
    assert(mtx_get_f32(&val, MTX_ROUND_TO_NEAREST_EVEN) == 0.0f);

    assert(mtx_set_f32(&val, 1.5f) == MTX_OK);
    assert(!val.negative);
    assert(val.limbs[0] == 3U);
    assert(val.exponent == -1);
    assert(mtx_get_f32(&val, MTX_ROUND_TO_NEAREST_EVEN) == 1.5f);

    assert(mtx_set_f32(&val, -0.125f) == MTX_OK);
    assert(val.negative);
    assert(val.limbs[0] == 1U);
    assert(val.exponent == -3);
    assert(mtx_get_f32(&val, MTX_ROUND_TO_NEAREST_EVEN) == -0.125f);

    assert(mtx_set_f32(&val, 100.25f) == MTX_OK);
    assert(mtx_get_f32(&val, MTX_ROUND_TO_NEAREST_EVEN) == 100.25f);

    mtx_clear(&val);
}

static void test_double_conversions(void)
{
    mtx_float val;
    assert(mtx_init(&val, 128U) == MTX_OK);

    assert(mtx_set_f64(&val, 0.0) == MTX_OK);
    assert(mtx_is_zero(&val));
    assert(mtx_get_f64(&val, MTX_ROUND_TO_NEAREST_EVEN) == 0.0);

    assert(mtx_set_d(&val, 3.141592653589793) == MTX_OK);
    assert(mtx_get_d(&val, MTX_ROUND_TO_NEAREST_EVEN) == 3.141592653589793);

    assert(mtx_set_f64(&val, -1.23456789e-20) == MTX_OK);
    assert(val.negative);
    assert(mtx_get_f64(&val, MTX_ROUND_TO_NEAREST_EVEN) == -1.23456789e-20);

    mtx_clear(&val);
}

static void test_comparisons(void)
{
    mtx_float a, b;
    assert(mtx_init(&a, 128U) == MTX_OK);
    assert(mtx_init(&b, 128U) == MTX_OK);

    assert(mtx_cmp(&a, &b) == 0);

    mtx_set_f64(&a, 5.0);
    mtx_set_f64(&b, 3.0);
    assert(mtx_cmp(&a, &b) > 0);
    assert(mtx_cmp(&b, &a) < 0);
    assert(mtx_cmp(&a, &a) == 0);

    mtx_set_f64(&b, -10.0);
    assert(mtx_cmp(&a, &b) > 0);
    assert(mtx_cmp(&b, &a) < 0);

    mtx_set_f64(&a, -20.0);
    assert(mtx_cmp(&a, &b) < 0);
    assert(mtx_cmp(&b, &a) > 0);

    mtx_clear(&b);
    mtx_clear(&a);
}

static void test_floating_arithmetic_basic(void)
{
    mtx_float a, b, r;
    assert(mtx_init(&a, 128U) == MTX_OK);
    assert(mtx_init(&b, 128U) == MTX_OK);
    assert(mtx_init(&r, 128U) == MTX_OK);

    /* 1.5 + 2.5 = 4.0 */
    mtx_set_f64(&a, 1.5);
    mtx_set_f64(&b, 2.5);
    assert(mtx_add(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == 4.0);

    /* 1.5 - 2.5 = -1.0 */
    assert(mtx_sub(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == -1.0);

    /* 1.5 * 2.5 = 3.75 */
    assert(mtx_mul(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == 3.75);

    /* a * 0 = 0 */
    mtx_set_zero(&b);
    assert(mtx_mul(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_is_zero(&r));

    /* a + 0 = a */
    assert(mtx_add(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == 1.5);

    /* a - a = 0 */
    assert(mtx_sub(&r, &a, &a, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_is_zero(&r));

    mtx_clear(&r);
    mtx_clear(&b);
    mtx_clear(&a);
}

static void test_floating_rounding_modes(void)
{
    mtx_float a, b, r;
    /* Low precision: 4 bits */
    assert(mtx_init(&a, 16U) == MTX_OK);
    assert(mtx_init(&b, 16U) == MTX_OK);
    assert(mtx_init(&r, 4U) == MTX_OK);

    /* a = 15 (1111_2), b = 1 (1_2) -> a + b = 16 (10000_2 = 1 * 2^4) */
    mtx_set_u64(&a, 15U);
    mtx_set_u64(&b, 1U);
    assert(mtx_add(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(r.limbs[0] == 1U && r.exponent == 4);

    /* a = 9 (1001_2), b = 2 (0010_2) -> 11 (1011_2) */
    mtx_set_u64(&a, 9U);
    mtx_set_u64(&b, 2U);
    assert(mtx_add(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(r.limbs[0] == 11U && r.exponent == 0);

    mtx_clear(&r);
    mtx_clear(&b);
    mtx_clear(&a);
}

static void test_aliasing(void)
{
    mtx_float a, b;
    assert(mtx_init(&a, 128U) == MTX_OK);
    assert(mtx_init(&b, 128U) == MTX_OK);

    mtx_set_f64(&a, 3.0);
    mtx_set_f64(&b, 2.0);

    /* a = a + b */
    assert(mtx_add(&a, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&a, MTX_ROUND_TO_NEAREST_EVEN) == 5.0);

    /* a = a * a */
    assert(mtx_mul(&a, &a, &a, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&a, MTX_ROUND_TO_NEAREST_EVEN) == 25.0);

    /* a = a - a */
    assert(mtx_sub(&a, &a, &a, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_is_zero(&a));

    mtx_clear(&b);
    mtx_clear(&a);
}

static void test_division(void)
{
    mtx_float a, b, r;
    assert(mtx_init(&a, 128U) == MTX_OK);
    assert(mtx_init(&b, 128U) == MTX_OK);
    assert(mtx_init(&r, 128U) == MTX_OK);

    /* 6.0 / 2.0 = 3.0 */
    mtx_set_f64(&a, 6.0);
    mtx_set_f64(&b, 2.0);
    assert(mtx_div(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == 3.0);

    /* 1.0 / 4.0 = 0.25 */
    mtx_set_f64(&a, 1.0);
    mtx_set_f64(&b, 4.0);
    assert(mtx_div(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == 0.25);

    /* Division by zero */
    mtx_set_zero(&b);
    assert(mtx_div(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_ERROR_DIVISION_BY_ZERO);

    /* 0.0 / 5.0 = 0.0 */
    mtx_set_zero(&a);
    mtx_set_f64(&b, 5.0);
    assert(mtx_div(&r, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_is_zero(&r));

    /* In-place division: a = 20.0, a = a / 4.0 -> 5.0 */
    mtx_set_f64(&a, 20.0);
    mtx_set_f64(&b, 4.0);
    assert(mtx_div(&a, &a, &b, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&a, MTX_ROUND_TO_NEAREST_EVEN) == 5.0);

    mtx_clear(&r);
    mtx_clear(&b);
    mtx_clear(&a);
}

static void test_sqrt(void)
{
    mtx_float a, r;
    assert(mtx_init(&a, 128U) == MTX_OK);
    assert(mtx_init(&r, 128U) == MTX_OK);

    /* sqrt(4.0) = 2.0 */
    mtx_set_f64(&a, 4.0);
    assert(mtx_sqrt(&r, &a, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN) == 2.0);

    /* sqrt(2.0) approx 1.4142135623730951 */
    mtx_set_f64(&a, 2.0);
    assert(mtx_sqrt(&r, &a, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    double val = mtx_get_f64(&r, MTX_ROUND_TO_NEAREST_EVEN);
    assert(fabs(val - 1.4142135623730951) < 1e-14);

    /* sqrt(0.0) = 0.0 */
    mtx_set_zero(&a);
    assert(mtx_sqrt(&r, &a, MTX_ROUND_TO_NEAREST_EVEN) == MTX_OK);
    assert(mtx_is_zero(&r));

    mtx_clear(&r);
    mtx_clear(&a);
}

int main(void)
{
    test_initialization();
    test_machine_integer();
    test_copy();
    test_copy_grows_inline_storage();
    test_signed_integer();
    test_float_conversions();
    test_double_conversions();
    test_comparisons();
    test_floating_arithmetic_basic();
    test_floating_rounding_modes();
    test_aliasing();
    test_division();
    test_sqrt();
    return 0;
}
