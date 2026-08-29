#include "mantix/limb.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t next_random(uint64_t *state)
{
    uint64_t value = *state;
    value ^= value >> 12U;
    value ^= value << 25U;
    value ^= value >> 27U;
    *state = value;
    return value * UINT64_C(0x2545f4914f6cdd1d);
}

static mtx_limb reference_add(mtx_limb *result, const mtx_limb *left,
                              const mtx_limb *right, size_t count)
{
    mtx_limb carry = 0U;
    for (size_t i = 0U; i < count; ++i) {
        __uint128_t sum = (__uint128_t)left[i] + right[i] + carry;
        result[i] = (mtx_limb)sum;
        carry = (mtx_limb)(sum >> 64U);
    }
    return carry;
}

static void reference_mul(mtx_limb *result, const mtx_limb *left,
                          const mtx_limb *right, size_t count)
{
    memset(result, 0, count * 2U * sizeof(*result));
    for (size_t i = 0U; i < count; ++i) {
        mtx_limb carry = 0U;
        for (size_t j = 0U; j < count; ++j) {
            __uint128_t product = (__uint128_t)left[i] * right[j] +
                                  result[i + j] + carry;
            result[i + j] = (mtx_limb)product;
            carry = (mtx_limb)(product >> 64U);
        }
        result[i + count] = carry;
    }
}

static void test_edges(void)
{
    mtx_limb left[] = {UINT64_MAX, UINT64_MAX, 0U};
    mtx_limb right[] = {1U, 0U, UINT64_MAX};
    mtx_limb result[3];

    assert(mtx_limb_add_n(result, left, right, 3U) == 1U);
    assert(result[0] == 0U && result[1] == 0U && result[2] == 0U);
    assert(mtx_limb_sub_n(result, result, right, 3U) == 1U);
    assert(memcmp(result, left, sizeof(left)) == 0);

    assert(mtx_limb_add_n(result, left, right, 0U) == 0U);
    assert(mtx_limb_sub_n(result, left, right, 0U) == 0U);
}

static void test_random_and_aliasing(void)
{
    enum { LIMBS = 64, ROUNDS = 1000 };
    mtx_limb left[LIMBS];
    mtx_limb right[LIMBS];
    mtx_limb expected[LIMBS];
    mtx_limb actual[LIMBS];
    uint64_t random = UINT64_C(0xc0ffee123456789a);

    for (size_t round = 0U; round < ROUNDS; ++round) {
        size_t count = round % LIMBS + 1U;
        for (size_t i = 0U; i < count; ++i) {
            left[i] = next_random(&random);
            right[i] = next_random(&random);
        }

        mtx_limb expected_carry = reference_add(expected, left, right, count);
        assert(mtx_limb_add_n(actual, left, right, count) == expected_carry);
        assert(memcmp(actual, expected, count * sizeof(*actual)) == 0);

        memcpy(actual, left, count * sizeof(*actual));
        assert(mtx_limb_add_n(actual, actual, right, count) == expected_carry);
        assert(memcmp(actual, expected, count * sizeof(*actual)) == 0);

        assert(mtx_limb_sub_n(actual, expected, right, count) == expected_carry);
        assert(memcmp(actual, left, count * sizeof(*actual)) == 0);
    }
}

static void test_multiply_edges(void)
{
    mtx_limb result[6] = {11U, 12U, 13U, 14U, 15U, 16U};
    const mtx_limb one[] = {1U};
    const mtx_limb maximum[] = {UINT64_MAX};
    const mtx_limb left[] = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
    const mtx_limb right[] = {UINT64_MAX, 0U, UINT64_MAX};
    mtx_limb expected[6];

    mtx_limb_mul_n(NULL, NULL, NULL, 0U);
    mtx_limb_mul_n(result, one, maximum, 0U);
    assert(result[0] == 11U);
    mtx_limb_mul_n(result, maximum, maximum, 1U);
    assert(result[0] == 1U && result[1] == UINT64_MAX - 1U);

    reference_mul(expected, left, right, 3U);
    mtx_limb_mul_n(result, left, right, 3U);
    assert(memcmp(result, expected, sizeof(expected)) == 0);
}

static void test_multiply_random(void)
{
    enum { LIMBS = 64, ROUNDS = 500 };
    mtx_limb left[LIMBS];
    mtx_limb right[LIMBS];
    mtx_limb expected[LIMBS * 2U];
    mtx_limb actual[LIMBS * 2U];
    uint64_t random = UINT64_C(0x83dc72e284a9c7b1);

    for (size_t round = 0U; round < ROUNDS; ++round) {
        size_t count = round % LIMBS + 1U;
        for (size_t i = 0U; i < count; ++i) {
            left[i] = next_random(&random);
            right[i] = next_random(&random);
        }
        reference_mul(expected, left, right, count);
        mtx_limb_mul_n(actual, left, right, count);
        if (memcmp(actual, expected,
                   count * 2U * sizeof(*actual)) != 0) {
            fprintf(stderr, "multiply mismatch: round=%zu count=%zu\n",
                    round, count);
            for (size_t i = 0U; i < count; ++i) {
                fprintf(stderr, "  input %zu: left=%016llx right=%016llx\n",
                        i, (unsigned long long)left[i],
                        (unsigned long long)right[i]);
            }
            for (size_t i = 0U; i < count * 2U; ++i) {
                fprintf(stderr, "  limb %zu: expected=%016llx actual=%016llx\n",
                        i, (unsigned long long)expected[i],
                        (unsigned long long)actual[i]);
            }
            abort();
        }
        assert(memcmp(actual, expected,
                      count * 2U * sizeof(*actual)) == 0);

        reference_mul(expected, left, left, count);
        mtx_limb_mul_n(actual, left, left, count);
        assert(memcmp(actual, expected,
                      count * 2U * sizeof(*actual)) == 0);
    }
}

int main(void)
{
    uint32_t features = mtx_cpu_features();
    assert(mtx_limb_backend() != NULL);
    assert((features & ~(uint32_t)(MTX_CPU_FEATURE_BMI2 | MTX_CPU_FEATURE_ADX |
                                  MTX_CPU_FEATURE_AVX2)) == 0U);
    test_edges();
    test_random_and_aliasing();
    test_multiply_edges();
    test_multiply_random();
    return 0;
}
