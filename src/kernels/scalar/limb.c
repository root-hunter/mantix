#include "mantix/limb.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#include <immintrin.h>

static inline bool mtx_has_adx(void)
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1U << 19)) != 0; /* bit 19 of EBX in leaf 7 is ADX */
    }
    return false;
}
#endif

uint32_t mtx_cpu_features(void)
{
    uint32_t features = MTX_CPU_FEATURE_NONE;

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("bmi2")) {
        features |= MTX_CPU_FEATURE_BMI2;
    }
    if (mtx_has_adx()) {
        features |= MTX_CPU_FEATURE_ADX;
    }
    if (__builtin_cpu_supports("avx2")) {
        features |= MTX_CPU_FEATURE_AVX2;
    }
    if (__builtin_cpu_supports("avx512f")) {
        features |= MTX_CPU_FEATURE_AVX512;
    }
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    features |= MTX_CPU_FEATURE_NEON;
#endif
#if defined(__ARM_FEATURE_SVE)
    features |= MTX_CPU_FEATURE_SVE;
#endif
#endif
    return features;
}

const char *mtx_limb_backend(void)
{
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("bmi2") && mtx_has_adx()) {
        return "x86_64-adc+bmi2+adx+avx512";
    }
    if (__builtin_cpu_supports("bmi2") && mtx_has_adx()) {
        return "x86_64-adc+bmi2+adx";
    }
    if (__builtin_cpu_supports("bmi2")) {
        return "x86_64-adc+bmi2";
    }
    return "x86_64-adc+u128";
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(__ARM_FEATURE_SVE)
    return "arm64-adcs+umulh+sve";
#else
    return "arm64-adcs+umulh+neon";
#endif
#elif defined(__SIZEOF_INT128__)
    return "portable-u128";
#else
    return "portable-c";
#endif
}

static void mtx_limb_mul_n_portable(mtx_limb *result, const mtx_limb *left,
                                    const mtx_limb *right, size_t count)
{
#if defined(__SIZEOF_INT128__)
    mtx_limb first_carry = 0U;
    for (size_t j = 0U; j < count; ++j) {
        __uint128_t product = (__uint128_t)left[j] * right[0] + first_carry;
        result[j] = (mtx_limb)product;
        first_carry = (mtx_limb)(product >> 64U);
    }
    result[count] = first_carry;

    for (size_t i = 1U; i < count; ++i) {
#else
    memset(result, 0, count * sizeof(*result));
    for (size_t i = 0U; i < count; ++i) {
#endif
        mtx_limb carry = 0U;

        for (size_t j = 0U; j < count; ++j) {
#if defined(__SIZEOF_INT128__)
            __uint128_t product = (__uint128_t)left[j] * right[i] +
                                  result[i + j] + carry;
            result[i + j] = (mtx_limb)product;
            carry = (mtx_limb)(product >> 64U);
#else
            const mtx_limb mask = UINT64_C(0xffffffff);
            mtx_limb left_low = left[j] & mask;
            mtx_limb left_high = left[j] >> 32U;
            mtx_limb right_low = right[i] & mask;
            mtx_limb right_high = right[i] >> 32U;
            mtx_limb low_product = left_low * right_low;
            mtx_limb cross_left = left_high * right_low;
            mtx_limb cross_right = left_low * right_high;
            mtx_limb middle = (low_product >> 32U) +
                              (cross_left & mask) + (cross_right & mask);
            mtx_limb low = (low_product & mask) | (middle << 32U);
            mtx_limb high = left_high * right_high +
                            (cross_left >> 32U) + (cross_right >> 32U) +
                            (middle >> 32U);
            mtx_limb sum = low + result[i + j];
            mtx_limb result_carry = (mtx_limb)(sum < low);
            mtx_limb final = sum + carry;
            mtx_limb carry_carry = (mtx_limb)(final < sum);

            result[i + j] = final;
            carry = high + result_carry + carry_carry;
#endif
        }
        result[i + count] = carry;
    }
}

#if defined(__SIZEOF_INT128__)
static void mtx_limb_mul_2(mtx_limb *result, const mtx_limb *left,
                           const mtx_limb *right)
{
    __uint128_t product_00 = (__uint128_t)left[0] * right[0];
    __uint128_t product_01 = (__uint128_t)left[0] * right[1];
    __uint128_t product_10 = (__uint128_t)left[1] * right[0];
    __uint128_t product_11 = (__uint128_t)left[1] * right[1];
    __uint128_t column = (product_00 >> 64U) + (mtx_limb)product_01;
    mtx_limb carry = (mtx_limb)(column >> 64U);

    result[0] = (mtx_limb)product_00;
    column = (__uint128_t)(mtx_limb)column + (mtx_limb)product_10;
    result[1] = (mtx_limb)column;
    column = (__uint128_t)(product_01 >> 64U) + (product_10 >> 64U) +
             carry + (mtx_limb)product_11 + (column >> 64U);
    result[2] = (mtx_limb)column;
    result[3] = (mtx_limb)(product_11 >> 64U) +
                (mtx_limb)(column >> 64U);
}

/* Four fixed limbs fit a compact, branch-free Comba schedule. */
static void mtx_limb_mul_4(mtx_limb *result, const mtx_limb *left,
                           const mtx_limb *right)
{
    mtx_limb low = 0U;
    mtx_limb middle = 0U;
    mtx_limb high = 0U;

#define MTX_MUL_ACCUMULATE(left_index, right_index)                            \
    do {                                                                       \
        __uint128_t mtx_product =                                              \
            (__uint128_t)left[(left_index)] * right[(right_index)];            \
        __uint128_t mtx_sum =                                                  \
            (__uint128_t)low + (mtx_limb)mtx_product;                          \
        low = (mtx_limb)mtx_sum;                                               \
        mtx_sum = (__uint128_t)middle + (mtx_limb)(mtx_product >> 64U) +        \
                  (mtx_limb)(mtx_sum >> 64U);                                  \
        middle = (mtx_limb)mtx_sum;                                            \
        high += (mtx_limb)(mtx_sum >> 64U);                                    \
    } while (0)
#define MTX_MUL_STORE(index)                                                   \
    do {                                                                       \
        result[(index)] = low;                                                 \
        low = middle;                                                          \
        middle = high;                                                         \
        high = 0U;                                                             \
    } while (0)

    MTX_MUL_ACCUMULATE(0U, 0U);
    MTX_MUL_STORE(0U);
    MTX_MUL_ACCUMULATE(0U, 1U);
    MTX_MUL_ACCUMULATE(1U, 0U);
    MTX_MUL_STORE(1U);
    MTX_MUL_ACCUMULATE(0U, 2U);
    MTX_MUL_ACCUMULATE(1U, 1U);
    MTX_MUL_ACCUMULATE(2U, 0U);
    MTX_MUL_STORE(2U);
    MTX_MUL_ACCUMULATE(0U, 3U);
    MTX_MUL_ACCUMULATE(1U, 2U);
    MTX_MUL_ACCUMULATE(2U, 1U);
    MTX_MUL_ACCUMULATE(3U, 0U);
    MTX_MUL_STORE(3U);
    MTX_MUL_ACCUMULATE(1U, 3U);
    MTX_MUL_ACCUMULATE(2U, 2U);
    MTX_MUL_ACCUMULATE(3U, 1U);
    MTX_MUL_STORE(4U);
    MTX_MUL_ACCUMULATE(2U, 3U);
    MTX_MUL_ACCUMULATE(3U, 2U);
    MTX_MUL_STORE(5U);
    MTX_MUL_ACCUMULATE(3U, 3U);
    MTX_MUL_STORE(6U);
    result[7] = low;

#undef MTX_MUL_STORE
#undef MTX_MUL_ACCUMULATE
}

#endif

#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
__attribute__((target("bmi2")))
static void mtx_limb_mul_n_bmi2(mtx_limb *result, const mtx_limb *left,
                                const mtx_limb *right, size_t count)
{
    unsigned long long first_carry = 0U;
    const unsigned long long first_multiplier =
        (unsigned long long)right[0];

#if defined(__clang__)
#pragma clang loop unroll_count(16)
#endif
    for (size_t j = 0U; j < count; ++j) {
        unsigned long long high;
        unsigned long long low =
            _mulx_u64((unsigned long long)left[j], first_multiplier, &high);
        unsigned char carry_carry =
            _addcarry_u64(0U, low, first_carry, &low);

        result[j] = (mtx_limb)low;
        first_carry = high + (unsigned long long)carry_carry;
    }
    result[count] = (mtx_limb)first_carry;

    for (size_t i = 1U; i < count; ++i) {
        unsigned long long carry = 0U;
        const unsigned long long multiplier =
            (unsigned long long)right[i];

#if defined(__clang__)
#pragma clang loop unroll_count(16)
#endif
        for (size_t j = 0U; j < count; ++j) {
            unsigned long long high;
            unsigned long long low =
                _mulx_u64((unsigned long long)left[j], multiplier, &high);
            unsigned char product_carry =
                _addcarry_u64(0U, low, (unsigned long long)result[i + j],
                              &low);
            unsigned char carry_carry =
                _addcarry_u64(0U, low, carry, &low);

            result[i + j] = (mtx_limb)low;
            carry = high + (unsigned long long)product_carry +
                    (unsigned long long)carry_carry;
        }
        result[i + count] = (mtx_limb)carry;
    }
}

__attribute__((always_inline))
static inline void mtx_limb_accumulate_diagonal_adx(
    mtx_limb *accumulator_0, mtx_limb *accumulator_1,
    mtx_limb *accumulator_2, const mtx_limb *left,
    const mtx_limb *right, size_t count)
{
    unsigned long long value_0 = (unsigned long long)*accumulator_0;
    unsigned long long value_1 = (unsigned long long)*accumulator_1;
    unsigned long long value_2 = (unsigned long long)*accumulator_2;
    unsigned long long high;
    unsigned long long low;
    unsigned long long zero;
    size_t pairs = count / 2U;
    size_t odd = count & 1U;

    __asm__ volatile(
        "xor %k[zero], %k[zero]\n\t"
        "test %[pairs], %[pairs]\n\t"
        "jz 2f\n\t"
        "1:\n\t"
        "mov (%[right]), %%rdx\n\t"
        "mulx (%[left]), %[low], %[high]\n\t"
        "adcx %[low], %[value_0]\n\t"
        "adcx %[high], %[value_1]\n\t"
        "adcx %[zero], %[value_2]\n\t"
        "mov -8(%[right]), %%rdx\n\t"
        "mulx 8(%[left]), %[low], %[high]\n\t"
        "adox %[low], %[value_0]\n\t"
        "adox %[high], %[value_1]\n\t"
        "adox %[zero], %[value_2]\n\t"
        "lea 16(%[left]), %[left]\n\t"
        "lea -16(%[right]), %[right]\n\t"
        "lea -1(%[pairs]), %[pairs]\n\t"
        "jrcxz 2f\n\t"
        "jmp 1b\n\t"
        "2:\n\t"
        "test %[odd], %[odd]\n\t"
        "jz 3f\n\t"
        "mov (%[right]), %%rdx\n\t"
        "mulx (%[left]), %[low], %[high]\n\t"
        "adcx %[low], %[value_0]\n\t"
        "adcx %[high], %[value_1]\n\t"
        "adcx %[zero], %[value_2]\n\t"
        "3:"
        : [value_0] "+&r"(value_0), [value_1] "+&r"(value_1),
          [value_2] "+&r"(value_2), [high] "=&r"(high),
          [low] "=&r"(low), [zero] "=&r"(zero), [left] "+&r"(left),
          [right] "+&r"(right), [pairs] "+&c"(pairs)
        : [odd] "r"(odd)
        : "rdx", "cc", "memory");
    *accumulator_0 = (mtx_limb)value_0;
    *accumulator_1 = (mtx_limb)value_1;
    *accumulator_2 = (mtx_limb)value_2;
}

__attribute__((target("bmi2,adx")))
static void mtx_limb_mul_n_adx(mtx_limb *result, const mtx_limb *left,
                               const mtx_limb *right, size_t count)
{
    mtx_limb accumulator_0 = 0U;
    mtx_limb accumulator_1 = 0U;
    mtx_limb accumulator_2 = 0U;
    const size_t columns = count * 2U - 1U;

    for (size_t column = 0U; column < columns; ++column) {
        size_t start = column < count ? 0U : column - count + 1U;
        size_t end = column < count ? column : count - 1U;

        mtx_limb_accumulate_diagonal_adx(
            &accumulator_0, &accumulator_1, &accumulator_2, left + start,
            right + (column - start), end - start + 1U);
        result[column] = accumulator_0;
        accumulator_0 = accumulator_1;
        accumulator_1 = accumulator_2;
        accumulator_2 = 0U;
    }
    result[columns] = accumulator_0;
}

#endif

__attribute__((noinline))
static void mtx_limb_mul_n_basecase(mtx_limb *result,
                                    const mtx_limb *left,
                                    const mtx_limb *right, size_t count)
{
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("bmi2")) {
        if (mtx_has_adx()) {
            mtx_limb_mul_n_adx(result, left, right, count);
            return;
        }
        mtx_limb_mul_n_bmi2(result, left, right, count);
        return;
    }
#endif
    mtx_limb_mul_n_portable(result, left, right, count);
}

#if defined(__SIZEOF_INT128__)
static void mtx_limb_mul_8(mtx_limb *result, const mtx_limb *left,
                           const mtx_limb *right)
{
    mtx_limb low = 0U;
    mtx_limb middle = 0U;
    mtx_limb high = 0U;

#define MTX_MUL8_ACC(i, j)                                                     \
    do {                                                                       \
        __uint128_t mtx_p = (__uint128_t)left[(i)] * right[(j)];               \
        __uint128_t mtx_s = (__uint128_t)low + (mtx_limb)mtx_p;               \
        low = (mtx_limb)mtx_s;                                                 \
        mtx_s = (__uint128_t)middle + (mtx_limb)(mtx_p >> 64U) +              \
                (mtx_limb)(mtx_s >> 64U);                                      \
        middle = (mtx_limb)mtx_s;                                              \
        high += (mtx_limb)(mtx_s >> 64U);                                      \
    } while (0)

#define MTX_MUL8_STORE(idx)                                                    \
    do {                                                                       \
        result[(idx)] = low;                                                   \
        low = middle;                                                          \
        middle = high;                                                         \
        high = 0U;                                                             \
    } while (0)

    MTX_MUL8_ACC(0, 0); MTX_MUL8_STORE(0);
    MTX_MUL8_ACC(0, 1); MTX_MUL8_ACC(1, 0); MTX_MUL8_STORE(1);
    MTX_MUL8_ACC(0, 2); MTX_MUL8_ACC(1, 1); MTX_MUL8_ACC(2, 0); MTX_MUL8_STORE(2);
    MTX_MUL8_ACC(0, 3); MTX_MUL8_ACC(1, 2); MTX_MUL8_ACC(2, 1); MTX_MUL8_ACC(3, 0); MTX_MUL8_STORE(3);
    MTX_MUL8_ACC(0, 4); MTX_MUL8_ACC(1, 3); MTX_MUL8_ACC(2, 2); MTX_MUL8_ACC(3, 1); MTX_MUL8_ACC(4, 0); MTX_MUL8_STORE(4);
    MTX_MUL8_ACC(0, 5); MTX_MUL8_ACC(1, 4); MTX_MUL8_ACC(2, 3); MTX_MUL8_ACC(3, 2); MTX_MUL8_ACC(4, 1); MTX_MUL8_ACC(5, 0); MTX_MUL8_STORE(5);
    MTX_MUL8_ACC(0, 6); MTX_MUL8_ACC(1, 5); MTX_MUL8_ACC(2, 4); MTX_MUL8_ACC(3, 3); MTX_MUL8_ACC(4, 2); MTX_MUL8_ACC(5, 1); MTX_MUL8_ACC(6, 0); MTX_MUL8_STORE(6);
    MTX_MUL8_ACC(0, 7); MTX_MUL8_ACC(1, 6); MTX_MUL8_ACC(2, 5); MTX_MUL8_ACC(3, 4); MTX_MUL8_ACC(4, 3); MTX_MUL8_ACC(5, 2); MTX_MUL8_ACC(6, 1); MTX_MUL8_ACC(7, 0); MTX_MUL8_STORE(7);
    MTX_MUL8_ACC(1, 7); MTX_MUL8_ACC(2, 6); MTX_MUL8_ACC(3, 5); MTX_MUL8_ACC(4, 4); MTX_MUL8_ACC(5, 3); MTX_MUL8_ACC(6, 2); MTX_MUL8_ACC(7, 1); MTX_MUL8_STORE(8);
    MTX_MUL8_ACC(2, 7); MTX_MUL8_ACC(3, 6); MTX_MUL8_ACC(4, 5); MTX_MUL8_ACC(5, 4); MTX_MUL8_ACC(6, 3); MTX_MUL8_ACC(7, 2); MTX_MUL8_STORE(9);
    MTX_MUL8_ACC(3, 7); MTX_MUL8_ACC(4, 6); MTX_MUL8_ACC(5, 5); MTX_MUL8_ACC(6, 4); MTX_MUL8_ACC(7, 3); MTX_MUL8_STORE(10);
    MTX_MUL8_ACC(4, 7); MTX_MUL8_ACC(5, 6); MTX_MUL8_ACC(6, 5); MTX_MUL8_ACC(7, 4); MTX_MUL8_STORE(11);
    MTX_MUL8_ACC(5, 7); MTX_MUL8_ACC(6, 6); MTX_MUL8_ACC(7, 5); MTX_MUL8_STORE(12);
    MTX_MUL8_ACC(6, 7); MTX_MUL8_ACC(7, 6); MTX_MUL8_STORE(13);
    MTX_MUL8_ACC(7, 7); MTX_MUL8_STORE(14);
    result[15] = low;

#undef MTX_MUL8_STORE
#undef MTX_MUL8_ACC
}

static void mtx_limb_mul_16(mtx_limb *result, const mtx_limb *left,
                            const mtx_limb *right)
{
    mtx_limb left_sum[8];
    mtx_limb right_sum[8];
    mtx_limb middle[17];

    mtx_limb carry_left = mtx_limb_add_n(left_sum, left, left + 8U, 8U);
    mtx_limb carry_right = mtx_limb_add_n(right_sum, right, right + 8U, 8U);

    mtx_limb_mul_8(result, left, right);
    mtx_limb_mul_8(result + 16U, left + 8U, right + 8U);
    mtx_limb_mul_8(middle, left_sum, right_sum);
    middle[16] = 0U;

    if (carry_left != 0U) {
        middle[16] += mtx_limb_add_n(middle + 8U, middle + 8U, right_sum, 8U);
    }
    if (carry_right != 0U) {
        middle[16] += mtx_limb_add_n(middle + 8U, middle + 8U, left_sum, 8U);
    }
    if (carry_left != 0U && carry_right != 0U) {
        middle[16] += 1U;
    }

    mtx_limb borrow = mtx_limb_sub_n(middle, middle, result, 16U);
    middle[16] -= borrow;
    borrow = mtx_limb_sub_n(middle, middle, result + 16U, 16U);
    middle[16] -= borrow;

    mtx_limb carry = mtx_limb_add_n(result + 8U, result + 8U, middle, 17U);
    for (size_t i = 25U; carry != 0U && i < 32U; ++i) {
        mtx_limb value = result[i];
        result[i] = value + carry;
        carry = (mtx_limb)(result[i] < value);
    }
}

static void mtx_limb_mul_32(mtx_limb *result, const mtx_limb *left,
                            const mtx_limb *right)
{
    mtx_limb left_sum[16];
    mtx_limb right_sum[16];
    mtx_limb middle[33];

    mtx_limb carry_left = mtx_limb_add_n(left_sum, left, left + 16U, 16U);
    mtx_limb carry_right = mtx_limb_add_n(right_sum, right, right + 16U, 16U);

    mtx_limb_mul_16(result, left, right);
    mtx_limb_mul_16(result + 32U, left + 16U, right + 16U);
    mtx_limb_mul_16(middle, left_sum, right_sum);
    middle[32] = 0U;

    if (carry_left != 0U) {
        middle[32] += mtx_limb_add_n(middle + 16U, middle + 16U, right_sum, 16U);
    }
    if (carry_right != 0U) {
        middle[32] += mtx_limb_add_n(middle + 16U, middle + 16U, left_sum, 16U);
    }
    if (carry_left != 0U && carry_right != 0U) {
        middle[32] += 1U;
    }

    mtx_limb borrow = mtx_limb_sub_n(middle, middle, result, 32U);
    middle[32] -= borrow;
    borrow = mtx_limb_sub_n(middle, middle, result + 32U, 32U);
    middle[32] -= borrow;

    mtx_limb carry = mtx_limb_add_n(result + 16U, result + 16U, middle, 33U);
    for (size_t i = 49U; carry != 0U && i < 64U; ++i) {
        mtx_limb value = result[i];
        result[i] = value + carry;
        carry = (mtx_limb)(result[i] < value);
    }
}

static void mtx_limb_mul_64(mtx_limb *result, const mtx_limb *left,
                            const mtx_limb *right)
{
    mtx_limb left_sum[32];
    mtx_limb right_sum[32];
    mtx_limb middle[65];

    mtx_limb carry_left = mtx_limb_add_n(left_sum, left, left + 32U, 32U);
    mtx_limb carry_right = mtx_limb_add_n(right_sum, right, right + 32U, 32U);

    mtx_limb_mul_32(result, left, right);
    mtx_limb_mul_32(result + 64U, left + 32U, right + 32U);
    mtx_limb_mul_32(middle, left_sum, right_sum);
    middle[64] = 0U;

    if (carry_left != 0U) {
        middle[64] += mtx_limb_add_n(middle + 32U, middle + 32U, right_sum, 32U);
    }
    if (carry_right != 0U) {
        middle[64] += mtx_limb_add_n(middle + 32U, middle + 32U, left_sum, 32U);
    }
    if (carry_left != 0U && carry_right != 0U) {
        middle[64] += 1U;
    }

    mtx_limb borrow = mtx_limb_sub_n(middle, middle, result, 64U);
    middle[64] -= borrow;
    borrow = mtx_limb_sub_n(middle, middle, result + 64U, 64U);
    middle[64] -= borrow;

    mtx_limb carry = mtx_limb_add_n(result + 32U, result + 32U, middle, 65U);
    for (size_t i = 97U; carry != 0U && i < 128U; ++i) {
        mtx_limb value = result[i];
        result[i] = value + carry;
        carry = (mtx_limb)(result[i] < value);
    }
}
#endif

void mtx_limb_mul_n(mtx_limb *result, const mtx_limb *left,
                    const mtx_limb *right, size_t count)
{
    if (__builtin_expect(count == 0U, 0)) {
        return;
    }
    if (count == 1U) {
#if defined(__SIZEOF_INT128__)
        __uint128_t product = (__uint128_t)left[0] * right[0];
        result[0] = (mtx_limb)product;
        result[1] = (mtx_limb)(product >> 64U);
        return;
#endif
    }
#if defined(__SIZEOF_INT128__)
    if (count == 2U) {
        mtx_limb_mul_2(result, left, right);
        return;
    }
    if (count == 4U) {
        mtx_limb_mul_4(result, left, right);
        return;
    }
    if (count == 8U) {
        mtx_limb_mul_8(result, left, right);
        return;
    }
    if (count == 16U) {
        mtx_limb_mul_16(result, left, right);
        return;
    }
    if (count == 32U) {
        mtx_limb_mul_32(result, left, right);
        return;
    }
    if (count == 64U) {
        mtx_limb_mul_64(result, left, right);
        return;
    }
#endif
    mtx_limb_mul_n_basecase(result, left, right, count);
}

mtx_limb mtx_limb_add_n(mtx_limb *result, const mtx_limb *left,
                        const mtx_limb *right, size_t count)
{
    if (__builtin_expect(count == 0U, 0)) {
        return 0U;
    }
    if (count == 1U) {
        mtx_limb left_value = left[0];
        mtx_limb right_value = right[0];
        mtx_limb sum = left_value + right_value;
        result[0] = sum;
        return (mtx_limb)(sum < left_value);
    }
#if defined(__x86_64__)
    if (count == 2U) {
        unsigned char carry;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "add (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "adc 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "setc %b[carry]\n\t"
            : [carry] "=&q"(carry)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)carry;
    }
    if (count == 3U) {
        unsigned char carry;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "add (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "adc 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "mov 16(%[left]), %%r8\n\t"
            "adc 16(%[right]), %%r8\n\t"
            "mov %%r8, 16(%[result])\n\t"
            "setc %b[carry]\n\t"
            : [carry] "=&q"(carry)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)carry;
    }
    if (count == 4U) {
        unsigned char carry;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "add (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "adc 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "mov 16(%[left]), %%r8\n\t"
            "adc 16(%[right]), %%r8\n\t"
            "mov %%r8, 16(%[result])\n\t"
            "mov 24(%[left]), %%r8\n\t"
            "adc 24(%[right]), %%r8\n\t"
            "mov %%r8, 24(%[result])\n\t"
            "setc %b[carry]\n\t"
            : [carry] "=&q"(carry)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)carry;
    }
    if (count == 8U) {
        unsigned char carry;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "add (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "adc 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "mov 16(%[left]), %%r8\n\t"
            "adc 16(%[right]), %%r8\n\t"
            "mov %%r8, 16(%[result])\n\t"
            "mov 24(%[left]), %%r8\n\t"
            "adc 24(%[right]), %%r8\n\t"
            "mov %%r8, 24(%[result])\n\t"
            "mov 32(%[left]), %%r8\n\t"
            "adc 32(%[right]), %%r8\n\t"
            "mov %%r8, 32(%[result])\n\t"
            "mov 40(%[left]), %%r8\n\t"
            "adc 40(%[right]), %%r8\n\t"
            "mov %%r8, 40(%[result])\n\t"
            "mov 48(%[left]), %%r8\n\t"
            "adc 48(%[right]), %%r8\n\t"
            "mov %%r8, 48(%[result])\n\t"
            "mov 56(%[left]), %%r8\n\t"
            "adc 56(%[right]), %%r8\n\t"
            "mov %%r8, 56(%[result])\n\t"
            "setc %b[carry]\n\t"
            : [carry] "=&q"(carry)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)carry;
    }

    unsigned char carry;
    size_t blocks;

    __asm__ volatile(
        "mov %[count], %[blocks]\n\t"
        "shr $2, %[blocks]\n\t"
        "and $3, %[count]\n\t"
        "test %[blocks], %[blocks]\n\t"
        "jz 3f\n\t"
        "clc\n\t"
        "1:\n\t"
        "mov (%[left]), %%r8\n\t"
        "adc (%[right]), %%r8\n\t"
        "mov %%r8, (%[result])\n\t"
        "mov 8(%[left]), %%r8\n\t"
        "adc 8(%[right]), %%r8\n\t"
        "mov %%r8, 8(%[result])\n\t"
        "mov 16(%[left]), %%r8\n\t"
        "adc 16(%[right]), %%r8\n\t"
        "mov %%r8, 16(%[result])\n\t"
        "mov 24(%[left]), %%r8\n\t"
        "adc 24(%[right]), %%r8\n\t"
        "mov %%r8, 24(%[result])\n\t"
        "lea 32(%[left]), %[left]\n\t"
        "lea 32(%[right]), %[right]\n\t"
        "lea 32(%[result]), %[result]\n\t"
        "dec %[blocks]\n\t"
        "jnz 1b\n\t"
        "setc %b[carry]\n\t"
        "test %[count], %[count]\n\t"
        "jz 5f\n\t"
        "add $255, %b[carry]\n\t"
        "2:\n\t"
        "mov (%[left]), %%r8\n\t"
        "adc (%[right]), %%r8\n\t"
        "mov %%r8, (%[result])\n\t"
        "lea 8(%[left]), %[left]\n\t"
        "lea 8(%[right]), %[right]\n\t"
        "lea 8(%[result]), %[result]\n\t"
        "dec %[count]\n\t"
        "jnz 2b\n\t"
        "setc %b[carry]\n\t"
        "jmp 5f\n\t"
        "3:\n\t"
        "test %[count], %[count]\n\t"
        "jz 4f\n\t"
        "clc\n\t"
        "jmp 2b\n\t"
        "4:\n\t"
        "xor %k[carry], %k[carry]\n\t"
        "5:"
        : [carry] "=&q"(carry), [result] "+r"(result),
          [left] "+r"(left), [right] "+r"(right), [count] "+r"(count),
          [blocks] "=&r"(blocks)
        :
        : "r8", "cc", "memory");
    return (mtx_limb)carry;
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    if (count == 2U) {
        mtx_limb r0, r1, carry;
        __asm__ volatile(
            "adds %0, %3, %5\n\t"
            "adcs %1, %4, %6\n\t"
            "adc  %2, xzr, xzr\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(carry)
            : "r"(left[0]), "r"(left[1]), "r"(right[0]), "r"(right[1])
            : "cc"
        );
        result[0] = r0; result[1] = r1;
        return carry;
    }
    if (count == 3U) {
        mtx_limb r0, r1, r2, carry;
        __asm__ volatile(
            "adds %0, %4, %7\n\t"
            "adcs %1, %5, %8\n\t"
            "adcs %2, %6, %9\n\t"
            "adc  %3, xzr, xzr\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(carry)
            : "r"(left[0]), "r"(left[1]), "r"(left[2]),
              "r"(right[0]), "r"(right[1]), "r"(right[2])
            : "cc"
        );
        result[0] = r0; result[1] = r1; result[2] = r2;
        return carry;
    }
    if (count == 4U) {
        mtx_limb r0, r1, r2, r3, carry;
        __asm__ volatile(
            "adds %0, %5, %9\n\t"
            "adcs %1, %6, %10\n\t"
            "adcs %2, %7, %11\n\t"
            "adcs %3, %8, %12\n\t"
            "adc  %4, xzr, xzr\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3), "=&r"(carry)
            : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]),
              "r"(right[0]), "r"(right[1]), "r"(right[2]), "r"(right[3])
            : "cc"
        );
        result[0] = r0; result[1] = r1; result[2] = r2; result[3] = r3;
        return carry;
    }
    if (count == 8U) {
        mtx_limb r0, r1, r2, r3, r4, r5, r6, r7, carry;
        __asm__ volatile(
            "adds %0, %9, %17\n\t"
            "adcs %1, %10, %18\n\t"
            "adcs %2, %11, %19\n\t"
            "adcs %3, %12, %20\n\t"
            "adcs %4, %13, %21\n\t"
            "adcs %5, %14, %22\n\t"
            "adcs %6, %15, %23\n\t"
            "adcs %7, %16, %24\n\t"
            "adc  %8, xzr, xzr\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3),
              "=&r"(r4), "=&r"(r5), "=&r"(r6), "=&r"(r7), "=&r"(carry)
            : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]),
              "r"(left[4]), "r"(left[5]), "r"(left[6]), "r"(left[7]),
              "r"(right[0]), "r"(right[1]), "r"(right[2]), "r"(right[3]),
              "r"(right[4]), "r"(right[5]), "r"(right[6]), "r"(right[7])
            : "cc"
        );
        result[0] = r0; result[1] = r1; result[2] = r2; result[3] = r3;
        result[4] = r4; result[5] = r5; result[6] = r6; result[7] = r7;
        return carry;
    }
    mtx_limb carry = 0U;
    for (size_t i = 0U; i < count; ++i) {
        __uint128_t sum = (__uint128_t)left[i] + right[i] + carry;
        result[i] = (mtx_limb)sum;
        carry = (mtx_limb)(sum >> 64U);
    }
    return carry;
#elif defined(__SIZEOF_INT128__)
    mtx_limb carry = 0U;

    for (size_t i = 0U; i < count; ++i) {
        __uint128_t sum = (__uint128_t)left[i] + right[i] + carry;
        result[i] = (mtx_limb)sum;
        carry = (mtx_limb)(sum >> 64U);
    }
    return carry;
#else
    mtx_limb carry = 0U;

    for (size_t i = 0U; i < count; ++i) {
        mtx_limb partial = left[i] + carry;
        mtx_limb carry_left = (mtx_limb)(partial < left[i]);
        mtx_limb sum = partial + right[i];
        mtx_limb carry_right = (mtx_limb)(sum < partial);
        result[i] = sum;
        carry = carry_left | carry_right;
    }
    return carry;
#endif
}

mtx_limb mtx_limb_sub_n(mtx_limb *result, const mtx_limb *left,
                        const mtx_limb *right, size_t count)
{
    if (__builtin_expect(count == 0U, 0)) {
        return 0U;
    }
    if (count == 1U) {
        mtx_limb left_value = left[0];
        mtx_limb right_value = right[0];
        mtx_limb difference = left_value - right_value;
        result[0] = difference;
        return (mtx_limb)(left_value < right_value);
    }
#if defined(__x86_64__)
    if (count == 2U) {
        unsigned char borrow;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "sub (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "sbb 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "setc %b[borrow]\n\t"
            : [borrow] "=&q"(borrow)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)borrow;
    }
    if (count == 3U) {
        unsigned char borrow;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "sub (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "sbb 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "mov 16(%[left]), %%r8\n\t"
            "sbb 16(%[right]), %%r8\n\t"
            "mov %%r8, 16(%[result])\n\t"
            "setc %b[borrow]\n\t"
            : [borrow] "=&q"(borrow)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)borrow;
    }
    if (count == 4U) {
        unsigned char borrow;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "sub (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "sbb 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "mov 16(%[left]), %%r8\n\t"
            "sbb 16(%[right]), %%r8\n\t"
            "mov %%r8, 16(%[result])\n\t"
            "mov 24(%[left]), %%r8\n\t"
            "sbb 24(%[right]), %%r8\n\t"
            "mov %%r8, 24(%[result])\n\t"
            "setc %b[borrow]\n\t"
            : [borrow] "=&q"(borrow)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)borrow;
    }
    if (count == 8U) {
        unsigned char borrow;
        __asm__ volatile(
            "mov (%[left]), %%r8\n\t"
            "sub (%[right]), %%r8\n\t"
            "mov %%r8, (%[result])\n\t"
            "mov 8(%[left]), %%r8\n\t"
            "sbb 8(%[right]), %%r8\n\t"
            "mov %%r8, 8(%[result])\n\t"
            "mov 16(%[left]), %%r8\n\t"
            "sbb 16(%[right]), %%r8\n\t"
            "mov %%r8, 16(%[result])\n\t"
            "mov 24(%[left]), %%r8\n\t"
            "sbb 24(%[right]), %%r8\n\t"
            "mov %%r8, 24(%[result])\n\t"
            "mov 32(%[left]), %%r8\n\t"
            "sbb 32(%[right]), %%r8\n\t"
            "mov %%r8, 32(%[result])\n\t"
            "mov 40(%[left]), %%r8\n\t"
            "sbb 40(%[right]), %%r8\n\t"
            "mov %%r8, 40(%[result])\n\t"
            "mov 48(%[left]), %%r8\n\t"
            "sbb 48(%[right]), %%r8\n\t"
            "mov %%r8, 48(%[result])\n\t"
            "mov 56(%[left]), %%r8\n\t"
            "sbb 56(%[right]), %%r8\n\t"
            "mov %%r8, 56(%[result])\n\t"
            "setc %b[borrow]\n\t"
            : [borrow] "=&q"(borrow)
            : [left] "r"(left), [right] "r"(right), [result] "r"(result)
            : "r8", "cc", "memory");
        return (mtx_limb)borrow;
    }

    unsigned char borrow;
    size_t blocks;

    __asm__ volatile(
        "mov %[count], %[blocks]\n\t"
        "shr $2, %[blocks]\n\t"
        "and $3, %[count]\n\t"
        "test %[blocks], %[blocks]\n\t"
        "jz 3f\n\t"
        "clc\n\t"
        "1:\n\t"
        "mov (%[left]), %%r8\n\t"
        "sbb (%[right]), %%r8\n\t"
        "mov %%r8, (%[result])\n\t"
        "mov 8(%[left]), %%r8\n\t"
        "sbb 8(%[right]), %%r8\n\t"
        "mov %%r8, 8(%[result])\n\t"
        "mov 16(%[left]), %%r8\n\t"
        "sbb 16(%[right]), %%r8\n\t"
        "mov %%r8, 16(%[result])\n\t"
        "mov 24(%[left]), %%r8\n\t"
        "sbb 24(%[right]), %%r8\n\t"
        "mov %%r8, 24(%[result])\n\t"
        "lea 32(%[left]), %[left]\n\t"
        "lea 32(%[right]), %[right]\n\t"
        "lea 32(%[result]), %[result]\n\t"
        "dec %[blocks]\n\t"
        "jnz 1b\n\t"
        "setc %b[borrow]\n\t"
        "test %[count], %[count]\n\t"
        "jz 5f\n\t"
        "add $255, %b[borrow]\n\t"
        "2:\n\t"
        "mov (%[left]), %%r8\n\t"
        "sbb (%[right]), %%r8\n\t"
        "mov %%r8, (%[result])\n\t"
        "lea 8(%[left]), %[left]\n\t"
        "lea 8(%[right]), %[right]\n\t"
        "lea 8(%[result]), %[result]\n\t"
        "dec %[count]\n\t"
        "jnz 2b\n\t"
        "setc %b[borrow]\n\t"
        "jmp 5f\n\t"
        "3:\n\t"
        "test %[count], %[count]\n\t"
        "jz 4f\n\t"
        "clc\n\t"
        "jmp 2b\n\t"
        "4:\n\t"
        "xor %k[borrow], %k[borrow]\n\t"
        "5:"
        : [borrow] "=&q"(borrow), [result] "+r"(result),
          [left] "+r"(left), [right] "+r"(right), [count] "+r"(count),
          [blocks] "=&r"(blocks)
        :
        : "r8", "cc", "memory");
    return (mtx_limb)borrow;
#elif defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    if (count == 2U) {
        mtx_limb r0, r1, borrow;
        __asm__ volatile(
            "subs %0, %3, %5\n\t"
            "sbcs %1, %4, %6\n\t"
            "cset %2, cc\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(borrow)
            : "r"(left[0]), "r"(left[1]), "r"(right[0]), "r"(right[1])
            : "cc"
        );
        result[0] = r0; result[1] = r1;
        return borrow;
    }
    if (count == 3U) {
        mtx_limb r0, r1, r2, borrow;
        __asm__ volatile(
            "subs %0, %4, %7\n\t"
            "sbcs %1, %5, %8\n\t"
            "sbcs %2, %6, %9\n\t"
            "cset %3, cc\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(borrow)
            : "r"(left[0]), "r"(left[1]), "r"(left[2]),
              "r"(right[0]), "r"(right[1]), "r"(right[2])
            : "cc"
        );
        result[0] = r0; result[1] = r1; result[2] = r2;
        return borrow;
    }
    if (count == 4U) {
        mtx_limb r0, r1, r2, r3, borrow;
        __asm__ volatile(
            "subs %0, %5, %9\n\t"
            "sbcs %1, %6, %10\n\t"
            "sbcs %2, %7, %11\n\t"
            "sbcs %3, %8, %12\n\t"
            "cset %4, cc\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3), "=&r"(borrow)
            : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]),
              "r"(right[0]), "r"(right[1]), "r"(right[2]), "r"(right[3])
            : "cc"
        );
        result[0] = r0; result[1] = r1; result[2] = r2; result[3] = r3;
        return borrow;
    }
    if (count == 8U) {
        mtx_limb r0, r1, r2, r3, r4, r5, r6, r7, borrow;
        __asm__ volatile(
            "subs %0, %9, %17\n\t"
            "sbcs %1, %10, %18\n\t"
            "sbcs %2, %11, %19\n\t"
            "sbcs %3, %12, %20\n\t"
            "sbcs %4, %13, %21\n\t"
            "sbcs %5, %14, %22\n\t"
            "sbcs %6, %15, %23\n\t"
            "sbcs %7, %16, %24\n\t"
            "cset %8, cc\n\t"
            : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3),
              "=&r"(r4), "=&r"(r5), "=&r"(r6), "=&r"(r7), "=&r"(borrow)
            : "r"(left[0]), "r"(left[1]), "r"(left[2]), "r"(left[3]),
              "r"(left[4]), "r"(left[5]), "r"(left[6]), "r"(left[7]),
              "r"(right[0]), "r"(right[1]), "r"(right[2]), "r"(right[3]),
              "r"(right[4]), "r"(right[5]), "r"(right[6]), "r"(right[7])
            : "cc"
        );
        result[0] = r0; result[1] = r1; result[2] = r2; result[3] = r3;
        result[4] = r4; result[5] = r5; result[6] = r6; result[7] = r7;
        return borrow;
    }
    mtx_limb borrow = 0U;
    for (size_t i = 0U; i < count; ++i) {
        mtx_limb partial = left[i] - borrow;
        mtx_limb borrow_left = (mtx_limb)(left[i] < borrow);
        mtx_limb difference = partial - right[i];
        mtx_limb borrow_right = (mtx_limb)(partial < right[i]);
        result[i] = difference;
        borrow = borrow_left | borrow_right;
    }
    return borrow;
#else
    mtx_limb borrow = 0U;

    for (size_t i = 0U; i < count; ++i) {
        mtx_limb partial = left[i] - borrow;
        mtx_limb borrow_left = (mtx_limb)(left[i] < borrow);
        mtx_limb difference = partial - right[i];
        mtx_limb borrow_right = (mtx_limb)(partial < right[i]);
        result[i] = difference;
        borrow = borrow_left | borrow_right;
    }
    return borrow;
#endif
}

mtx_limb mtx_limb_lshift(mtx_limb *result, const mtx_limb *src,
                         size_t count, unsigned shift)
{
    if (count == 0U) {
        return 0U;
    }
    if (__builtin_expect(shift == 0U, 0)) {
        if (result != src) {
            memmove(result, src, count * sizeof(mtx_limb));
        }
        return 0U;
    }
    unsigned rshift = 64U - shift;
    mtx_limb carry = src[count - 1U] >> rshift;

#if (defined(__x86_64__) || defined(__aarch64__)) && (defined(__clang__) || defined(__GNUC__))
    if (count == 1U) {
        result[0] = src[0] << shift;
        return carry;
    }
    if (count == 2U) {
        mtx_limb s0 = src[0], s1 = src[1];
        result[1] = (s1 << shift) | (s0 >> rshift);
        result[0] = s0 << shift;
        return carry;
    }
    if (count == 4U) {
        mtx_limb s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3];
        result[3] = (s3 << shift) | (s2 >> rshift);
        result[2] = (s2 << shift) | (s1 >> rshift);
        result[1] = (s1 << shift) | (s0 >> rshift);
        result[0] = s0 << shift;
        return carry;
    }
#endif

    for (size_t i = count - 1U; i > 0U; --i) {
        result[i] = (src[i] << shift) | (src[i - 1U] >> rshift);
    }
    result[0] = src[0] << shift;
    return carry;
}

mtx_limb mtx_limb_rshift(mtx_limb *result, const mtx_limb *src,
                         size_t count, unsigned shift)
{
    if (__builtin_expect(count == 0U, 0)) {
        return 0U;
    }
    if (__builtin_expect(shift == 0U, 0)) {
        if (result != src) {
            memmove(result, src, count * sizeof(mtx_limb));
        }
        return 0U;
    }
    unsigned lshift = 64U - shift;
    mtx_limb dropped = src[0] & ((UINT64_C(1) << shift) - 1U);

#if (defined(__x86_64__) || defined(__aarch64__)) && (defined(__clang__) || defined(__GNUC__))
    if (count == 1U) {
        result[0] = src[0] >> shift;
        return dropped;
    }
    if (count == 2U) {
        mtx_limb s0 = src[0], s1 = src[1];
        result[0] = (s0 >> shift) | (s1 << lshift);
        result[1] = s1 >> shift;
        return dropped;
    }
    if (count == 4U) {
        mtx_limb s0 = src[0], s1 = src[1], s2 = src[2], s3 = src[3];
        result[0] = (s0 >> shift) | (s1 << lshift);
        result[1] = (s1 >> shift) | (s2 << lshift);
        result[2] = (s2 >> shift) | (s3 << lshift);
        result[3] = s3 >> shift;
        return dropped;
    }
#endif

    for (size_t i = 0U; i < count - 1U; ++i) {
        result[i] = (src[i] >> shift) | (src[i + 1U] << lshift);
    }
    result[count - 1U] = src[count - 1U] >> shift;
    return dropped;
}

int mtx_limb_cmp_n(const mtx_limb *left, const mtx_limb *right, size_t count)
{
    for (size_t i = count; i > 0U; --i) {
        mtx_limb a = left[i - 1U];
        mtx_limb b = right[i - 1U];
        if (a > b) {
            return 1;
        }
        if (a < b) {
            return -1;
        }
    }
    return 0;
}

void mtx_limb_div_qr(mtx_limb *quotient, mtx_limb *remainder,
                     const mtx_limb *dividend, size_t d_count,
                     const mtx_limb *divisor, size_t v_count)
{
    while (v_count > 0U && divisor[v_count - 1U] == 0U) {
        --v_count;
    }
    while (d_count > 0U && dividend[d_count - 1U] == 0U) {
        --d_count;
    }

    if (v_count == 0U) {
        return;
    }

    if (d_count < v_count) {
        if (quotient != NULL) {
            quotient[0] = 0U;
        }
        if (remainder != NULL) {
            memcpy(remainder, dividend, d_count * sizeof(mtx_limb));
            memset(remainder + d_count, 0, (v_count - d_count) * sizeof(mtx_limb));
        }
        return;
    }

    /* Single limb divisor fast-path */
    if (v_count == 1U) {
        mtx_limb v0 = divisor[0];
        mtx_limb rem = 0U;
        for (size_t i = d_count; i > 0U; --i) {
#if defined(__SIZEOF_INT128__)
            __uint128_t num = ((__uint128_t)rem << 64U) | dividend[i - 1U];
            if (quotient != NULL) {
                quotient[i - 1U] = (mtx_limb)(num / v0);
            }
            rem = (mtx_limb)(num % v0);
#else
            mtx_limb hi = rem;
            mtx_limb lo = dividend[i - 1U];
            mtx_limb q = 0U;
            /* 64-bit step division */
            for (int b = 63; b >= 0; --b) {
                hi = (hi << 1U) | (lo >> 63U);
                lo <<= 1U;
                if (hi >= v0) {
                    hi -= v0;
                    q |= (UINT64_C(1) << b);
                }
            }
            if (quotient != NULL) {
                quotient[i - 1U] = q;
            }
            rem = hi;
#endif
        }
        if (remainder != NULL) {
            remainder[0] = rem;
        }
        return;
    }

    /* Multi-limb Knuth Algorithm D */
    unsigned shift = (unsigned)__builtin_clzll(divisor[v_count - 1U]);

    mtx_limb stack_u[128];
    mtx_limb stack_v[64];
    mtx_limb *u = stack_u;
    mtx_limb *v = stack_v;

    size_t u_len = d_count + 1U;
    if (u_len > sizeof(stack_u) / sizeof(stack_u[0])) {
        u = malloc(u_len * sizeof(mtx_limb));
        if (u == NULL) return;
    }
    if (v_count > sizeof(stack_v) / sizeof(stack_v[0])) {
        v = malloc(v_count * sizeof(mtx_limb));
        if (v == NULL) {
            if (u != stack_u) free(u);
            return;
        }
    }

    if (shift == 0U) {
        memcpy(v, divisor, v_count * sizeof(mtx_limb));
        memcpy(u, dividend, d_count * sizeof(mtx_limb));
        u[d_count] = 0U;
    } else {
        mtx_limb v_carry = mtx_limb_lshift(v, divisor, v_count, shift);
        (void)v_carry;
        mtx_limb u_carry = mtx_limb_lshift(u, dividend, d_count, shift);
        u[d_count] = u_carry;
    }

    mtx_limb v_top1 = v[v_count - 1U];
    mtx_limb v_top2 = v[v_count - 2U];

    for (size_t j = d_count - v_count + 1U; j > 0U; --j) {
        size_t idx = j - 1U;
        mtx_limb u_top = u[idx + v_count];
        mtx_limb u_mid = u[idx + v_count - 1U];
        mtx_limb u_bot = u[idx + v_count - 2U];

        mtx_limb q_hat;
        mtx_limb r_hat;

        if (u_top == v_top1) {
            q_hat = UINT64_MAX;
            r_hat = u_top + u_mid;
            if (r_hat < u_top) {
                /* Overflow: r_hat >= 2^64 */
            } else {
#if defined(__SIZEOF_INT128__)
                while ((__uint128_t)q_hat * v_top2 > (((__uint128_t)r_hat << 64U) | u_bot)) {
                    --q_hat;
                    r_hat += v_top1;
                    if (r_hat < v_top1) break;
                }
#endif
            }
        } else {
#if defined(__SIZEOF_INT128__)
            __uint128_t num = ((__uint128_t)u_top << 64U) | u_mid;
            q_hat = (mtx_limb)(num / v_top1);
            r_hat = (mtx_limb)(num % v_top1);
            while ((__uint128_t)q_hat * v_top2 > (((__uint128_t)r_hat << 64U) | u_bot)) {
                --q_hat;
                r_hat += v_top1;
                if (r_hat < v_top1) break;
            }
#else
            q_hat = u_mid / v_top1;
#endif
        }

        /* Multiply and subtract: u[idx .. idx + v_count] -= q_hat * v[0 .. v_count - 1] */
        mtx_limb carry = 0U;
        for (size_t k = 0U; k < v_count; ++k) {
#if defined(__SIZEOF_INT128__)
            __uint128_t prod = (__uint128_t)q_hat * v[k] + carry;
            mtx_limb prod_low = (mtx_limb)prod;
            carry = (mtx_limb)(prod >> 64U);
            mtx_limb cur = u[idx + k];
            u[idx + k] = cur - prod_low;
            if (cur < prod_low) {
                ++carry;
            }
#else
            u[idx + k] -= q_hat * v[k];
#endif
        }

        if (u[idx + v_count] < carry) {
            u[idx + v_count] -= carry;
            /* Add back */
            --q_hat;
            mtx_limb add_carry = 0U;
            for (size_t k = 0U; k < v_count; ++k) {
                mtx_limb sum = u[idx + k] + v[k] + add_carry;
                add_carry = (mtx_limb)(sum < u[idx + k] || (add_carry && sum == u[idx + k]));
                u[idx + k] = sum;
            }
            u[idx + v_count] += add_carry;
        } else {
            u[idx + v_count] -= carry;
        }

        if (quotient != NULL) {
            quotient[idx] = q_hat;
        }
    }

    if (remainder != NULL) {
        if (shift == 0U) {
            memcpy(remainder, u, v_count * sizeof(mtx_limb));
        } else {
            mtx_limb dropped = mtx_limb_rshift(remainder, u, v_count, shift);
            (void)dropped;
        }
    }

    if (u != stack_u) free(u);
    if (v != stack_v) free(v);
}

void mtx_limb_sqrt(mtx_limb *root, const mtx_limb *num, size_t count)
{
    while (count > 0U && num[count - 1U] == 0U) {
        --count;
    }
    if (count == 0U) {
        root[0] = 0U;
        return;
    }

    size_t root_len = (count + 1U) / 2U;
    memset(root, 0, root_len * sizeof(mtx_limb));

    if (count == 1U) {
        mtx_limb s = (mtx_limb)sqrt((double)num[0]);
        while ((__uint128_t)(s + 1U) * (s + 1U) <= num[0]) {
            ++s;
        }
        while ((__uint128_t)s * s > num[0]) {
            --s;
        }
        root[0] = s;
        return;
    }

    size_t top_limb_idx = (count - 1U) / 2U;
    if (count % 2 == 0) {
        double top_val = (double)num[count - 1U] * 18446744073709551616.0 + (double)num[count - 2U];
        double r_est = sqrt(top_val);
        uint64_t u_est = (uint64_t)r_est;
        root[top_limb_idx] = u_est;
    } else {
        double top_val = (double)num[count - 1U] * 18446744073709551616.0;
        if (count >= 2U) {
            top_val += (double)num[count - 2U];
        }
        double r_est = sqrt(top_val);
        uint64_t u_est = (uint64_t)r_est;
        root[top_limb_idx] = u_est >> 32U;
        if (top_limb_idx >= 1U) {
            root[top_limb_idx - 1U] = u_est << 32U;
        }
    }
    if (root[top_limb_idx] == 0U) {
        root[top_limb_idx] = 1U;
    }

    /* Newton iterations: doubles precision from 53 bits each step */
    mtx_limb stack_buf[256];
    mtx_limb *q_buf = stack_buf;
    mtx_limb *rem_buf = stack_buf + (count + 2U);
    size_t buf_needed = (count + 2U) * 2U;
    if (buf_needed > sizeof(stack_buf) / sizeof(stack_buf[0])) {
        q_buf = malloc(buf_needed * sizeof(mtx_limb));
        if (q_buf == NULL) return;
        rem_buf = q_buf + (count + 2U);
    }

    int iters = (root_len <= 2U) ? 2 : ((root_len <= 4U) ? 3 : ((root_len <= 8U) ? 4 : 5));
    for (int iter = 0; iter < iters; ++iter) {
        size_t cur_root_len = root_len;
        while (cur_root_len > 0U && root[cur_root_len - 1U] == 0U) {
            --cur_root_len;
        }
        if (cur_root_len == 0U) break;

        memset(q_buf, 0, (count + 2U) * sizeof(mtx_limb));
        mtx_limb_div_qr(q_buf, rem_buf, num, count, root, cur_root_len);

        /* Check if root == q_buf (exact convergence) */
        bool converged = true;
        for (size_t i = 0U; i < root_len; ++i) {
            if (root[i] != q_buf[i]) {
                converged = false;
                break;
            }
        }

        /* root = (root + q_buf) / 2 */
        mtx_limb carry = mtx_limb_add_n(root, root, q_buf, root_len);
        mtx_limb_rshift(root, root, root_len, 1U);
        if (carry != 0U) {
            root[root_len - 1U] |= (UINT64_C(1) << 63U);
        }

        if (converged) break;
    }

    if (q_buf != stack_buf) free(q_buf);
}
