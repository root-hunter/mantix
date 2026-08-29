#include "mantix/mantix.h"
#include "mantix/limb.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

__attribute__((noinline))
static mtx_status mtx_reserve(mtx_float *x, size_t capacity)
{
    mtx_limb *new_limbs;

    if (capacity <= x->capacity) {
        return MTX_OK;
    }
    if (capacity > SIZE_MAX / sizeof(*x->limbs)) {
        return MTX_ERROR_OUT_OF_MEMORY;
    }

    if (x->limbs == &x->inline_limb) {
        new_limbs = malloc(capacity * sizeof(*x->limbs));
        if (new_limbs != NULL && x->used != 0U) {
            memcpy(new_limbs, x->limbs, x->used * sizeof(*x->limbs));
        }
    } else {
        new_limbs = realloc(x->limbs, capacity * sizeof(*x->limbs));
    }
    if (new_limbs == NULL) {
        return MTX_ERROR_OUT_OF_MEMORY;
    }

    x->limbs = new_limbs;
    x->capacity = capacity;
    return MTX_OK;
}

static inline unsigned mtx_trailing_zeroes_u64(uint64_t value)
{
#if defined(__clang__) || defined(__GNUC__)
    return value == 0U ? 64U : (unsigned)__builtin_ctzll(value);
#else
    if (value == 0U) return 64U;
    unsigned count = 0U;
    while ((value & UINT64_C(1)) == 0U) {
        value >>= 1U;
        ++count;
    }
    return count;
#endif
}

static inline unsigned mtx_leading_zeroes_u64(uint64_t value)
{
#if defined(__clang__) || defined(__GNUC__)
    return value == 0U ? 64U : (unsigned)__builtin_clzll(value);
#else
    if (value == 0U) return 64U;
    unsigned count = 0U;
    while ((value & (UINT64_C(1) << 63U)) == 0U) {
        value <<= 1U;
        ++count;
    }
    return count;
#endif
}

static inline size_t mtx_count_bits(const mtx_limb *limbs, size_t used)
{
    while (used > 0U && limbs[used - 1U] == 0U) {
        --used;
    }
    if (used == 0U) {
        return 0U;
    }
    return (used - 1U) * 64U + (64U - mtx_leading_zeroes_u64(limbs[used - 1U]));
}

mtx_status mtx_init(mtx_float *x, size_t precision)
{
    if (x == NULL || precision == 0U) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }

    *x = (mtx_float){0};
    x->limbs = &x->inline_limb;
    x->capacity = 1U;
    x->precision = precision;
    return MTX_OK;
}

void mtx_clear(mtx_float *x)
{
    if (x == NULL) {
        return;
    }

    if (x->limbs != &x->inline_limb) {
        free(x->limbs);
    }
    *x = (mtx_float){0};
}

void mtx_set_zero(mtx_float *x)
{
    if (x == NULL) {
        return;
    }

    x->used = 0U;
    x->exponent = 0;
    x->negative = false;
}

mtx_status mtx_set_u64(mtx_float *x, uint64_t value)
{
    if (__builtin_expect(x == NULL || x->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (__builtin_expect(value == 0U, 0)) {
        mtx_set_zero(x);
        return MTX_OK;
    }

    if (__builtin_expect(x->capacity == 0U, 0)) {
        mtx_status status = mtx_reserve(x, 1U);
        if (status != MTX_OK) {
            return status;
        }
    }

    if ((value & UINT64_C(1)) != 0U) {
        x->limbs[0] = value;
        x->used = 1U;
        x->exponent = 0;
        x->negative = false;
        return MTX_OK;
    }

    unsigned shift = mtx_trailing_zeroes_u64(value);
    x->limbs[0] = value >> shift;
    x->used = 1U;
    x->exponent = (int64_t)shift;
    x->negative = false;
    return MTX_OK;
}

mtx_status mtx_set_i64(mtx_float *x, int64_t value)
{
    if (value >= 0) {
        return mtx_set_u64(x, (uint64_t)value);
    }
    uint64_t u = (uint64_t)(-(value + 1)) + 1U;
    mtx_status st = mtx_set_u64(x, u);
    if (st == MTX_OK) {
        x->negative = true;
    }
    return st;
}

mtx_status mtx_set_f32(mtx_float *x, float value)
{
    if (__builtin_expect(x == NULL || x->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bool neg = (bits >> 31U) != 0U;
    uint32_t exp_field = (bits >> 23U) & 0xFFU;
    uint32_t mant_field = bits & 0x7FFFFFU;

    if (exp_field == 255U) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (exp_field == 0U && mant_field == 0U) {
        mtx_set_zero(x);
        return MTX_OK;
    }

    uint64_t significand;
    int64_t exponent;
    if (exp_field == 0U) {
        significand = mant_field;
        exponent = -126 - 23;
    } else {
        significand = (UINT64_C(1) << 23U) | mant_field;
        exponent = (int64_t)exp_field - 127 - 23;
    }

    if (__builtin_expect(x->capacity == 0U, 0)) {
        mtx_status st = mtx_reserve(x, 1U);
        if (st != MTX_OK) return st;
    }

    unsigned shift = mtx_trailing_zeroes_u64(significand);
    x->limbs[0] = significand >> shift;
    x->used = 1U;
    x->exponent = exponent + (int64_t)shift;
    x->negative = neg;
    return MTX_OK;
}

mtx_status mtx_set_f64(mtx_float *x, double value)
{
    if (__builtin_expect(x == NULL || x->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    uint64_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bool neg = (bits >> 63U) != 0U;
    uint64_t exp_field = (bits >> 52U) & 0x7FFU;
    uint64_t mant_field = bits & UINT64_C(0xFFFFFFFFFFFFF);

    if (exp_field == 2047U) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (exp_field == 0U && mant_field == 0U) {
        mtx_set_zero(x);
        return MTX_OK;
    }

    uint64_t significand;
    int64_t exponent;
    if (exp_field == 0U) {
        significand = mant_field;
        exponent = -1022 - 52;
    } else {
        significand = (UINT64_C(1) << 52U) | mant_field;
        exponent = (int64_t)exp_field - 1023 - 52;
    }

    if (__builtin_expect(x->capacity == 0U, 0)) {
        mtx_status st = mtx_reserve(x, 1U);
        if (st != MTX_OK) return st;
    }

    unsigned shift = mtx_trailing_zeroes_u64(significand);
    x->limbs[0] = significand >> shift;
    x->used = 1U;
    x->exponent = exponent + (int64_t)shift;
    x->negative = neg;
    return MTX_OK;
}

mtx_status mtx_set_d(mtx_float *x, double value)
{
    return mtx_set_f64(x, value);
}

mtx_status mtx_set(mtx_float *dst, const mtx_float *src)
{
    if (__builtin_expect(dst == NULL || src == NULL || dst->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (__builtin_expect(dst == src, 0)) {
        return MTX_OK;
    }

    size_t used = src->used;
    if (__builtin_expect(used > dst->capacity, 0)) {
        mtx_status status = mtx_reserve(dst, used);
        if (status != MTX_OK) {
            return status;
        }
    }
    if (used != 0U) {
        if (used == 1U) {
            dst->limbs[0] = src->limbs[0];
        } else if (used == 2U) {
            dst->limbs[0] = src->limbs[0];
            dst->limbs[1] = src->limbs[1];
        } else if (used == 3U) {
            dst->limbs[0] = src->limbs[0];
            dst->limbs[1] = src->limbs[1];
            dst->limbs[2] = src->limbs[2];
        } else if (used == 4U) {
            dst->limbs[0] = src->limbs[0];
            dst->limbs[1] = src->limbs[1];
            dst->limbs[2] = src->limbs[2];
            dst->limbs[3] = src->limbs[3];
        } else {
            memcpy(dst->limbs, src->limbs, used * sizeof(*src->limbs));
        }
    }

    dst->used = used;
    dst->exponent = src->exponent;
    dst->negative = src->negative;
    return MTX_OK;
}

mtx_status mtx_neg(mtx_float *r, const mtx_float *a)
{
    mtx_status st = mtx_set(r, a);
    if (st == MTX_OK && !mtx_is_zero(r)) {
        r->negative = !r->negative;
    }
    return st;
}

mtx_status mtx_abs(mtx_float *r, const mtx_float *a)
{
    mtx_status st = mtx_set(r, a);
    if (st == MTX_OK) {
        r->negative = false;
    }
    return st;
}

bool mtx_is_zero(const mtx_float *x)
{
    return x != NULL && x->used == 0U;
}

bool mtx_is_normalized(const mtx_float *x)
{
    if (x == NULL || x->precision == 0U || x->used > x->capacity) {
        return false;
    }
    if (x->used == 0U) {
        return x->exponent == 0 && !x->negative;
    }
    if (x->limbs == NULL || x->limbs[x->used - 1U] == 0U) {
        return false;
    }
    return (x->limbs[0] & UINT64_C(1)) != 0U;
}

int mtx_cmp(const mtx_float *a, const mtx_float *b)
{
    if (a == NULL || b == NULL) {
        return 0;
    }
    bool a_zero = mtx_is_zero(a);
    bool b_zero = mtx_is_zero(b);
    if (a_zero && b_zero) {
        return 0;
    }
    if (a_zero) {
        return b->negative ? 1 : -1;
    }
    if (b_zero) {
        return a->negative ? -1 : 1;
    }
    if (a->negative != b->negative) {
        return a->negative ? -1 : 1;
    }

    int sign = a->negative ? -1 : 1;

    size_t a_bits = mtx_count_bits(a->limbs, a->used);
    size_t b_bits = mtx_count_bits(b->limbs, b->used);

    int64_t a_msb = a->exponent + (int64_t)a_bits - 1;
    int64_t b_msb = b->exponent + (int64_t)b_bits - 1;

    if (a_msb > b_msb) return 1 * sign;
    if (a_msb < b_msb) return -1 * sign;

    int64_t min_exp = a->exponent < b->exponent ? a->exponent : b->exponent;
    size_t a_shift = (size_t)(a->exponent - min_exp);
    size_t b_shift = (size_t)(b->exponent - min_exp);

    size_t max_limbs = (a->used + a_shift / 64U + 2U > b->used + b_shift / 64U + 2U ?
                        a->used + a_shift / 64U + 2U : b->used + b_shift / 64U + 2U);

    mtx_limb stack_a[64];
    mtx_limb stack_b[64];
    mtx_limb *buf_a = stack_a;
    mtx_limb *buf_b = stack_b;

    if (max_limbs > 64U) {
        buf_a = calloc(max_limbs, sizeof(mtx_limb));
        buf_b = calloc(max_limbs, sizeof(mtx_limb));
        if (buf_a == NULL || buf_b == NULL) {
            free(buf_a); free(buf_b);
            return 0;
        }
    } else {
        memset(stack_a, 0, max_limbs * sizeof(mtx_limb));
        memset(stack_b, 0, max_limbs * sizeof(mtx_limb));
    }

    size_t a_shift_limbs = a_shift / 64U;
    unsigned a_shift_rem = (unsigned)(a_shift % 64U);
    if (a_shift_rem == 0U) {
        memcpy(buf_a + a_shift_limbs, a->limbs, a->used * sizeof(mtx_limb));
    } else {
        mtx_limb c = mtx_limb_lshift(buf_a + a_shift_limbs, a->limbs, a->used, a_shift_rem);
        buf_a[a_shift_limbs + a->used] = c;
    }

    size_t b_shift_limbs = b_shift / 64U;
    unsigned b_shift_rem = (unsigned)(b_shift % 64U);
    if (b_shift_rem == 0U) {
        memcpy(buf_b + b_shift_limbs, b->limbs, b->used * sizeof(mtx_limb));
    } else {
        mtx_limb c = mtx_limb_lshift(buf_b + b_shift_limbs, b->limbs, b->used, b_shift_rem);
        buf_b[b_shift_limbs + b->used] = c;
    }

    int cmp = mtx_limb_cmp_n(buf_a, buf_b, max_limbs);
    if (buf_a != stack_a) {
        free(buf_a);
        free(buf_b);
    }
    return cmp * sign;
}

static mtx_status mtx_round_and_canonicalize(mtx_float *r, mtx_limb *limbs,
                                             size_t used, int64_t exponent,
                                             bool negative, mtx_rounding rnd)
{
    while (used > 0U && limbs[used - 1U] == 0U) {
        --used;
    }
    if (used == 0U) {
        mtx_set_zero(r);
        return MTX_OK;
    }

    size_t bits = (used - 1U) * 64U + (64U - mtx_leading_zeroes_u64(limbs[used - 1U]));
    if (bits <= r->precision) {
        if ((limbs[0] & UINT64_C(1)) == 0U) {
            size_t tz_limbs = 0U;
            while (tz_limbs < used && limbs[tz_limbs] == 0U) {
                ++tz_limbs;
            }
            if (tz_limbs > 0U) {
                used -= tz_limbs;
                memmove(limbs, limbs + tz_limbs, used * sizeof(mtx_limb));
                exponent += (int64_t)(tz_limbs * 64U);
            }
            unsigned tz = mtx_trailing_zeroes_u64(limbs[0]);
            if (tz > 0U) {
                mtx_limb_rshift(limbs, limbs, used, tz);
                exponent += (int64_t)tz;
                while (used > 0U && limbs[used - 1U] == 0U) {
                    --used;
                }
            }
        }
        if (__builtin_expect(used > r->capacity, 0)) {
            mtx_status st = mtx_reserve(r, used);
            if (st != MTX_OK) return st;
        }
        if (used == 1U) {
            r->limbs[0] = limbs[0];
        } else if (used == 2U) {
            r->limbs[0] = limbs[0];
            r->limbs[1] = limbs[1];
        } else if (used == 3U) {
            r->limbs[0] = limbs[0];
            r->limbs[1] = limbs[1];
            r->limbs[2] = limbs[2];
        } else if (used == 4U) {
            r->limbs[0] = limbs[0];
            r->limbs[1] = limbs[1];
            r->limbs[2] = limbs[2];
            r->limbs[3] = limbs[3];
        } else {
            memcpy(r->limbs, limbs, used * sizeof(mtx_limb));
        }
        r->used = used;
        r->exponent = exponent;
        r->negative = negative;
        return MTX_OK;
    }
    if (bits > r->precision) {
        size_t drop = bits - r->precision;
        size_t drop_limbs = drop / 64U;
        unsigned drop_rem = (unsigned)(drop % 64U);

        bool guard = false;
        bool sticky = false;

        for (size_t i = 0U; i < drop_limbs; ++i) {
            if (limbs[i] != 0U) {
                sticky = true;
                break;
            }
        }

        if (drop_rem == 0U) {
            if (drop_limbs > 0U) {
                guard = (limbs[drop_limbs - 1U] >> 63U) & 1U;
                if ((limbs[drop_limbs - 1U] & ~(UINT64_C(1) << 63U)) != 0U) {
                    sticky = true;
                }
            }
        } else {
            mtx_limb part = limbs[drop_limbs];
            guard = (part >> (drop_rem - 1U)) & 1U;
            if (drop_rem > 1U && (part & ((UINT64_C(1) << (drop_rem - 1U)) - 1U)) != 0U) {
                sticky = true;
            }
        }

        size_t new_used = 0U;
        if (drop_limbs < used) {
            size_t remaining = used - drop_limbs;
            if (drop_rem == 0U) {
                memmove(limbs, limbs + drop_limbs, remaining * sizeof(mtx_limb));
            } else {
                mtx_limb_rshift(limbs, limbs + drop_limbs, remaining, drop_rem);
            }
            new_used = remaining;
            while (new_used > 0U && limbs[new_used - 1U] == 0U) {
                --new_used;
            }
        }
        used = new_used;
        exponent += (int64_t)drop;

        bool round_up = false;
        switch (rnd) {
        case MTX_ROUND_TO_NEAREST_EVEN:
            if (guard && (sticky || (used > 0U && (limbs[0] & 1U) != 0U))) {
                round_up = true;
            }
            break;
        case MTX_ROUND_TOWARD_ZERO:
            round_up = false;
            break;
        case MTX_ROUND_TOWARD_POSITIVE:
            if (!negative && (guard || sticky)) {
                round_up = true;
            }
            break;
        case MTX_ROUND_TOWARD_NEGATIVE:
            if (negative && (guard || sticky)) {
                round_up = true;
            }
            break;
        }

        if (round_up) {
            mtx_limb carry = 1U;
            for (size_t i = 0U; carry != 0U && i < used; ++i) {
                mtx_limb val = limbs[i];
                limbs[i] = val + carry;
                carry = (mtx_limb)(limbs[i] < val);
            }
            if (carry != 0U) {
                limbs[used] = carry;
                ++used;
            }
            if (mtx_count_bits(limbs, used) > r->precision) {
                mtx_limb_rshift(limbs, limbs, used, 1U);
                exponent += 1;
                while (used > 0U && limbs[used - 1U] == 0U) {
                    --used;
                }
            }
        }
    }

    while (used > 0U && limbs[used - 1U] == 0U) {
        --used;
    }
    if (used == 0U) {
        mtx_set_zero(r);
        return MTX_OK;
    }

    size_t tz_limbs = 0U;
    while (tz_limbs < used && limbs[tz_limbs] == 0U) {
        ++tz_limbs;
    }
    if (tz_limbs > 0U) {
        used -= tz_limbs;
        memmove(limbs, limbs + tz_limbs, used * sizeof(mtx_limb));
        exponent += (int64_t)(tz_limbs * 64U);
    }

    unsigned tz = mtx_trailing_zeroes_u64(limbs[0]);
    if (tz > 0U) {
        mtx_limb_rshift(limbs, limbs, used, tz);
        exponent += (int64_t)tz;
        while (used > 0U && limbs[used - 1U] == 0U) {
            --used;
        }
    }

    if (used > r->capacity) {
        mtx_status st = mtx_reserve(r, used);
        if (st != MTX_OK) return st;
    }

    memcpy(r->limbs, limbs, used * sizeof(mtx_limb));
    r->used = used;
    r->exponent = exponent;
    r->negative = negative;
    return MTX_OK;
}

mtx_status mtx_mul(mtx_float *r, const mtx_float *a, const mtx_float *b,
                   mtx_rounding rnd)
{
    if (__builtin_expect(r == NULL || a == NULL || b == NULL || r->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (mtx_is_zero(a) || mtx_is_zero(b)) {
        mtx_set_zero(r);
        return MTX_OK;
    }

    bool neg = a->negative ^ b->negative;
    int64_t exp = a->exponent + b->exponent;

    if (a->used == 1U && b->used == 1U && r->precision <= 64U) {
#if defined(__SIZEOF_INT128__)
        __uint128_t prod = (__uint128_t)a->limbs[0] * b->limbs[0];
        mtx_limb prod_limbs[2] = {(mtx_limb)prod, (mtx_limb)(prod >> 64U)};
        size_t prod_used = prod_limbs[1] != 0U ? 2U : 1U;
        return mtx_round_and_canonicalize(r, prod_limbs, prod_used, exp, neg, rnd);
#endif
    }

    size_t count = a->used > b->used ? a->used : b->used;
    size_t alloc_limbs = count * 2U + 2U;
    mtx_limb stack_buf[130];
    mtx_limb *buf = stack_buf;
    if (alloc_limbs * 2U > sizeof(stack_buf) / sizeof(stack_buf[0])) {
        buf = malloc(alloc_limbs * 2U * sizeof(mtx_limb));
        if (buf == NULL) return MTX_ERROR_OUT_OF_MEMORY;
    }

    mtx_limb *left = buf;
    mtx_limb *right = buf + count;
    mtx_limb *product = buf + count * 2U;

    memset(left, 0, count * sizeof(mtx_limb));
    memset(right, 0, count * sizeof(mtx_limb));
    memcpy(left, a->limbs, a->used * sizeof(mtx_limb));
    memcpy(right, b->limbs, b->used * sizeof(mtx_limb));

    mtx_limb_mul_n(product, left, right, count);

    size_t prod_used = count * 2U;
    while (prod_used > 0U && product[prod_used - 1U] == 0U) {
        --prod_used;
    }

    mtx_status status = mtx_round_and_canonicalize(r, product, prod_used, exp, neg, rnd);
    if (buf != stack_buf) {
        free(buf);
    }
    return status;
}

mtx_status mtx_add(mtx_float *r, const mtx_float *a, const mtx_float *b,
                   mtx_rounding rnd)
{
    if (__builtin_expect(r == NULL || a == NULL || b == NULL || r->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (mtx_is_zero(a)) {
        mtx_status st = mtx_set(r, b);
        if (st != MTX_OK) return st;
        return mtx_round_and_canonicalize(r, r->limbs, r->used, r->exponent, r->negative, rnd);
    }
    if (mtx_is_zero(b)) {
        mtx_status st = mtx_set(r, a);
        if (st != MTX_OK) return st;
        return mtx_round_and_canonicalize(r, r->limbs, r->used, r->exponent, r->negative, rnd);
    }

    /* Equal exponent fast path */
    if (a->exponent == b->exponent && a->negative == b->negative) {
        size_t count = a->used > b->used ? a->used : b->used;
        mtx_limb stack_r[66];
        mtx_limb *res = stack_r;
        if (count + 1U > 66U) {
            res = malloc((count + 1U) * sizeof(mtx_limb));
            if (res == NULL) return MTX_ERROR_OUT_OF_MEMORY;
        }
        mtx_limb carry = 0U;
        if (a->used == b->used) {
            carry = mtx_limb_add_n(res, a->limbs, b->limbs, count);
        } else {
            const mtx_float *big = a->used > b->used ? a : b;
            const mtx_float *small = a->used > b->used ? b : a;
            carry = mtx_limb_add_n(res, big->limbs, small->limbs, small->used);
            for (size_t i = small->used; i < big->used; ++i) {
                mtx_limb v = big->limbs[i];
                res[i] = v + carry;
                carry = (mtx_limb)(res[i] < v);
            }
        }
        size_t res_used = count;
        if (carry != 0U) {
            res[count] = carry;
            res_used = count + 1U;
        }
        mtx_status st = mtx_round_and_canonicalize(r, res, res_used, a->exponent, a->negative, rnd);
        if (res != stack_r) free(res);
        return st;
    }

    const mtx_float *op_a = a;
    const mtx_float *op_b = b;
    if (op_a->exponent < op_b->exponent) {
        op_a = b;
        op_b = a;
    }

    uint64_t delta_exp = (uint64_t)(op_a->exponent - op_b->exponent);
    int64_t base_exp = op_b->exponent;

    bool effective_sub = (op_a->negative != op_b->negative);

    /* If exponent difference is larger than precision window, op_b is negligible */
    if (__builtin_expect(delta_exp > op_b->used * 64U + r->precision + 64U, 0)) {
        return mtx_set(r, op_a);
    }

    size_t shift_limbs = (size_t)(delta_exp / 64U);
    unsigned shift_rem = (unsigned)(delta_exp % 64U);

    size_t a_aligned_limbs = op_a->used + shift_limbs + 2U;
    size_t max_limbs = (a_aligned_limbs > op_b->used ? a_aligned_limbs : op_b->used) + 2U;

    mtx_limb stack_buf[200];
    mtx_limb *buf = stack_buf;

    if (max_limbs * 3U > sizeof(stack_buf) / sizeof(stack_buf[0])) {
        buf = malloc(max_limbs * 3U * sizeof(mtx_limb));
        if (buf == NULL) {
            return MTX_ERROR_OUT_OF_MEMORY;
        }
    }

    mtx_limb *buf_a = buf;
    mtx_limb *buf_b = buf + max_limbs;
    mtx_limb *buf_r = buf + max_limbs * 2U;

    memset(buf_a, 0, max_limbs * sizeof(mtx_limb));
    memset(buf_b, 0, max_limbs * sizeof(mtx_limb));

    if (shift_rem == 0U) {
        memcpy(buf_a + shift_limbs, op_a->limbs, op_a->used * sizeof(mtx_limb));
    } else {
        mtx_limb carry = mtx_limb_lshift(buf_a + shift_limbs, op_a->limbs, op_a->used, shift_rem);
        buf_a[shift_limbs + op_a->used] = carry;
    }
    memcpy(buf_b, op_b->limbs, op_b->used * sizeof(mtx_limb));

    size_t result_used = max_limbs;
    bool res_negative = op_a->negative;

    if (!effective_sub) {
        mtx_limb carry = mtx_limb_add_n(buf_r, buf_a, buf_b, max_limbs - 1U);
        buf_r[max_limbs - 1U] = carry;
    } else {
        int cmp = mtx_limb_cmp_n(buf_a, buf_b, max_limbs);
        if (cmp == 0) {
            mtx_set_zero(r);
            if (buf != stack_buf) free(buf);
            return MTX_OK;
        } else if (cmp > 0) {
            mtx_limb_sub_n(buf_r, buf_a, buf_b, max_limbs);
            res_negative = op_a->negative;
        } else {
            mtx_limb_sub_n(buf_r, buf_b, buf_a, max_limbs);
            res_negative = op_b->negative;
        }
    }

    while (result_used > 0U && buf_r[result_used - 1U] == 0U) {
        --result_used;
    }

    mtx_status status = mtx_round_and_canonicalize(r, buf_r, result_used, base_exp, res_negative, rnd);

    if (buf != stack_buf) {
        free(buf);
    }
    return status;
}

mtx_status mtx_sub(mtx_float *r, const mtx_float *a, const mtx_float *b,
                   mtx_rounding rnd)
{
    if (__builtin_expect(r == NULL || a == NULL || b == NULL || r->precision == 0U, 0)) {
        return MTX_ERROR_INVALID_ARGUMENT;
    }
    if (mtx_is_zero(b)) {
        mtx_status st = mtx_set(r, a);
        if (st != MTX_OK) return st;
        return mtx_round_and_canonicalize(r, r->limbs, r->used, r->exponent, r->negative, rnd);
    }
    mtx_float neg_b = *b;
    neg_b.negative = !b->negative;
    return mtx_add(r, a, &neg_b, rnd);
}

double mtx_get_f64(const mtx_float *x, mtx_rounding rnd)
{
    if (x == NULL || mtx_is_zero(x)) {
        return 0.0;
    }
    mtx_float rounded;
    if (mtx_init(&rounded, 53U) != MTX_OK) {
        return 0.0;
    }
    mtx_set(&rounded, x);
    mtx_round_and_canonicalize(&rounded, rounded.limbs, rounded.used, rounded.exponent, rounded.negative, rnd);

    double val = 0.0;
    for (size_t i = rounded.used; i > 0U; --i) {
        val = val * 18446744073709551616.0 + (double)rounded.limbs[i - 1U];
    }
    val = ldexp(val, (int)rounded.exponent);
    if (rounded.negative) {
        val = -val;
    }
    mtx_clear(&rounded);
    return val;
}

double mtx_get_d(const mtx_float *x, mtx_rounding rnd)
{
    return mtx_get_f64(x, rnd);
}

float mtx_get_f32(const mtx_float *x, mtx_rounding rnd)
{
    return (float)mtx_get_f64(x, rnd);
}

const char *mtx_status_string(mtx_status status)
{
    switch (status) {
    case MTX_OK:
        return "success";
    case MTX_ERROR_INVALID_ARGUMENT:
        return "invalid argument";
    case MTX_ERROR_OUT_OF_MEMORY:
        return "out of memory";
    case MTX_ERROR_EXPONENT_OVERFLOW:
        return "exponent overflow";
    default:
        return "unknown error";
    }
}
