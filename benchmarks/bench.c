#include "mantix/mantix.h"
#include "mantix/limb.h"

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef MANTIX_HAVE_GMP
#include <gmp.h>
#endif

#ifdef MANTIX_HAVE_MPFR
#include <mpfr.h>
#endif

enum { MAX_SAMPLES = 31 };

#define STRINGIFY_VALUE_(value) #value
#define STRINGIFY_VALUE(value) STRINGIFY_VALUE_(value)
#define BENCH_COMPILER_VERSION                                                 \
    "clang-" STRINGIFY_VALUE(__clang_major__) "."                            \
    STRINGIFY_VALUE(__clang_minor__) "." STRINGIFY_VALUE(__clang_patchlevel__)

#if defined(__x86_64__)
#define BENCH_ARCHITECTURE "x86_64"
#elif defined(__aarch64__)
#define BENCH_ARCHITECTURE "aarch64"
#else
#define BENCH_ARCHITECTURE "unknown"
#endif

typedef void (*bench_fn)(void *context, uint64_t iterations);

typedef struct benchmark {
    const char *library;
    const char *version;
    const char *name;
    size_t precision;
    bench_fn function;
    void *context;
} benchmark;

typedef struct options {
    uint64_t iterations;
    unsigned samples;
    bool csv;
    const char *output_path;
} options;

typedef struct mantix_context {
    mtx_float source;
    mtx_float destination;
    mtx_limb *add_left;
    mtx_limb *add_right;
    mtx_limb *add_result;
    mtx_limb *mul_result;
    size_t limb_count;
    size_t precision;
} mantix_context;

#ifdef MANTIX_HAVE_GMP
typedef struct gmp_context {
    mpf_t source;
    mpf_t destination;
    size_t precision;
} gmp_context;
#endif

#ifdef MANTIX_HAVE_MPFR
typedef struct mpfr_context {
    mpfr_t source;
    mpfr_t destination;
    size_t precision;
} mpfr_context;
#endif

static volatile uint64_t benchmark_sink;

static uint64_t input_at(uint64_t index)
{
    return (index * UINT64_C(0x9e3779b97f4a7c15)) | UINT64_C(1);
}

static double now_seconds(void)
{
    struct timespec time;

    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (double)time.tv_sec + (double)time.tv_nsec * 1.0e-9;
}

static int compare_double(const void *left, const void *right)
{
    const double a = *(const double *)left;
    const double b = *(const double *)right;
    return (a > b) - (a < b);
}

static uint64_t parse_u64(const char *text, const char *option)
{
    char *end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0U) {
        fprintf(stderr, "invalid value for %s: %s\n", option, text);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)value;
}

static options parse_options(int argc, char **argv)
{
    options result = {UINT64_C(100000), 7U, false, NULL};

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            result.iterations = parse_u64(argv[++i], "--iterations");
        } else if (strcmp(argv[i], "--samples") == 0 && i + 1 < argc) {
            uint64_t samples = parse_u64(argv[++i], "--samples");
            if (samples < 3U || samples > MAX_SAMPLES || (samples & 1U) == 0U) {
                fprintf(stderr, "--samples must be an odd number from 3 to %d\n",
                        MAX_SAMPLES);
                exit(EXIT_FAILURE);
            }
            result.samples = (unsigned)samples;
        } else if (strcmp(argv[i], "--csv") == 0) {
            result.csv = true;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            result.output_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            puts("usage: mantix_bench [--iterations N] [--samples ODD] "
                 "[--csv] [--output FILE]");
            exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            exit(EXIT_FAILURE);
        }
    }
    return result;
}

static void native_set(void *context, uint64_t iterations)
{
    volatile uint64_t value = 0U;
    (void)context;

    for (uint64_t i = 0U; i < iterations; ++i) {
        value = input_at(i);
    }
    benchmark_sink ^= value;
}

static void mantix_set(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        if (mtx_set_u64(&state->destination, input_at(i)) != MTX_OK) {
            abort();
        }
    }
    benchmark_sink ^= state->destination.limbs[0];
}

static void mantix_copy(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        if (mtx_set(&state->destination, &state->source) != MTX_OK) {
            abort();
        }
    }
    benchmark_sink ^= state->destination.limbs[state->destination.used - 1U];
}

static void mantix_lifecycle(void *context, uint64_t iterations)
{
    const mantix_context *state = context;
    uint64_t checksum = 0U;

    for (uint64_t i = 0U; i < iterations; ++i) {
        mtx_float value;
        if (mtx_init(&value, state->precision) != MTX_OK ||
            mtx_set_u64(&value, input_at(i)) != MTX_OK) {
            abort();
        }
        checksum ^= value.limbs[0];
        mtx_clear(&value);
    }
    benchmark_sink ^= checksum;
}

static void mantix_add_n(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    mtx_limb carry = 0U;

    for (uint64_t i = 0U; i < iterations; ++i) {
        carry ^= mtx_limb_add_n(state->add_result, state->add_left,
                                state->add_right, state->limb_count);
    }
    benchmark_sink ^= state->add_result[0] ^ carry;
}

static void mantix_mul_n(void *context, uint64_t iterations)
{
    mantix_context *state = context;

    for (uint64_t i = 0U; i < iterations; ++i) {
        mtx_limb_mul_n(state->mul_result, state->add_left,
                       state->add_right, state->limb_count);
    }
    benchmark_sink ^= state->mul_result[0] ^
                      state->mul_result[state->limb_count * 2U - 1U];
}

static void mantix_fp_add(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        if (mtx_add(&state->destination, &state->source, &state->destination,
                    MTX_ROUND_TO_NEAREST_EVEN) != MTX_OK) {
            abort();
        }
    }
    benchmark_sink ^= state->destination.limbs[0];
}

static void mantix_fp_mul(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        if (mtx_mul(&state->destination, &state->source, &state->destination,
                    MTX_ROUND_TO_NEAREST_EVEN) != MTX_OK) {
            abort();
        }
    }
    benchmark_sink ^= state->destination.limbs[0];
}

static void mantix_set_f32(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    float val = 123.456f;
    for (uint64_t i = 0U; i < iterations; ++i) {
        if (mtx_set_f32(&state->destination, val) != MTX_OK) {
            abort();
        }
    }
    benchmark_sink ^= state->destination.limbs[0];
}

static void mantix_context_init(mantix_context *state, size_t precision)
{
    size_t limbs = (precision + 63U) / 64U;

    state->precision = precision;
    state->limb_count = limbs;
    if (mtx_init(&state->source, precision) != MTX_OK ||
        mtx_init(&state->destination, precision) != MTX_OK) {
        abort();
    }
    state->source.limbs = malloc(limbs * sizeof(*state->source.limbs));
    state->destination.limbs = malloc(limbs * sizeof(*state->destination.limbs));
    if (state->source.limbs == NULL || state->destination.limbs == NULL) {
        abort();
    }
    state->add_left = malloc(limbs * sizeof(*state->add_left));
    state->add_right = malloc(limbs * sizeof(*state->add_right));
    state->add_result = malloc(limbs * sizeof(*state->add_result));
    state->mul_result = malloc(limbs * 2U * sizeof(*state->mul_result));
    if (state->add_left == NULL || state->add_right == NULL ||
        state->add_result == NULL || state->mul_result == NULL) {
        abort();
    }
    state->source.capacity = limbs;
    state->source.used = limbs;
    state->source.exponent = 0;
    state->destination.capacity = limbs;
    state->destination.used = limbs;
    state->destination.exponent = 0;
    for (size_t i = 0U; i < limbs; ++i) {
        state->source.limbs[i] = input_at((uint64_t)i);
        state->destination.limbs[i] = input_at((uint64_t)i);
        state->add_left[i] = input_at((uint64_t)(i * 2U));
        state->add_right[i] = input_at((uint64_t)(i * 2U + 1U));
    }
    if (!mtx_is_normalized(&state->source) || !mtx_is_normalized(&state->destination)) {
        abort();
    }
}

static void mantix_context_clear(mantix_context *state)
{
    free(state->mul_result);
    free(state->add_result);
    free(state->add_right);
    free(state->add_left);
    mtx_clear(&state->destination);
    mtx_clear(&state->source);
}

#ifdef MANTIX_HAVE_GMP
static void gmp_add_n(void *context, uint64_t iterations)
{
    mantix_context *state = context;
    mp_limb_t carry = 0U;

    _Static_assert(sizeof(mp_limb_t) == sizeof(mtx_limb),
                   "benchmark requires 64-bit GMP limbs");
    for (uint64_t i = 0U; i < iterations; ++i) {
        carry ^= mpn_add_n((mp_ptr)state->add_result,
                           (mp_srcptr)state->add_left,
                           (mp_srcptr)state->add_right,
                           (mp_size_t)state->limb_count);
    }
    benchmark_sink ^= state->add_result[0] ^ (uint64_t)carry;
}

static void gmp_mul_n(void *context, uint64_t iterations)
{
    mantix_context *state = context;

    for (uint64_t i = 0U; i < iterations; ++i) {
        mpn_mul_n((mp_ptr)state->mul_result, (mp_srcptr)state->add_left,
                  (mp_srcptr)state->add_right,
                  (mp_size_t)state->limb_count);
    }
    benchmark_sink ^= state->mul_result[0] ^
                      state->mul_result[state->limb_count * 2U - 1U];
}

static void gmp_set(void *context, uint64_t iterations)
{
    gmp_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        mpf_set_ui(state->destination, (unsigned long)input_at(i));
    }
    benchmark_sink ^= (uint64_t)mpf_get_ui(state->destination);
}

static void gmp_copy(void *context, uint64_t iterations)
{
    gmp_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        mpf_set(state->destination, state->source);
    }
    benchmark_sink ^= (uint64_t)mpf_get_ui(state->destination);
}

static void gmp_fp_add(void *context, uint64_t iterations)
{
    gmp_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        mpf_add(state->destination, state->source, state->destination);
    }
    benchmark_sink ^= (uint64_t)mpf_get_ui(state->destination);
}

static void gmp_fp_mul(void *context, uint64_t iterations)
{
    gmp_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        mpf_mul(state->destination, state->source, state->destination);
    }
    benchmark_sink ^= (uint64_t)mpf_get_ui(state->destination);
}

static void gmp_lifecycle(void *context, uint64_t iterations)
{
    const gmp_context *state = context;
    uint64_t checksum = 0U;

    for (uint64_t i = 0U; i < iterations; ++i) {
        mpf_t value;
        mpf_init2(value, (mp_bitcnt_t)state->precision);
        mpf_set_ui(value, (unsigned long)input_at(i));
        checksum ^= (uint64_t)mpf_get_ui(value);
        mpf_clear(value);
    }
    benchmark_sink ^= checksum;
}

static void gmp_context_init(gmp_context *state, size_t precision)
{
    mpz_t integer;

    state->precision = precision;
    mpf_init2(state->source, (mp_bitcnt_t)precision);
    mpf_init2(state->destination, (mp_bitcnt_t)precision);
    mpz_init(integer);
    mpz_setbit(integer, (mp_bitcnt_t)(precision - 1U));
    mpz_setbit(integer, 0U);
    mpf_set_z(state->source, integer);
    mpz_clear(integer);
}

static void gmp_context_clear(gmp_context *state)
{
    mpf_clear(state->destination);
    mpf_clear(state->source);
}
#endif

#ifdef MANTIX_HAVE_MPFR
static void mpfr_set_u64(void *context, uint64_t iterations)
{
    mpfr_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        (void)mpfr_set_uj(state->destination, (uintmax_t)input_at(i),
                          MPFR_RNDN);
    }
    benchmark_sink ^= (uint64_t)mpfr_get_uj(state->destination, MPFR_RNDN);
}

static void mpfr_set_f32(void *context, uint64_t iterations)
{
    mpfr_context *state = context;
    float val = 123.456f;
    for (uint64_t i = 0U; i < iterations; ++i) {
        (void)mpfr_set_flt(state->destination, val, MPFR_RNDN);
    }
    benchmark_sink ^= (uint64_t)mpfr_get_uj(state->destination, MPFR_RNDN);
}

static void mpfr_copy(void *context, uint64_t iterations)
{
    mpfr_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        (void)mpfr_set(state->destination, state->source, MPFR_RNDN);
    }
    benchmark_sink ^= (uint64_t)mpfr_get_uj(state->destination, MPFR_RNDN);
}

static void mpfr_fp_add(void *context, uint64_t iterations)
{
    mpfr_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        (void)mpfr_add(state->destination, state->source, state->destination,
                       MPFR_RNDN);
    }
    benchmark_sink ^= (uint64_t)mpfr_get_uj(state->destination, MPFR_RNDN);
}

static void mpfr_fp_mul(void *context, uint64_t iterations)
{
    mpfr_context *state = context;
    for (uint64_t i = 0U; i < iterations; ++i) {
        (void)mpfr_mul(state->destination, state->source, state->destination,
                       MPFR_RNDN);
    }
    benchmark_sink ^= (uint64_t)mpfr_get_uj(state->destination, MPFR_RNDN);
}

static void mpfr_lifecycle(void *context, uint64_t iterations)
{
    const mpfr_context *state = context;
    uint64_t checksum = 0U;

    for (uint64_t i = 0U; i < iterations; ++i) {
        mpfr_t value;
        mpfr_init2(value, (mpfr_prec_t)state->precision);
        (void)mpfr_set_uj(value, (uintmax_t)input_at(i), MPFR_RNDN);
        checksum ^= (uint64_t)mpfr_get_uj(value, MPFR_RNDN);
        mpfr_clear(value);
    }
    benchmark_sink ^= checksum;
}

static void mpfr_context_init(mpfr_context *state, size_t precision)
{
    state->precision = precision;
    mpfr_init2(state->source, (mpfr_prec_t)precision);
    mpfr_init2(state->destination, (mpfr_prec_t)precision);
    (void)mpfr_set_ui(state->source, 1U, MPFR_RNDN);
    (void)mpfr_mul_2ui(state->source, state->source,
                       (unsigned long)(precision - 1U), MPFR_RNDN);
    (void)mpfr_add_ui(state->source, state->source, 1U, MPFR_RNDN);
}

static void mpfr_context_clear(mpfr_context *state)
{
    mpfr_clear(state->destination);
    mpfr_clear(state->source);
}
#endif

static void run_benchmark(FILE *output, const benchmark *test,
                          const options *settings)
{
    double timings[MAX_SAMPLES];

    test->function(test->context, settings->iterations / 10U + 1U);
    for (unsigned sample = 0U; sample < settings->samples; ++sample) {
        double start = now_seconds();
        test->function(test->context, settings->iterations);
        timings[sample] = (now_seconds() - start) * 1.0e9 /
                          (double)settings->iterations;
    }
    qsort(timings, settings->samples, sizeof(timings[0]), compare_double);

    if (settings->csv) {
        fprintf(output, "%s,%s,%s,%s,%s,%s,%zu,%" PRIu64 ",%u,%.3f,%.3f\n",
                test->library, test->version, BENCH_COMPILER_VERSION,
                BENCH_ARCHITECTURE, mtx_limb_backend(), test->name,
                test->precision,
                settings->iterations, settings->samples,
                timings[settings->samples / 2U], timings[0]);
    } else {
        fprintf(output, "%-8s  %-18s  %14.3f  %14.3f\n",
                test->library, test->name,
                timings[settings->samples / 2U], timings[0]);
    }
}

static void print_table_header(FILE *output)
{
    fprintf(output, "%-8s  %-18s  %14s  %14s\n",
            "library", "benchmark", "median ns/op", "min ns/op");
    fputs("--------  ------------------  --------------  --------------\n",
          output);
}

static void print_cpu_features(FILE *output, uint32_t features)
{
    fputs("  cpu features :", output);
    if (features == MTX_CPU_FEATURE_NONE) {
        fputs(" none", output);
    }
    if ((features & MTX_CPU_FEATURE_BMI2) != 0U) {
        fputs(" BMI2", output);
    }
    if ((features & MTX_CPU_FEATURE_ADX) != 0U) {
        fputs(" ADX", output);
    }
    if ((features & MTX_CPU_FEATURE_AVX2) != 0U) {
        fputs(" AVX2", output);
    }
    fputc('\n', output);
}

int main(int argc, char **argv)
{
    static const size_t precisions[] = {
        64U, 128U, 256U, 512U, 1024U, 2048U, 4096U,
    };
    options settings = parse_options(argc, argv);
    FILE *output = stdout;

    if (settings.output_path != NULL) {
        output = fopen(settings.output_path, "w");
        if (output == NULL) {
            perror(settings.output_path);
            return EXIT_FAILURE;
        }
    }
    if (settings.csv) {
        fputs("library,version,compiler,architecture,mantix_backend,benchmark,"
              "precision_bits,iterations,samples,median_ns_per_op,"
              "min_ns_per_op\n", output);
    } else {
        fprintf(output, "Mantix benchmark suite\n\n"
                        "  version      : %s\n"
                        "  compiler     : %s\n"
                        "  architecture : %s\n"
                        "  backend      : %s\n",
                MANTIX_VERSION_STRING, BENCH_COMPILER_VERSION,
                BENCH_ARCHITECTURE, mtx_limb_backend());
        print_cpu_features(output, mtx_cpu_features());
        fprintf(output, "  workload     : %" PRIu64
                        " iterations, %u samples\n\n"
                        "Baseline\n",
                settings.iterations, settings.samples);
        print_table_header(output);
    }
    benchmark native = {"native", "c17", "set_u64_floor", 64U,
                        native_set, NULL};
    run_benchmark(output, &native, &settings);

    for (size_t i = 0U; i < sizeof(precisions) / sizeof(precisions[0]); ++i) {
        size_t precision = precisions[i];
        mantix_context mantix;

        if (!settings.csv) {
            size_t limbs = precision / 64U;
            fprintf(output, "\nPrecision: %zu bits (%zu limb%s)\n",
                    precision, limbs, limbs == 1U ? "" : "s");
            print_table_header(output);
        }
        mantix_context_init(&mantix, precision);
        benchmark mantix_tests[] = {
            {"mantix", MANTIX_VERSION_STRING, "add_n_hot", precision,
             mantix_add_n, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "mul_n_hot", precision,
             mantix_mul_n, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "fp_add_hot", precision,
             mantix_fp_add, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "fp_mul_hot", precision,
             mantix_fp_mul, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "set_u64_hot", precision,
             mantix_set, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "set_f32_hot", precision,
             mantix_set_f32, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "copy_hot", precision,
             mantix_copy, &mantix},
            {"mantix", MANTIX_VERSION_STRING, "init_set_clear", precision,
             mantix_lifecycle, &mantix},
        };
#ifdef MANTIX_HAVE_GMP
        gmp_context gmp;
        gmp_context_init(&gmp, precision);
        benchmark gmp_tests[] = {
            {"gmp", gmp_version, "add_n_hot", precision,
             gmp_add_n, &mantix},
            {"gmp", gmp_version, "mul_n_hot", precision,
             gmp_mul_n, &mantix},
            {"gmp", gmp_version, "fp_add_hot", precision,
             gmp_fp_add, &gmp},
            {"gmp", gmp_version, "fp_mul_hot", precision,
             gmp_fp_mul, &gmp},
            {"gmp", gmp_version, "set_u64_hot", precision, gmp_set, &gmp},
            {"gmp", gmp_version, "copy_hot", precision, gmp_copy, &gmp},
            {"gmp", gmp_version, "init_set_clear", precision,
             gmp_lifecycle, &gmp},
        };
#endif
#ifdef MANTIX_HAVE_MPFR
        mpfr_context mpfr;
        mpfr_context_init(&mpfr, precision);
        benchmark mpfr_tests[] = {
            {"mpfr", mpfr_get_version(), "fp_add_hot", precision,
             mpfr_fp_add, &mpfr},
            {"mpfr", mpfr_get_version(), "fp_mul_hot", precision,
             mpfr_fp_mul, &mpfr},
            {"mpfr", mpfr_get_version(), "set_u64_hot", precision,
             mpfr_set_u64, &mpfr},
            {"mpfr", mpfr_get_version(), "set_f32_hot", precision,
             mpfr_set_f32, &mpfr},
            {"mpfr", mpfr_get_version(), "copy_hot", precision,
             mpfr_copy, &mpfr},
            {"mpfr", mpfr_get_version(), "init_set_clear", precision,
             mpfr_lifecycle, &mpfr},
        };
#endif
        for (size_t j = 0U; j < sizeof(mantix_tests) / sizeof(mantix_tests[0]); ++j) {
            run_benchmark(output, &mantix_tests[j], &settings);
#ifdef MANTIX_HAVE_GMP
            if (strcmp(mantix_tests[j].name, "set_f32_hot") != 0) {
                for (size_t k = 0U; k < sizeof(gmp_tests) / sizeof(gmp_tests[0]); ++k) {
                    if (strcmp(gmp_tests[k].name, mantix_tests[j].name) == 0) {
                        run_benchmark(output, &gmp_tests[k], &settings);
                        break;
                    }
                }
            }
#endif
#ifdef MANTIX_HAVE_MPFR
            for (size_t k = 0U; k < sizeof(mpfr_tests) / sizeof(mpfr_tests[0]); ++k) {
                if (strcmp(mpfr_tests[k].name, mantix_tests[j].name) == 0) {
                    run_benchmark(output, &mpfr_tests[k], &settings);
                    break;
                }
            }
#endif
        }
#ifdef MANTIX_HAVE_MPFR
        mpfr_context_clear(&mpfr);
#endif
#ifdef MANTIX_HAVE_GMP
        gmp_context_clear(&gmp);
#endif
        mantix_context_clear(&mantix);
    }

    if (output != stdout && fclose(output) != 0) {
        perror(settings.output_path);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
