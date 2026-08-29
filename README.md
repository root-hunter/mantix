# Mantix

Mantix is an early-stage, low-level C library for arbitrary-precision binary
floating-point arithmetic.

The project currently requires **Clang 23.x**, CMake 3.20 or newer, and Ninja.

The core representation is:

```text
(-1)^sign * significand * 2^exponent
```

The significand is an arbitrary-length, little-endian array of 64-bit limbs.
Non-zero values are kept canonical by removing powers of two from the
significand and adding them to the exponent.

Each value contains one inline limb. Zero and machine-sized integers therefore
require no heap allocation; larger significands transparently move to owned
dynamic storage.

## Source layout

```text
src/
  core/                 High-level values, ownership, and public API
  kernels/
    scalar/             Portable and scalar CPU limb kernels
    avx2/               Planned AVX2 kernels
    avx512/             Planned AVX-512 kernels
    amx/                Planned Intel AMX kernels
    neon/               Planned Arm NEON kernels
    cuda/               Planned NVIDIA CUDA kernels
```

Only the scalar backend is implemented today. Architecture-specific backends
will be added independently and selected by a dispatch layer once their APIs
and supported workloads are defined.

## Build and test

```sh
make test
make example
make sanitize
make benchmark
```

Or use CMake directly:

```sh
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=clang-23 -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

## Current scope

The initial API establishes value representation, ownership, initialization,
canonical zero, conversion from `uint64_t`, copying, and invariant checks.
Arithmetic and rounding are deliberately left for the next development stage.

The low-level limb API in `mantix/limb.h` provides carry-accelerated addition,
subtraction, and full-width equal-size multiplication. On x86-64 it uses the
processor carry flag and selects a BMI2 `mulx` multiplication kernel at runtime;
other targets use the portable 128-bit implementation. Runtime detection also
exposes ADX and AVX2 for future dual-carry and batch kernels.

Multiplication uses one- and two-limb fast paths, a fully unrolled four-limb
Comba kernel, a fixed-size eight-limb BMI2 kernel, a 16-way-unrolled BMI2
basecase, and stack-only Karatsuba steps for the 32- and 64-limb workloads. The
Karatsuba products are written directly into their final output ranges to keep
temporary storage small. Dispatch thresholds are benchmark-driven and the
low-level limb API performs no heap allocation.

## Benchmarks

Benchmarks are always built in Release mode with Clang 23. When their respective
development files are installed, the high-level workloads are also run against
GMP's `mpf_t` and MPFR's `mpfr_t`.
The `add_n_hot` and `mul_n_hot` workloads compare Mantix directly with GMP's
`mpn` kernels on the same limb buffers, avoiding differences in high-level
number representation. MPFR is included in `set_u64_hot`, `copy_hot`, and
`init_set_clear`; it does not expose equivalent public limb-level kernels.

```sh
# Human-readable results (defaults: 100000 iterations, 7 samples)
make benchmark

# Override workload size
make benchmark BENCH_ARGS="--iterations 1000000 --samples 11"

# Save a machine-readable snapshot
make benchmark-save BENCH_OUT=benchmarks/results/v0.1.0.csv

# Compare Mantix results from two revisions; negative delta means lower latency
python3 benchmarks/compare.py old.csv new.csv --library mantix
```

Run benchmarks on an otherwise idle machine, with the same compiler, CPU power
policy, iteration count, and sample count. Compare medians; the minimum is also
reported as a useful approximation of execution without scheduler noise.
