# Kernels

Mantix keeps low-level arithmetic kernels separate from the high-level value
and ownership code in `src/core`.

The `scalar` backend is the portable baseline and currently also contains the
x86-64 ADC and BMI2 scalar fast paths. The remaining backend directories are
reserved for implementations that operate on the same internal kernel API:

- `avx2`: x86 AVX2 vector kernels
- `avx512`: x86 AVX-512 vector kernels
- `amx`: Intel AMX matrix kernels
- `neon`: Arm NEON vector kernels
- `cuda`: NVIDIA CUDA kernels

Backend-specific code should not expose new public symbols. Runtime feature
detection and backend selection belong in a common dispatch layer.
