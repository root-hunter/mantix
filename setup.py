import os
from setuptools import setup, find_packages, Extension

mantix_core_sources = [
    "python/src/_core.c",
    "src/core/mantix.c",
    "src/kernels/scalar/limb.c",
]

mantix_module = Extension(
    "mantix._core",
    sources=mantix_core_sources,
    include_dirs=["include", "python/src"],
    extra_compile_args=["-O3", "-march=native", "-std=c17", "-Wall", "-Wextra"],
)

setup(
    name="mantix",
    version="0.1.0",
    description="High-performance arbitrary-precision binary floating-point library",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[mantix_module],
    python_requires=">=3.8",
)
