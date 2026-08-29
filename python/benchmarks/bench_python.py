import time
import sys
from decimal import Decimal
import mantix
import gmpy2
import mpmath

def time_op(func, iterations=1_000_000):
    # Warmup
    for _ in range(20_000):
        func()
    
    start = time.perf_counter_ns()
    for _ in range(iterations):
        func()
    end = time.perf_counter_ns()
    return (end - start) / iterations

def run_benchmarks(precision_bits=53):
    print(f"\n=======================================================")
    print(f"  Benchmark Precision: {precision_bits} bits")
    print(f"=======================================================")
    
    # 1. Float creation / conversion
    print(f"\n--- Benchmark 1: Object Creation from float (ns/op) ---")
    val = 123.456789
    
    t_mantix = time_op(lambda: mantix.Float(val, precision=precision_bits))
    t_gmpy2 = time_op(lambda: gmpy2.mpfr(val, precision_bits))
    t_mpmath = time_op(lambda: mpmath.mpf(val))
    t_decimal = time_op(lambda: Decimal(val))
    t_py = time_op(lambda: float(val))
    
    print(f"  mantix.Float   : {t_mantix:7.2f} ns")
    print(f"  gmpy2.mpfr     : {t_gmpy2:7.2f} ns")
    print(f"  decimal.Decimal: {t_decimal:7.2f} ns")
    print(f"  mpmath.mpf     : {t_mpmath:7.2f} ns")
    print(f"  python float   : {t_py:7.2f} ns")
    
    # 2. Multiplication
    print(f"\n--- Benchmark 2: Multiplication (a * b) (ns/op) ---")
    m_a = mantix.Float(1.5, precision=precision_bits)
    m_b = mantix.Float(2.5, precision=precision_bits)
    
    g_a = gmpy2.mpfr(1.5, precision_bits)
    g_b = gmpy2.mpfr(2.5, precision_bits)
    
    mp_a = mpmath.mpf(1.5)
    mp_b = mpmath.mpf(2.5)
    
    d_a = Decimal("1.5")
    d_b = Decimal("2.5")
    
    py_a = 1.5
    py_b = 2.5
    
    t_mantix_mul = time_op(lambda: m_a * m_b)
    t_gmpy2_mul = time_op(lambda: g_a * g_b)
    t_mpmath_mul = time_op(lambda: mp_a * mp_b)
    t_decimal_mul = time_op(lambda: d_a * d_b)
    t_py_mul = time_op(lambda: py_a * py_b)
    
    print(f"  mantix.Float   : {t_mantix_mul:7.2f} ns")
    print(f"  gmpy2.mpfr     : {t_gmpy2_mul:7.2f} ns")
    print(f"  decimal.Decimal: {t_decimal_mul:7.2f} ns")
    print(f"  mpmath.mpf     : {t_mpmath_mul:7.2f} ns")
    print(f"  python float   : {t_py_mul:7.2f} ns")
    
    # 3. Addition
    print(f"\n--- Benchmark 3: Addition (a + b) (ns/op) ---")
    t_mantix_add = time_op(lambda: m_a + m_b)
    t_gmpy2_add = time_op(lambda: g_a + g_b)
    t_mpmath_add = time_op(lambda: mp_a + mp_b)
    t_decimal_add = time_op(lambda: d_a + d_b)
    t_py_add = time_op(lambda: py_a + py_b)
    
    print(f"  mantix.Float   : {t_mantix_add:7.2f} ns")
    print(f"  gmpy2.mpfr     : {t_gmpy2_add:7.2f} ns")
    print(f"  decimal.Decimal: {t_decimal_add:7.2f} ns")
    print(f"  mpmath.mpf     : {t_mpmath_add:7.2f} ns")
    print(f"  python float   : {t_py_add:7.2f} ns")

    # 4. Division
    print(f"\n--- Benchmark 4: Division (a / b) (ns/op) ---")
    t_mantix_div = time_op(lambda: m_a / m_b)
    t_gmpy2_div = time_op(lambda: g_a / g_b)
    t_mpmath_div = time_op(lambda: mp_a / mp_b)
    t_decimal_div = time_op(lambda: d_a / d_b)
    t_py_div = time_op(lambda: py_a / py_b)

    print(f"  mantix.Float   : {t_mantix_div:7.2f} ns")
    print(f"  gmpy2.mpfr     : {t_gmpy2_div:7.2f} ns")
    print(f"  decimal.Decimal: {t_decimal_div:7.2f} ns")
    print(f"  mpmath.mpf     : {t_mpmath_div:7.2f} ns")
    print(f"  python float   : {t_py_div:7.2f} ns")

    # 5. Square Root
    print(f"\n--- Benchmark 5: Square Root (sqrt(a)) (ns/op) ---")
    t_mantix_sqrt = time_op(lambda: mantix.sqrt(m_a))
    t_gmpy2_sqrt = time_op(lambda: gmpy2.sqrt(g_a))
    t_mpmath_sqrt = time_op(lambda: mpmath.sqrt(mp_a))
    t_decimal_sqrt = time_op(lambda: d_a.sqrt())

    print(f"  mantix.sqrt    : {t_mantix_sqrt:7.2f} ns")
    print(f"  gmpy2.sqrt     : {t_gmpy2_sqrt:7.2f} ns")
    print(f"  decimal.Decimal: {t_decimal_sqrt:7.2f} ns")
    print(f"  mpmath.sqrt    : {t_mpmath_sqrt:7.2f} ns")

    # 6. In-place Accumulation
    print(f"\n--- Benchmark 6: In-place Accumulation (acc += val) (ns/op) ---")
    def bench_inplace_mantix(N=1000):
        acc = mantix.Float(0.0, precision=precision_bits)
        val = mantix.Float(1.25, precision=precision_bits)
        for _ in range(N):
            acc += val
        return acc

    def bench_inplace_gmpy2(N=1000):
        acc = gmpy2.mpfr(0.0, precision_bits)
        val = gmpy2.mpfr(1.25, precision_bits)
        for _ in range(N):
            acc += val
        return acc

    def bench_inplace_py(N=1000):
        acc = 0.0
        val = 1.25
        for _ in range(N):
            acc += val
        return acc

    N = 1000
    loops = 1000
    start = time.perf_counter_ns()
    for _ in range(loops):
        bench_inplace_mantix(N)
    t_mantix_inplace = (time.perf_counter_ns() - start) / (loops * N)

    start = time.perf_counter_ns()
    for _ in range(loops):
        bench_inplace_gmpy2(N)
    t_gmpy2_inplace = (time.perf_counter_ns() - start) / (loops * N)

    start = time.perf_counter_ns()
    for _ in range(loops):
        bench_inplace_py(N)
    t_py_inplace = (time.perf_counter_ns() - start) / (loops * N)

    print(f"  mantix.Float   : {t_mantix_inplace:7.2f} ns/op")
    print(f"  gmpy2.mpfr     : {t_gmpy2_inplace:7.2f} ns/op")
    print(f"  python float   : {t_py_inplace:7.2f} ns/op")

    # 5. Vector Dot Product (1000 elements)
    print(f"\n--- Benchmark 5: Vector Dot Product (1000 elements) (μs/call) ---")
    vec_len = 1000
    m_v1 = [mantix.Float(float(i % 10), precision=precision_bits) for i in range(vec_len)]
    m_v2 = [mantix.Float(float((i + 1) % 10), precision=precision_bits) for i in range(vec_len)]
    
    g_v1 = [gmpy2.mpfr(float(i % 10), precision_bits) for i in range(vec_len)]
    g_v2 = [gmpy2.mpfr(float((i + 1) % 10), precision_bits) for i in range(vec_len)]
    
    py_v1 = [float(i % 10) for i in range(vec_len)]
    py_v2 = [float((i + 1) % 10) for i in range(vec_len)]

    calls = 1000
    start = time.perf_counter_ns()
    for _ in range(calls):
        mantix.dot(m_v1, m_v2)
    t_mantix_dot = (time.perf_counter_ns() - start) / (calls * 1000.0)

    start = time.perf_counter_ns()
    for _ in range(calls):
        sum(x * y for x, y in zip(g_v1, g_v2))
    t_gmpy2_dot = (time.perf_counter_ns() - start) / (calls * 1000.0)

    start = time.perf_counter_ns()
    for _ in range(calls):
        sum(x * y for x, y in zip(py_v1, py_v2))
    t_py_dot = (time.perf_counter_ns() - start) / (calls * 1000.0)

    print(f"  mantix.dot     : {t_mantix_dot:7.2f} μs")
    print(f"  gmpy2 (zip)    : {t_gmpy2_dot:7.2f} μs")
    print(f"  python float   : {t_py_dot:7.2f} μs")

if __name__ == "__main__":
    for prec in [53, 128, 256, 1024]:
        run_benchmarks(prec)
