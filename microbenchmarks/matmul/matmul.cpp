// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
//
// Author: Przemyslaw Karpinski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
//  This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//

#include <iostream>
#include <memory>

#include <immintrin.h> 
#include <cmath>
#include <time.h>
#include <stdlib.h>

// Files containing different implementations
#include "matmul_common.h"
#include "matmul_naive.h"
#include "matmul_fox.h"
#if defined(__SSE__)
#include "matmul_SSE.h"
#endif
#if defined(__AVX__)
#include "matmul_AVX.h"
#endif
#if defined(__AVX512F__)
#include "matmul_AVX512.h"
#endif
#include "matmul_UMESIMD.h"

const int MATRIX_RANK = 1000; // Array size increased to show the peeling effect.
//alignas(32) float x[ARRAY_SIZE];

int main()
{
    const int ITERATIONS = 20;

    std::cout << "The result is amount of time it takes to calculate multiplication of two \n"
                 "square matrices (" << MATRIX_RANK << "x" << MATRIX_RANK << ").\n"
                 "All measured algorithms are non-blocking.\n"
                 "All timing results in clock cycles. \n"
                 "RMS error calculated in regard to scalar (naive) version (single or double precision).\n"
                 "Speedup calculated with scalar (naive) floating point result as reference.\n\n"
                 "SIMD version uses following operations: \n"
                 " LOADA, FMULADDV, HADD\n\n";

    // SCALAR code, single precision
    TimingStatistics stats_scalar_naive_f;
    Statistics<float> error_scalar_naive_f;

    for (int i = 0; i < ITERATIONS; i++) {
        RESULTS<float> results = test_scalar_naive<float, MATRIX_RANK>();

        stats_scalar_naive_f.update(results.elapsed);
        error_scalar_naive_f.update(results.RMS_error);
    }

    std::cout << "Scalar code (naive, float): " << (unsigned long long) stats_scalar_naive_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_naive_f.getStdDev()
        << ", RMS error: " << error_scalar_naive_f.getAverage()
        << " (speedup: 1.0x)\n";


    // SCALAR code, single precision, Fox's algorithm
    TimingStatistics stats_scalar_fox_f;
    Statistics<float> error_scalar_fox_f;

    for (int i = 0; i < ITERATIONS; i++) {
        RESULTS<float> results = test_scalar_fox<float, MATRIX_RANK>();

        stats_scalar_fox_f.update(results.elapsed);
        error_scalar_fox_f.update(results.RMS_error);
    }

    std::cout << "Scalar code (Fox, float): " << (unsigned long long) stats_scalar_fox_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_fox_f.getStdDev()
        << ", RMS error: " << error_scalar_fox_f.getAverage()
        << " (speedup: " << stats_scalar_fox_f.calculateSpeedup(stats_scalar_naive_f) << "x)\n";

    // SSE intrinsic code, single precision
#if defined(__SSE__)
    benchmark_sse_32f<MATRIX_RANK>("SSE intrinsics code (4x32f): ", ITERATIONS, stats_scalar_naive_f);
#else
    std::cout << "SSE intrinsics code (4x32f): SSE instruction set not detected\n";
#endif

    // AVX/AVX2 intrinsic code, single precision
#if defined(__AVX__) || defined(__AVX2__)
    benchmark_avx_32f<MATRIX_RANK>("AVX/AVX2 intrinsics code (8x32f): ", ITERATIONS, stats_scalar_naive_f);
#else
    std::cout << "AVX/AVX2 intrinsics code (8x32f): AVX/AVX2 instruction set not detected\n";
#endif

    // AVX512 intrinsic code, single precision
#if defined(__AVX512F__)
    benchmark_avx512_32f<MATRIX_RANK>("AVX512 instrinsics code (16x32f): ", ITERATIONS, stats_scalar_naive_f);
#else
    std::cout << "AVX512 intrinsics code (16x32f): AVX512 instruction set not detected\n";
#endif

    // SIMD code, single precision
    benchmarkSIMD<UME::SIMD::SIMD1_32f, MATRIX_RANK>("SIMD code (1x32f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD2_32f, MATRIX_RANK>("SIMD code (2x32f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD4_32f, MATRIX_RANK>("SIMD code (4x32f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD8_32f, MATRIX_RANK>("SIMD code (8x32f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD16_32f, MATRIX_RANK>("SIMD code (16x32f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD32_32f, MATRIX_RANK>("SIMD code (32x32f): ", ITERATIONS, stats_scalar_naive_f);


    // SCALAR code, double precision
    TimingStatistics stats_scalar_naive_d;
    Statistics<double> error_scalar_naive_d;

    for (int i = 0; i < ITERATIONS; i++) {
        RESULTS<double> results = test_scalar_naive<double, MATRIX_RANK>();

        stats_scalar_naive_d.update(results.elapsed);
        error_scalar_naive_d.update(results.RMS_error);
    }

    std::cout << "Scalar code (naive, double): " << (unsigned long long) stats_scalar_naive_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_naive_d.getStdDev()
        << ", RMS error: " << error_scalar_naive_d.getAverage()
        << " (speedup: " << stats_scalar_naive_d.calculateSpeedup(stats_scalar_naive_f) << "x)\n";

    // scalar code, double precision, Fox's algorithm
    TimingStatistics stats_scalar_fox_d;
    Statistics<double> errors_scalar_fox_d;

    for (int i = 0; i < ITERATIONS; i++) {
        RESULTS<double> results = test_scalar_fox<double, MATRIX_RANK>();

        stats_scalar_fox_d.update(results.elapsed);
        errors_scalar_fox_d.update(results.RMS_error);
    }

    std::cout << "Scalar code (Fox, double): " << (unsigned long long) stats_scalar_fox_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_fox_d.getStdDev()
        << ", RMS error: " << errors_scalar_fox_d.getAverage()
        << " (speedup: " << stats_scalar_fox_d.calculateSpeedup(stats_scalar_naive_f) << "x)\n";

    // SSE intrinsic code, double precision
#if defined(__SSE__)
    benchmark_sse_64f<MATRIX_RANK>("SSE intrinsics code (2x64f): ", ITERATIONS, stats_scalar_naive_f);
#else
    std::cout << "SSE intrinsics code (2x64f): AVX/AVX2 instruction set not detected\n";
#endif

    // AVX/AVX2 intrinsic code, double precision
#if defined(__AVX__) || defined(__AVX2__)
    benchmark_avx_64f<MATRIX_RANK>("AVX/AVX2 intrinsics code (4x64f): ", ITERATIONS, stats_scalar_naive_f);
#else
    std::cout << "AVX/AVX2 intrinsics code (4x64f): AVX/AVX2 instruction set not detected\n";
#endif

    // AVX512 intrinsic code, double precision
#if defined(__AVX512F__)
    benchmark_avx512_64f<MATRIX_RANK>("AVX512 intrinsics code (8x64f): ", ITERATIONS, stats_scalar_naive_f);
#else
    std::cout << "AVX512 intrinsics code (8x64f): AVX512 instruction set not detected\n";
#endif

    // SIMD code, double precision
    benchmarkSIMD<UME::SIMD::SIMD1_64f, MATRIX_RANK>("SIMD code (1x64f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD2_64f, MATRIX_RANK>("SIMD code (2x64f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD4_64f, MATRIX_RANK>("SIMD code (4x64f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD8_64f, MATRIX_RANK>("SIMD code (8x64f): ", ITERATIONS, stats_scalar_naive_f);
    benchmarkSIMD<UME::SIMD::SIMD16_64f, MATRIX_RANK>("SIMD code (16x64f): ", ITERATIONS, stats_scalar_naive_f);

    return 0;
}