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

#include <random>

#include "../../UMESimd.h"
#include "../utilities/TimingStatistics.h"

using namespace UME::SIMD;

// Definitions required for benchmarking kernels.
template<typename SCALAR_FLOAT_T>
SCALAR_FLOAT_T HUGE_VALUE() {
    return SCALAR_FLOAT_T(0.0);
}

template<>
float HUGE_VALUE<float>() {
    return HUGE_VALF;
}

template<>
double HUGE_VALUE<double>() {
    return HUGE_VAL;
}

template<typename SCALAR_FLOAT_T>
SCALAR_FLOAT_T NEXT_AFTER(SCALAR_FLOAT_T from, SCALAR_FLOAT_T to) {
    return std::nextafter(from, to);
}

template<>
float NEXT_AFTER(float from, float to) {
    return std::nextafterf(from, to);
}

template<>
double NEXT_AFTER(double from, double to) {
    return std::nextafter(from, to);
}

template<typename SCALAR_FLOAT_T>
struct benchmark_results {
    unsigned long long elapsedTime;
    SCALAR_FLOAT_T sin_error_ulp;
    SCALAR_FLOAT_T cos_error_ulp;
};

#include "sincos_scalar.h"
#include "sincos_vdt.h"
#include "sincos_ume.h"

int main()
{
    const int ITERATIONS = 1000;
    const int ARRAY_SIZE = 10240;

    TimingStatistics stats_scalar_f, stats_scalar_d, stats_scalar_vdt_f, stats_scalar_vdt_d;

    float max_err_sin_f = 0.0f, max_err_cos_f = 0.0f;
    double max_err_sin_d = 0.0, max_err_cos_d = 0.0;


    std::cout << "The result is amount of time it takes to calculate sine and cosine of: " << ARRAY_SIZE << " elements.\n"
        "All timing results in clock cycles. \n"
        "Speedup calculated with scalar single precision floating point result as reference.\n"
        "VDT version used as a reference for auto-vectorization capabilities.\n"
        "SIMD version uses following operations: \n"
        " SIN, COS, SINCOS\n\n";

    // ----------------------------------------
    // Benchmark using single precision.
    // ----------------------------------------

    // 1. Benchmark using std::sin/cos functions. This version will be used as reference.
    for (int i = 0; i < ITERATIONS; i++) {
        benchmark_results<float> res = test_sincos_scalar<float>(ARRAY_SIZE);
        if (max_err_sin_f < res.sin_error_ulp) max_err_sin_f = res.sin_error_ulp;
        if (max_err_cos_f < res.cos_error_ulp) max_err_cos_f = res.cos_error_ulp;
        stats_scalar_f.update(res.elapsedTime);
    }

    std::cout << "Scalar code (float): " << (unsigned long long)stats_scalar_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
        << " (speedup: 1.0x)" << " Error sin: " << max_err_sin_f << " error cos: " << max_err_cos_f
        << std::endl;

    // 2. Benchmark using VDT sincos function. This version has been designed to auto-vectorize smoothly.
    max_err_sin_f = 0.0f;
    max_err_cos_f = 0.0f;
    for (int i = 0; i < ITERATIONS; i++) {
        benchmark_results<float> res = test_sincos_vdt_scalar<float>(ARRAY_SIZE);
        if (max_err_sin_f < res.sin_error_ulp) max_err_sin_f = res.sin_error_ulp;
        if (max_err_cos_f < res.cos_error_ulp) max_err_cos_f = res.cos_error_ulp;
        stats_scalar_vdt_f.update(res.elapsedTime);
    }

    std::cout << "VDT code (float): " << (unsigned long long)stats_scalar_vdt_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_f.getStdDev()
        << " (speedup: " << stats_scalar_f.getAverage() / stats_scalar_vdt_f.getAverage() << ") "
        << " Error sin: " << max_err_sin_f << " error cos: " << max_err_cos_f
        << std::endl;

    // 3. Benchmark using UME::SIMD embedded SINCOS functions.
    benchmarkUMESIMD<float, 1>("SIMD code(1x32f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 2>("SIMD code(2x32f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 4>("SIMD code(4x32f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 8>("SIMD code(8x32f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 16>("SIMD code(16x32f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 32>("SIMD code(32x32f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);

    // 4. Benchmark using UME::SIMD separate SIN/COS functions.
    benchmarkUMESIMD_separate<float, 1>("SIMD code(1x32f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<float, 2>("SIMD code(2x32f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<float, 4>("SIMD code(4x32f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<float, 8>("SIMD code(8x32f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<float, 16>("SIMD code(16x32f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<float, 32>("SIMD code(32x32f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);

    for (int i = 0; i < ITERATIONS; i++) {
        benchmark_results<double> res = test_sincos_scalar<double>(ARRAY_SIZE);
        if (max_err_sin_d < res.sin_error_ulp) max_err_sin_d = res.sin_error_ulp;
        if (max_err_cos_d < res.cos_error_ulp) max_err_cos_d = res.cos_error_ulp;
        stats_scalar_d.update(res.elapsedTime);
    }

    // ----------------------------------------
    // Benchmark using double precision. 
    // Scalar float used as a reference.
    // ----------------------------------------

    // 5. Benchmark using std::sin/cos functions.
    std::cout << "\nScalar code (double): " << (unsigned long long)stats_scalar_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
        << " (speedup: " << stats_scalar_f.getAverage() / stats_scalar_d.getAverage() << ") "
        << " Error sin: " << max_err_sin_d << " error cos: " << max_err_cos_d
        << std::endl;

    max_err_sin_d = 0.0;
    max_err_cos_d = 0.0;
    for (int i = 0; i < ITERATIONS; i++) {
        benchmark_results<double> res = test_sincos_vdt_scalar<double>(ARRAY_SIZE);
        if (max_err_sin_d < res.sin_error_ulp) max_err_sin_d = res.sin_error_ulp;
        if (max_err_cos_d < res.cos_error_ulp) max_err_cos_d = res.cos_error_ulp;
        stats_scalar_vdt_d.update(res.elapsedTime);
    }

    // 6. Benchmark using VDT sincos function. This version has been designed to auto-vectorize smoothly.
    std::cout << "VDT code (double): " << (unsigned long long)stats_scalar_vdt_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_d.getStdDev()
        << " (speedup: " << stats_scalar_f.getAverage() / stats_scalar_vdt_d.getAverage() << ") "
        << " Error sin: " << max_err_sin_d << " error cos: " << max_err_cos_d
        << std::endl;

    // 7. Benchmark using UME::SIMD embedded SINCOS functions.
    benchmarkUMESIMD<double, 1>("SIMD code(1x64f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 2>("SIMD code(2x64f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 4>("SIMD code(4x64f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 8>("SIMD code(8x64f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 16>("SIMD code(16x64f) :", ITERATIONS, ARRAY_SIZE, stats_scalar_f);

    // 8. Benchmark using UME::SIMD separate SIN/COS functions.
    benchmarkUMESIMD_separate<double, 1>("SIMD code(1x64f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<double, 2>("SIMD code(2x64f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<double, 4>("SIMD code(4x64f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<double, 8>("SIMD code(8x64f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD_separate<double, 16>("SIMD code(16x64f) separate sin/cos: ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);

    return 0;
}
