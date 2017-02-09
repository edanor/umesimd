// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
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
    SCALAR_FLOAT_T error_ulp;
};

template<typename SCALAR_FLOAT_T>
struct ExplogResults {
    TimingStatistics time_exp;
    TimingStatistics time_log;
    TimingStatistics time_log2;
    TimingStatistics time_log10;
    
    SCALAR_FLOAT_T max_err_exp;
    SCALAR_FLOAT_T max_err_log;
    SCALAR_FLOAT_T max_err_log2;
    SCALAR_FLOAT_T max_err_log10;
    
    void update_exp(benchmark_results<SCALAR_FLOAT_T> const & res) {
        time_exp.update(res.elapsedTime);
        if (max_err_exp < res.error_ulp) max_err_exp = res.error_ulp;
    }
    void update_log(benchmark_results<SCALAR_FLOAT_T> const & res) {
        time_log.update(res.elapsedTime);
        if (max_err_log < res.error_ulp) max_err_log = res.error_ulp;
    }
    void update_log2(benchmark_results<SCALAR_FLOAT_T> const & res) {
        time_log2.update(res.elapsedTime);
        if (max_err_log2 < res.error_ulp) max_err_log2 = res.error_ulp;
    }
    void update_log10(benchmark_results<SCALAR_FLOAT_T> const & res) {
        time_log10.update(res.elapsedTime);
        if (max_err_log10 < res.error_ulp) max_err_log10 = res.error_ulp;
    }
};

#include "explog.h"
#include "explog_scalar.h"
#include "explog_ume.h"
#include "explog_vdt.h"

int main()
{
    const int ITERATIONS = 1000;
    const int ARRAY_SIZE = 10240;

    ExplogResults<float> stats_scalar_f, stats_scalar_vdt_f;
    ExplogResults<double> stats_scalar_d, stats_scalar_vdt_d;

    std::cout << "The result is amount of time it takes to calculate exp, log_10 (base-10), log_2 (base-2) and log (base-e) of: " << ARRAY_SIZE << " elements.\n"
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar single precision floating point result as reference.\n"
        "VDT version used as a reference for auto-vectorization capabilities.\n"
        "SIMD version uses following operations: \n"
        " EXP, LOG10, LOG2, LOG\n\n";

    // ----------------------------------------
    // Benchmark using single precision.
    // ----------------------------------------

    // 1. Benchmark using std::exp/log functions. This version will be used as reference.
    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_f.update_exp(test_exp_scalar<float>(ARRAY_SIZE));
        stats_scalar_f.update_log(test_log_scalar<float>(ARRAY_SIZE));
        stats_scalar_f.update_log2(test_log2_scalar<float>(ARRAY_SIZE));
        stats_scalar_f.update_log10(test_log10_scalar<float>(ARRAY_SIZE));
    }

    std::cout << "Scalar code (float): \n"
    
        << "    EXP: time:   " << (unsigned long long)stats_scalar_f.time_exp.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.time_exp.getStdDev()
        << " (speedup: 1.0x) " 
        << "  Error:   " << (unsigned int) stats_scalar_f.max_err_exp   << "\n"
        
        << "    LOG: time:   " << (unsigned long long)stats_scalar_f.time_log.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.time_log.getStdDev()
        << " (speedup: 1.0x) " 
        << "  Error:   " << (unsigned int) stats_scalar_f.max_err_log   << "\n"
        
        << "    LOG2: time:  " << (unsigned long long)stats_scalar_f.time_log2.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.time_log2.getStdDev()
        << " (speedup: 1.0x) " 
        << "  Error:   " << (unsigned int) stats_scalar_f.max_err_log2   << "\n"

        << "    LOG10: time: " << (unsigned long long)stats_scalar_f.time_log10.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.time_log10.getStdDev()
        << " (speedup: 1.0x) " 
        << "  Error:   " << (unsigned int) stats_scalar_f.max_err_log10   << std::endl;
        
    // 2. Benchmark using VDT fast_exp functions. This version has been designed to auto-vectorize smoothly.
    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_vdt_f.update_exp(test_exp_vdt_scalar<float>(ARRAY_SIZE));
        stats_scalar_vdt_f.update_log(test_log_vdt_scalar<float>(ARRAY_SIZE));
        stats_scalar_vdt_f.update_log2(test_log2_vdt_scalar<float>(ARRAY_SIZE));
        stats_scalar_vdt_f.update_log10(test_log10_vdt_scalar<float>(ARRAY_SIZE));
    }

    std::cout << "VDT code (float): \n"
        << "    EXP: time:   " << (unsigned long long)stats_scalar_vdt_f.time_exp.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_f.time_exp.getStdDev()
        << " (speedup: " << stats_scalar_f.time_exp.getAverage() / stats_scalar_vdt_f.time_exp.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_f.max_err_exp   << "\n"
    
        << "    LOG: time:   " << (unsigned long long)stats_scalar_vdt_f.time_log.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_f.time_log.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log.getAverage() / stats_scalar_vdt_f.time_log.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_f.max_err_log   << "\n"
        
        << "    LOG2: time:   " << (unsigned long long)stats_scalar_vdt_f.time_log2.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_f.time_log2.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log2.getAverage() / stats_scalar_vdt_f.time_log2.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_f.max_err_log2   << "\n"
        
        << "    LOG10: time:   " << (unsigned long long)stats_scalar_vdt_f.time_log10.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_f.time_log10.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log10.getAverage() / stats_scalar_vdt_f.time_log10.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_f.max_err_log10   << std::endl;

    // 3. Benchmark using UME::SIMD separate EXP/LOG/LOG2/LOG10 functions.
    benchmarkUMESIMD<float, 1>("SIMD code(1x32f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 2>("SIMD code(2x32f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 4>("SIMD code(4x32f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 8>("SIMD code(8x32f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 16>("SIMD code(16x32f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<float, 32>("SIMD code(32x32f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);

    // ----------------------------------------
    // Benchmark using double precision. 
    // Scalar float used as a reference.
    // ----------------------------------------
    
    // 4. Benchmark using std::exp/log functions.
    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_d.update_exp(test_exp_scalar<double>(ARRAY_SIZE));
        stats_scalar_d.update_log(test_log_scalar<double>(ARRAY_SIZE));
        stats_scalar_d.update_log2(test_log2_scalar<double>(ARRAY_SIZE));
        stats_scalar_d.update_log10(test_log10_scalar<double>(ARRAY_SIZE));
    }
    
    std::cout << "Scalar code (double): \n" 
        << "    EXP: time:   " << (unsigned long long)stats_scalar_d.time_exp.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.time_exp.getStdDev()
        << " (speedup: " << stats_scalar_f.time_exp.getAverage() / stats_scalar_d.time_exp.getAverage() << ") "
        << "  Error:   " << (unsigned int)stats_scalar_d.max_err_exp   << "\n"
    
        << "    LOG: time:   " << (unsigned long long)stats_scalar_d.time_log.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.time_log.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log.getAverage() / stats_scalar_d.time_log.getAverage() << ") "
        << "  Error:   " << (unsigned int)stats_scalar_d.max_err_log   << "\n"
        
        << "    LOG2: time:   " << (unsigned long long)stats_scalar_d.time_log2.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.time_log2.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log2.getAverage() / stats_scalar_d.time_log2.getAverage() << ") "
        << "  Error:   " << (unsigned int)stats_scalar_d.max_err_log2   << "\n"
        
        << "    LOG10: time:   " << (unsigned long long)stats_scalar_d.time_log10.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.time_log10.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log10.getAverage() / stats_scalar_d.time_log10.getAverage() << ") "
        << "  Error:   " << (unsigned int)stats_scalar_d.max_err_log10   << std::endl;
        
    // 5. Benchmark using VDT fast_exp/fast_log function. This version has been designed to auto-vectorize smoothly.
    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_vdt_d.update_exp(test_exp_vdt_scalar<double>(ARRAY_SIZE));
        stats_scalar_vdt_d.update_log(test_log_vdt_scalar<double>(ARRAY_SIZE));
        stats_scalar_vdt_d.update_log2(test_log2_vdt_scalar<double>(ARRAY_SIZE));
        stats_scalar_vdt_d.update_log10(test_log10_vdt_scalar<double>(ARRAY_SIZE));
    }

    std::cout << "VDT code (double): \n"
        << "    EXP: time:   " << (unsigned long long)stats_scalar_vdt_d.time_exp.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_d.time_exp.getStdDev()
        << " (speedup: " << stats_scalar_f.time_exp.getAverage() / stats_scalar_vdt_d.time_exp.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_d.max_err_exp   << "\n"
    
        << "    LOG: time:   " << (unsigned long long)stats_scalar_vdt_d.time_log.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_d.time_log.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log.getAverage() / stats_scalar_vdt_d.time_log.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_d.max_err_log   << "\n"
        
        << "    LOG2: time:   " << (unsigned long long)stats_scalar_vdt_d.time_log2.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_d.time_log2.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log2.getAverage() / stats_scalar_vdt_d.time_log2.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_d.max_err_log2   << "\n"
        
        << "    LOG10: time:   " << (unsigned long long)stats_scalar_vdt_d.time_log10.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_vdt_d.time_log10.getStdDev()
        << " (speedup: " << stats_scalar_f.time_log10.getAverage() / stats_scalar_vdt_d.time_log10.getAverage() << ") "
        << "  Error:   " << (unsigned int) stats_scalar_vdt_d.max_err_log10   << std::endl;

    // 3. Benchmark using UME::SIMD separate EXP/LOG/LOG2/LOG10 functions.
    benchmarkUMESIMD<double, 1>("SIMD code(1x64f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 2>("SIMD code(2x64f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 4>("SIMD code(4x64f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 8>("SIMD code(8x64f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);
    benchmarkUMESIMD<double, 16>("SIMD code(16x64f) ", ITERATIONS, ARRAY_SIZE, stats_scalar_f);

    return 0;
}
