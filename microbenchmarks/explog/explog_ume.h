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

#ifndef EXPLOG_UME_H_
#define EXPLOG_UME_H_

#include "explog_scalar.h"

// Kernel for benchmarking using EXP function.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_exp_ume(const int ARRAY_SIZE)
{
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> x;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y;

    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T input[LEN];
    SCALAR_FLOAT_T output[LEN];
    SCALAR_FLOAT_T values[LEN];

    generate_some_exp_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i += VEC_LEN) {
        x.load(&input[i]);

        y = x.exp();

        y.store(&values[i]);
    }
    
    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    return result;
}

// Kernel for benchmarking using LOG function.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log_ume(const int ARRAY_SIZE)
{
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> x;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y;

    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T input[LEN];
    SCALAR_FLOAT_T output[LEN];
    SCALAR_FLOAT_T values[LEN];

    generate_some_log_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i += VEC_LEN) {
        x.load(&input[i]);

        y = x.log();

        y.store(&values[i]);
    }
    
    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    return result;
}

// Kernel for benchmarking using LOG2 function.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log2_ume(const int ARRAY_SIZE)
{
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> x;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y;

    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T input[LEN];
    SCALAR_FLOAT_T output[LEN];
    SCALAR_FLOAT_T values[LEN];

    generate_some_log2_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i += VEC_LEN) {
        x.load(&input[i]);

        y = x.log2();

        y.store(&values[i]);
    }
    
    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    return result;
}

// Kernel for benchmarking using LOG10 function.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
UME_NEVER_INLINE benchmark_results<SCALAR_FLOAT_T> test_log10_ume(const int ARRAY_SIZE)
{
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> x;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y;

    unsigned long long start, end;    // Time measurements

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T input[LEN];
    SCALAR_FLOAT_T output[LEN];
    SCALAR_FLOAT_T values[LEN];

    generate_some_log10_values<SCALAR_FLOAT_T>(LEN, input, output);

    start = get_timestamp();

    for (int i = 0; i < LEN; i += VEC_LEN) {
        x.load(&input[i]);

        y = x.log10();

        y.store(&values[i]);
    }
    
    end = get_timestamp();

    SCALAR_FLOAT_T max_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values[i]);
        SCALAR_FLOAT_T error_ulp = (values[i] - output[i]) / reference_value_ulp;

        if (max_err < std::abs(error_ulp)) max_err = std::abs(error_ulp);
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.error_ulp = max_err;
    return result;
}

// Benchmark using SINCOS function call.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
void benchmarkUMESIMD(std::string resultPrefix, int iterations, int array_size, ExplogResults<float> & reference)
{
    ExplogResults<SCALAR_FLOAT_T> result;
    SCALAR_FLOAT_T max_err_sin = 0, max_err_cos = 0;

    for (int i = 0; i < iterations; i++)
    {
        result.update_exp(test_exp_ume<SCALAR_FLOAT_T, VEC_LEN>(array_size));
        result.update_log(test_log_ume<SCALAR_FLOAT_T, VEC_LEN>(array_size));
        result.update_log2(test_log2_ume<SCALAR_FLOAT_T, VEC_LEN>(array_size));
        result.update_log10(test_log10_ume<SCALAR_FLOAT_T, VEC_LEN>(array_size));
    }

    std::cout << resultPrefix.c_str() << "\n"
        << "    EXP: time:   " << (unsigned long long)result.time_exp.getAverage()
        << ", dev: " << (unsigned long long) result.time_exp.getStdDev()
        << " (speedup: " << reference.time_exp.getAverage() / result.time_exp.getAverage() << ") "
        << "  Error:   " << (unsigned int) result.max_err_exp   << "\n"
    
        << "    LOG: time:   " << (unsigned long long)result.time_log.getAverage()
        << ", dev: " << (unsigned long long) result.time_log.getStdDev()
        << " (speedup: " << reference.time_log.getAverage() / result.time_log.getAverage() << ") "
        << "  Error:   " << (unsigned int) result.max_err_log   << "\n"
        
        << "    LOG2: time:   " << (unsigned long long)result.time_log2.getAverage()
        << ", dev: " << (unsigned long long) result.time_log2.getStdDev()
        << " (speedup: " << reference.time_log2.getAverage() / result.time_log2.getAverage() << ") "
        << "  Error:   " << (unsigned int) result.max_err_log2   << "\n"
        
        << "    LOG10: time:   " << (unsigned long long)result.time_log10.getAverage()
        << ", dev: " << (unsigned long long) result.time_log10.getStdDev()
        << " (speedup: " << reference.time_log10.getAverage() / result.time_log10.getAverage() << ") "
        << "  Error:   " << (unsigned int) result.max_err_log10   << std::endl;
}

#endif
