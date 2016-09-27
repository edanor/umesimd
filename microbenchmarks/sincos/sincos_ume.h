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

#ifndef SINCOS_UME_H_
#define SINCOS_UME_H_

// Kernel for benchmarking using SINCOS function.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
benchmark_results<SCALAR_FLOAT_T> test_sincos_ume(int array_size)
{
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> x;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y_sin;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y_cos;

    unsigned long long start, end;    // Time measurements

    std::random_device rd;
    std::mt19937 gen(rd());

    const int LEN = array_size;
    SCALAR_FLOAT_T inputA[LEN];
    SCALAR_FLOAT_T output_sin[LEN];
    SCALAR_FLOAT_T output_cos[LEN];
    SCALAR_FLOAT_T values_sin[LEN];
    SCALAR_FLOAT_T values_cos[LEN];

    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(-5 * SCALAR_FLOAT_T(M_PI), 5 * SCALAR_FLOAT_T(M_PI));

    for (int i = 0; i < LEN; i++) {
        inputA[i] = dist(gen);
        output_sin[i] = std::sin(inputA[i]);
        output_cos[i] = std::cos(inputA[i]);
    }

    start = get_timestamp();

    for (int i = 0; i < LEN; i += VEC_LEN) {
        x.load(&inputA[i]);

        x.sincos(y_sin, y_cos);

        //call_sincos<SCALAR_FLOAT_T, VEC_LEN>(x, y_sin, y_cos);

        y_sin.store(&values_sin[i]);
        y_cos.store(&values_cos[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_sin_err = 0;
    SCALAR_FLOAT_T max_cos_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values_sin[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values_sin[i]);
        SCALAR_FLOAT_T error_ulp = (values_sin[i] - output_sin[i]) / reference_value_ulp;

        if (max_sin_err < std::abs(error_ulp)) max_sin_err = std::abs(error_ulp);
        //if (output_sin[i] != values_sin[i])
        //    std::cout << " Difference in sin[" << i << "]: " << values_sin[i]
        //    << " should be: " << output_sin[i]
        //    << " error(ulp): " << error_ulp << std::endl;

        next = NEXT_AFTER(values_cos[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        reference_value_ulp = std::abs(next - values_cos[i]);
        error_ulp = (values_cos[i] - output_cos[i]) / reference_value_ulp;

        if (max_cos_err < std::abs(error_ulp)) max_cos_err = std::abs(error_ulp);
        //if (output_cos[i] != values_cos[i])
        //    std::cout << " Difference in cos[" << i << "]: " << values_cos[i]
        //    << " should be: " << output_cos[i]
        //    << " error(ulp): " << error_ulp << std::endl;
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.sin_error_ulp = max_sin_err;
    result.cos_error_ulp = max_cos_err;
    return result;
}

// Kernel for benchmarking using separate SIN/COS functions.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
benchmark_results<SCALAR_FLOAT_T> test_sincos_ume_separate(int array_size)
{
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> x;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y_sin;
    SIMDVec<SCALAR_FLOAT_T, VEC_LEN> y_cos;

    unsigned long long start, end;    // Time measurements

    std::random_device rd;
    std::mt19937 gen(rd());

    const int LEN = array_size;
    SCALAR_FLOAT_T inputA[LEN];
    SCALAR_FLOAT_T output_sin[LEN];
    SCALAR_FLOAT_T output_cos[LEN];
    SCALAR_FLOAT_T values_sin[LEN];
    SCALAR_FLOAT_T values_cos[LEN];

    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(-5 * SCALAR_FLOAT_T(M_PI), 5 * SCALAR_FLOAT_T(M_PI));

    for (int i = 0; i < LEN; i++) {
        inputA[i] = dist(gen);
        output_sin[i] = std::sin(inputA[i]);
        output_cos[i] = std::cos(inputA[i]);
    }

    start = get_timestamp();

    for (int i = 0; i < LEN; i += VEC_LEN) {
        x.load(&inputA[i]);

        y_sin = x.sin();
        y_cos = x.cos();

        y_sin.store(&values_sin[i]);
        y_cos.store(&values_cos[i]);
    }

    end = get_timestamp();

    SCALAR_FLOAT_T max_sin_err = 0;
    SCALAR_FLOAT_T max_cos_err = 0;

    for (int i = 0; i < LEN; i++) {
        SCALAR_FLOAT_T next = NEXT_AFTER(values_sin[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        SCALAR_FLOAT_T reference_value_ulp = std::abs(next - values_sin[i]);
        SCALAR_FLOAT_T error_ulp = (values_sin[i] - output_sin[i]) / reference_value_ulp;

        if (max_sin_err < std::abs(error_ulp)) max_sin_err = std::abs(error_ulp);
        //if (output_sin[i] != values_sin[i])
        //    std::cout << " Difference in sin[" << i << "]: " << values_sin[i]
        //    << " should be: " << output_sin[i]
        //    << " error(ulp): " << error_ulp << std::endl;

        next = NEXT_AFTER(values_cos[i], HUGE_VALUE<SCALAR_FLOAT_T>());
        reference_value_ulp = std::abs(next - values_cos[i]);
        error_ulp = (values_cos[i] - output_cos[i]) / reference_value_ulp;

        if (max_cos_err < std::abs(error_ulp)) max_cos_err = std::abs(error_ulp);
        //if (output_cos[i] != values_cos[i])
        //    std::cout << " Difference in cos[" << i << "]: " << values_cos[i]
        //    << " should be: " << output_cos[i]
        //    << " error(ulp): " << error_ulp << std::endl;
    }

    benchmark_results<SCALAR_FLOAT_T> result;
    result.elapsedTime = end - start;
    result.sin_error_ulp = max_sin_err;
    result.cos_error_ulp = max_cos_err;
    return result;
}

// Benchmark using SINCOS function call.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
void benchmarkUMESIMD(std::string resultPrefix, int iterations, int array_size, TimingStatistics & reference)
{
    TimingStatistics stats;
    benchmark_results<SCALAR_FLOAT_T> result;
    SCALAR_FLOAT_T max_err_sin = 0, max_err_cos = 0;

    for (int i = 0; i < iterations; i++)
    {
        result = test_sincos_ume<SCALAR_FLOAT_T, VEC_LEN>(array_size);
        if (max_err_sin < result.sin_error_ulp) max_err_sin = result.sin_error_ulp;
        if (max_err_cos < result.cos_error_ulp) max_err_cos = result.cos_error_ulp;

        stats.update(result.elapsedTime);
    }

    std::cout << resultPrefix.c_str() << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << " Error sin: " << max_err_sin << " error cos: " << max_err_cos
        << std::endl;
}

// Benchmark using separate SIN/COS function calls.
template<typename SCALAR_FLOAT_T, int VEC_LEN>
void benchmarkUMESIMD_separate(std::string resultPrefix, int iterations, int array_size, TimingStatistics & reference)
{
    TimingStatistics stats;
    benchmark_results<SCALAR_FLOAT_T> result;
    SCALAR_FLOAT_T max_err_sin = 0, max_err_cos = 0;

    for (int i = 0; i < iterations; i++)
    {
        result = test_sincos_ume_separate<SCALAR_FLOAT_T, VEC_LEN>(array_size);
        if (max_err_sin < result.sin_error_ulp) max_err_sin = result.sin_error_ulp;
        if (max_err_cos < result.cos_error_ulp) max_err_cos = result.cos_error_ulp;

        stats.update(result.elapsedTime);
    }

    std::cout << resultPrefix.c_str() << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << " Error sin: " << max_err_sin << " error cos: " << max_err_cos
        << std::endl;
}

#endif
