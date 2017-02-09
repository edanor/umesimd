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

#ifndef SINCOS_SCALAR_H_
#define SINCOS_SCALAR_H_

#include "sincos.h"

// Kernel for benchmarking using std::sin/std::cos function calls.
template<typename SCALAR_FLOAT_T>
benchmark_results<SCALAR_FLOAT_T> test_sincos_scalar(int ARRAY_SIZE)
{
    unsigned long long start, end;    // Time measurements

    std::random_device rd;
    std::mt19937 gen(rd());

    const int LEN = ARRAY_SIZE;
    SCALAR_FLOAT_T *inputA = (SCALAR_FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), sizeof(SCALAR_FLOAT_T));
    SCALAR_FLOAT_T *output_sin = (SCALAR_FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), sizeof(SCALAR_FLOAT_T));
    SCALAR_FLOAT_T *output_cos = (SCALAR_FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), sizeof(SCALAR_FLOAT_T));
    SCALAR_FLOAT_T *values_sin = (SCALAR_FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), sizeof(SCALAR_FLOAT_T));
    SCALAR_FLOAT_T *values_cos = (SCALAR_FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(SCALAR_FLOAT_T), sizeof(SCALAR_FLOAT_T));

    std::uniform_real_distribution<SCALAR_FLOAT_T> dist(-5 * SCALAR_FLOAT_T(M_PI), 5 * SCALAR_FLOAT_T(M_PI));

    for (int i = 0; i < LEN; i++) {
        inputA[i] = dist(gen);
        output_sin[i] = std::sin(inputA[i]);
        output_cos[i] = std::cos(inputA[i]);
    }

    start = get_timestamp();

    for (int i = 0; i < LEN; i++) {
        values_sin[i] = std::sin(inputA[i]);
        values_cos[i] = std::cos(inputA[i]);
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

    UME::DynamicMemory::AlignedFree(inputA);
    UME::DynamicMemory::AlignedFree(output_sin);
    UME::DynamicMemory::AlignedFree(output_cos);
    UME::DynamicMemory::AlignedFree(values_sin);
    UME::DynamicMemory::AlignedFree(values_cos);

    return result;
}

#endif
