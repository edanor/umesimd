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

#ifndef MATMUL_UMESIMD_H_
#define MATMUL_UMESIMD_H_

#include "matmul_common.h"

template<typename FLOAT_VEC_TYPE, int MAT_RANK>
RESULTS<typename UME::SIMD::SIMDTraits<FLOAT_VEC_TYPE>::SCALAR_T> test_SIMD()
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_TYPE>::SCALAR_T FLOAT_T;
    uint32_t ALIGNMENT = FLOAT_VEC_TYPE::alignment();
    const uint32_t SIMD_STRIDE = FLOAT_VEC_TYPE::length();

    unsigned long long start, end; // Time measurements
    FLOAT_T *A, *B, *B_T, *C;

    // All arrays should be padded, so that rows start at optimal alignment.
    // Making each row of A and column of B padded, also simplifies 
    int PADDING = SIMD_STRIDE - (MAT_RANK % SIMD_STRIDE);

    // Allocate alligned to a single scalar
    A = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc((MAT_RANK + PADDING)*MAT_RANK*sizeof(FLOAT_T), ALIGNMENT);
    // B doesn't have to be padded, since we will transpose it at the beginning of computation
    B = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(MAT_RANK*MAT_RANK*sizeof(FLOAT_T), ALIGNMENT);
    B_T = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc((MAT_RANK + PADDING)*MAT_RANK*sizeof(FLOAT_T), ALIGNMENT);
    C = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc((MAT_RANK)*MAT_RANK*sizeof(FLOAT_T), ALIGNMENT);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < MAT_RANK; i++)
    {
        for (int j = 0; j < MAT_RANK; j++)
        {
            // Generate random numbers in range (0.0;1.0)
            A[i*(MAT_RANK + PADDING) + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            B[i*MAT_RANK + j] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
            C[i*(MAT_RANK)+j] = float(0);
        }
        for (int j = MAT_RANK; j < MAT_RANK + PADDING; j++) {
            A[i*(MAT_RANK + PADDING) + j] = 0.0f;
            //C[i*(MAT_RANK + PADDING) + j] = 0.0f;
        }
    }
    /*
    std::cout << "\n\nA = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << A[i*(MAT_RANK +PADDING) + j] << " ";
    }
    std::cout << ";\n";
    }

    std::cout << "\n\nB = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B[i*MAT_RANK + j] << " ";
    }
    std::cout << ";\n";
    }*/


    start = __rdtsc();

    // Transpose B matrix to a row-major form
    for (int i = 0; i < MAT_RANK; i++) {
        for (int j = 0; j < MAT_RANK;j++)
        {
            B_T[i*(MAT_RANK + PADDING) + j] = B[j*MAT_RANK + i];
        }
        for (int j = MAT_RANK; j < MAT_RANK + PADDING; j++) {
            B_T[i*(MAT_RANK + PADDING) + j] = 0.0f;
        }
    }

    // For each row in C
    for (int i = 0; i < MAT_RANK; i++) {
        // For each column in C
        for (int j = 0; j < MAT_RANK; j++) {
            FLOAT_VEC_TYPE t0(FLOAT_T(0));
            // Traverse single row of A and single column of B
            for (int k = 0; k < MAT_RANK + PADDING; k += SIMD_STRIDE) {
                FLOAT_VEC_TYPE t1, t2, t3;

                t1.loada(&A[i*(MAT_RANK + PADDING) + k]);
                t2.loada(&B_T[j*(MAT_RANK + PADDING) + k]);
                t3 = t1.fmuladd(t2, t0);
                t0 = t3;
            }
            C[i*(MAT_RANK)+j] = t0.hadd();
        }
    }

    end = __rdtsc();
    /*
    std::cout << "\n\nB': \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B_T[i*(MAT_RANK + PADDING) + j] << " ";
    }
    std::cout << std::endl;
    }

    std::cout << "\n\nC = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << C[i*(MAT_RANK + PADDING) + j] << " ";
    }
    std::cout << ";\n";
    }*/

    FLOAT_T error = calculate_RMS_error_SIMD<FLOAT_T, MAT_RANK, SIMD_STRIDE>(A, B, C);
    //std::cout << "SIMD RMS error: " << error << std::endl;

    UME::DynamicMemory::AlignedFree(A);
    UME::DynamicMemory::AlignedFree(B);
    UME::DynamicMemory::AlignedFree(B_T);
    UME::DynamicMemory::AlignedFree(C);

    RESULTS<FLOAT_T> results;
    results.elapsed = end - start;
    results.RMS_error = error;
    return results;
}

template<typename FLOAT_VEC_T, int MAT_RANK>
void benchmarkSIMD(std::string const & resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::SCALAR_T FLOAT_T;
    TimingStatistics stats;
    Statistics<FLOAT_T> errors;

    for (int i = 0; i < iterations; i++)
    {
        RESULTS<FLOAT_T> results = test_SIMD<FLOAT_VEC_T, MAT_RANK>();
        stats.update(results.elapsed);
        errors.update(results.RMS_error);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", RMS error: " << errors.getAverage()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

#endif
