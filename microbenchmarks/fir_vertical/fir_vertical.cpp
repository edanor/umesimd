// The MIT License (MIT)
//
// Copyright (c) 2015 CERN
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

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h> 
#endif

#include <cmath>
#include <time.h>
#include <stdlib.h>

//#define UME_SIMD_SHOW_EMULATION_WARNINGS 1
#include "../../UMESimd.h"
#include "../utilities/TimingStatistics.h"

const int ARRAY_SIZE = 512*1024;
const int ITERATIONS = 20;

//#define ENABLE_DEBUG
#include "fir_vertical_umesimd.h"
#include "fir_vertical_intel.h"

// FIR_ORDER == 1 means a 0-order gain filter y[t]=a*x[t]
// FIR_ORDER == 4 means a 3-order FIR filter y[t]=a*x[t] + b*x[t-1] + c*x[t-2]
template<typename FLOAT_T, int FIR_ORDER>
UME_NEVER_INLINE TIMING_RES test_scalar()
{
    unsigned long long start, end; // Time measurements
    FLOAT_T coeffs[FIR_ORDER];
    FLOAT_T state[FIR_ORDER];
    int state_begin = 0;
    FLOAT_T *x;
    FLOAT_T *y;

    x = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), sizeof(FLOAT_T));
    y = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), sizeof(FLOAT_T));

    //srand ((unsigned int)time(NULL));
    srand(0);
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1.0)
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }

    for(int i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        coeffs[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        state[i] = static_cast<FLOAT_T>(0);
    }

    start = get_timestamp();

#if defined ENABLE_DEBUG
        std::cout << "Coefficients: \n";
        for (int i = 0; i < FIR_ORDER; i++) {
            std::cout << coeffs[i] << " ";
        }
        std::cout <<"\nFIR evolution: \n";
#endif

    for(int i = 0; i < ARRAY_SIZE; i++) {
        // update state
        state[state_begin] = x[i];
        state_begin = (state_begin + 1) % FIR_ORDER;
        // calculate output
        y[i] = static_cast<FLOAT_T>(0);
        for(int j = 0; j < FIR_ORDER; j++) {
            // We will assume here that coefficients are inversed, that is coeffs[0]
            // represents gain for the oldest sample, and coeffs[FIR_ORDER] - for the 
            // most recent one.
            y[i] = y[i] + state[(state_begin + j) % FIR_ORDER] * coeffs[j];
        }

#if defined ENABLE_DEBUG
        std::cout << "\nx: " << x[i];
        std::cout << "\nstate: \n";
        for (int j = 0; j < FIR_ORDER; j++) {
            std::cout << state[(state_begin + j) % FIR_ORDER] << " ";
        }
        std::cout << "\ny: " << y[i];
#endif
    }
    end = get_timestamp();

    // Perform reduction to avoid dead-code removals.
    volatile FLOAT_T red = static_cast<FLOAT_T>(0);
    for(int i = 0; i < ARRAY_SIZE; i++) {
        red += y[i];
    }
    // cast to void to avoid reduction
    (void)red;
    
    UME::DynamicMemory::AlignedFree(x);
    UME::DynamicMemory::AlignedFree(y);

    return end - start;
}

int main()
{
    std::cout << "All timing results in nanoseconds. \n"
                 "Speedup calculated with scalar floating point result as reference.\n\n"
                 "SIMD version uses following operations: \n"
                 "int: LOAD-CONSTR, SWIZZLEA\n"
                 "float: LOAD-CONSTR, GATHERV, MULV, HADD\n"
                 "swizzle: LOAD-CONSTR\n"
                 "\n";
/*
    {
        TimingStatistics stats_scalar_f, stats_scalar_d;

        for (int i = 0; i < ITERATIONS; i++) {
            TIMING_RES t0 = test_scalar<float, 4>();
            stats_scalar_f.update(t0);
        }
        
        std::cout << "Scalar code (float): " << (unsigned long long) stats_scalar_f.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
            << " (speedup: 1.0x)\n";

        for (int i = 0; i < ITERATIONS; i++) {
            stats_scalar_d.update(test_scalar<double, 4>());
        }

        std::cout << "Scalar code (double): " << (unsigned long long) stats_scalar_d.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
            << " (speedup: " << stats_scalar_d.calculateSpeedup(stats_scalar_f) << "x)\n";

    }*/
    
    std::cout << "\n\nFIR-4:\n";
    {
        TimingStatistics stats_scalar_f, stats_scalar_d;

        for (int i = 0; i < ITERATIONS; i++) {
            TIMING_RES t0 = test_scalar<float, 4>();
            stats_scalar_f.update(t0);
        }

        std::cout << "Scalar code (float): " << (unsigned long long) stats_scalar_f.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
            << " (speedup: 1.0x)\n";
        
        for (int i = 0; i < ITERATIONS; i++) {
            stats_scalar_d.update(test_scalar<double, 4>());
        }
        
        std::cout << "Scalar code (double): " << (unsigned long long) stats_scalar_d.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
            << " (speedup: " << stats_scalar_d.calculateSpeedup(stats_scalar_f) << "x)\n";

        benchmarkSIMD<float, 4>("SIMD code (4x32f): ", ITERATIONS, stats_scalar_f);
        benchmarkSIMD<double, 4>("SIMD code (4x64f): ", ITERATIONS, stats_scalar_f);

        benchmarkSIMD_FIR4<float>("SIMD code (4x32f, fixed permute): ", ITERATIONS, stats_scalar_f);
        benchmarkSIMD_FIR4<double>("SIMD code (4x64f, fixed permute): ", ITERATIONS, stats_scalar_f);
    }

    std::cout << "\n\nFIR-8:\n";
    {
        TimingStatistics stats_scalar_f, stats_scalar_d;

        for (int i = 0; i < ITERATIONS; i++) {
            TIMING_RES t0 = test_scalar<float, 8>();
            stats_scalar_f.update(t0);
        }

        std::cout << "Scalar code (float): " << (unsigned long long) stats_scalar_f.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
            << " (speedup: 1.0x)\n";

        for (int i = 0; i < ITERATIONS; i++) {
            stats_scalar_d.update(test_scalar<double, 8>());
        }
        
        std::cout << "Scalar code (double): " << (unsigned long long) stats_scalar_d.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
            << " (speedup: " << stats_scalar_d.calculateSpeedup(stats_scalar_f) << "x)\n";
            
        benchmarkIntel_FIR8_float("AVX2 Intrinsics: ", ITERATIONS, stats_scalar_f);

        benchmarkSIMD<float, 8>("SIMD code (8x32f): ", ITERATIONS, stats_scalar_f);
        benchmarkSIMD<double, 8>("SIMD code (8x64f): ", ITERATIONS, stats_scalar_f);
        
        benchmarkSIMD_FIR8<float>("SIMD code (8x32f, fixed permute): ", ITERATIONS, stats_scalar_f);
        benchmarkSIMD_FIR8<double>("SIMD code (8x64f, fixed permute): ", ITERATIONS, stats_scalar_f);
    }

    std::cout << "\n\nFIR-16:\n";
    {
        TimingStatistics stats_scalar_f, stats_scalar_d;

        for (int i = 0; i < ITERATIONS; i++) {
            TIMING_RES t0 = test_scalar<float, 16>();
            stats_scalar_f.update(t0);
        }

        std::cout << "Scalar code (float): " << (unsigned long long) stats_scalar_f.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
            << " (speedup: 1.0x)\n";

        for (int i = 0; i < ITERATIONS; i++) {
            stats_scalar_d.update(test_scalar<double, 16>());
        }

        std::cout << "Scalar code (double): " << (unsigned long long) stats_scalar_d.getAverage()
            << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
            << " (speedup: " << stats_scalar_d.calculateSpeedup(stats_scalar_f) << "x)\n";

            
        benchmarkIntel_FIR16_float("AVX512 Intrinsics: ", ITERATIONS, stats_scalar_f);

        benchmarkSIMD<float, 16>("SIMD code (16x32f): ", ITERATIONS, stats_scalar_f);
        benchmarkSIMD<double, 16>("SIMD code (16x64f): ", ITERATIONS, stats_scalar_f);

        benchmarkSIMD_FIR16<float>("SIMD code (16x32f, fixed permute): ", ITERATIONS, stats_scalar_f);
        benchmarkSIMD_FIR16<double>("SIMD code (16x64f, fixed permute): ", ITERATIONS, stats_scalar_f);
    }

    return 0;
}