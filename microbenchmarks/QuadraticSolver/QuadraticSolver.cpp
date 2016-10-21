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

#include <time.h>

#include "../../UMESimd.h"
#include "../utilities/TimingStatistics.h"

// This size effectively gives 200 MB of allocation for single precision, and 400 MB for double precision
// on x86 compatible machine. This means that data will travel between main memory and cache on most of
// existing machines.
const int ARRAY_SIZE = 1024 * 1024 * 8;

// Different implementations:
#include "QuadraticSolverNaive.h"
#include "QuadraticSolverOptimized.h"
#include "QuadraticSolverSIMD.h"
#include "QuadraticSolverAVX2.h"
#include "QuadraticSolverSIMD_nontemplate.h"

TimingStatistics benchmarkScalarNaiveFloat(std::string resultPrefix, int iterations)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++) {
        stats.update(run_scalar_naive<float, int32_t>());
    }

    std::cout << resultPrefix.c_str() << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: 1.0x)\n";

    return stats;
}

void benchmarkScalarNaiveDouble(
    std::string resultPrefix, 
    int iterations, 
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++) {
        stats.update(run_scalar_naive<double, int64_t>());
    }

    std::cout << resultPrefix.c_str() << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

template <typename FLOAT_T>
void benchmarkScalarOptimized(
    std::string resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++) {
        stats.update(run_scalar_optimized<FLOAT_T>());
    }

    std::cout << resultPrefix.c_str() << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}


template<typename FLOAT_T, uint32_t LENGTH>
void benchmarkSIMD(std::string const & resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(run_SIMD<FLOAT_T, LENGTH>());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

#ifdef __AVX2__
void benchmarkAVX2(std::string const & resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(run_AVX2());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}
#endif

void benchmarkSIMD_nontemplate(char * resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(run_SIMD_nontemplate());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

typedef unsigned long long TIMING_RES;
int main()
{
    int ITERATIONS = 20;

    std::cout << "The result is amount of time it takes to calculate solution of: " << ARRAY_SIZE << " quadratic equations.\n"
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD version uses following operations: \n"
        "   32/64f vectors: SET-CONSTR, LOAD-CONSTR, DIVV (operator /), MULV (operator*),\n"
        "                   MULS (operator*), BLENDV, FMULADDV, SQRT, CMPLTV (operator <),\n"
        "                   CMPGTV (operator >=), FTOI, STORE\n"
        "   32/64i vectors: COPY-CONST, STORE, DEGRADE\n"
        "     # of executions per measurement: " << ITERATIONS << "\n\n";


    TimingStatistics ref;

    ref = benchmarkScalarNaiveFloat(std::string("Scalar naive (float):"), ITERATIONS);
    //benchmarkScalarOptimized<float>("Scalar optimized (float):", ITERATIONS, ref);

#ifdef __AVX2__
    benchmarkAVX2("AVX2 intrinsic code (float, 8):", ITERATIONS, ref);
#else
    std::cout << "AVX2 intrinsic code (float, 8): unavailable\n";
#endif

    //benchmarkSIMD_nontemplate("UME::SIMD (float, 8) nontemplate:", ITERATIONS, ref);

    benchmarkSIMD<float, 1>("UME::SIMD (float, 1): ", ITERATIONS, ref);
    benchmarkSIMD<float, 2>("UME::SIMD (float, 2): ", ITERATIONS, ref);
    benchmarkSIMD<float, 4>("UME::SIMD (float, 4): ", ITERATIONS, ref);
    benchmarkSIMD<float, 8>("UME::SIMD (float, 8): ", ITERATIONS, ref);
    benchmarkSIMD<float, 16>("UME::SIMD (float, 16): ", ITERATIONS, ref);
    benchmarkSIMD<float, 32>("UME::SIMD (float, 32): ", ITERATIONS, ref);

    benchmarkScalarNaiveDouble(std::string("Scalar naive (double): "), ITERATIONS, ref);
    //benchmarkScalarOptimized<double>("Scalar optimized (double): ", ITERATIONS, ref);
    benchmarkSIMD<double, 1>("UME::SIMD (double, 1): ", ITERATIONS, ref);
    benchmarkSIMD<double, 2>("UME::SIMD (double, 2): ", ITERATIONS, ref);
    benchmarkSIMD<double, 4>("UME::SIMD (double, 4): ", ITERATIONS, ref);
    benchmarkSIMD<double, 8>("UME::SIMD (double, 8): ", ITERATIONS, ref);
    benchmarkSIMD<double, 16>("UME::SIMD (double, 16): ", ITERATIONS, ref);

    return 0;
}
