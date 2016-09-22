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

#if defined(__i386__) || defined(__x86_64__)
#include <immintrin.h> 
#endif

#include <cmath>
#include <time.h>
#include <stdlib.h>

//#define UME_SIMD_SHOW_EMULATION_WARNINGS 1
#include "../../UMESimd.h"

#include "../utilities/TimingStatistics.h"

// Introducing inline assembly forces compiler to generate
#define BREAK_COMPILER_OPTIMIZATION() __asm__ ("NOP");

const int ARRAY_SIZE = 600000+7; // Array size increased to show the peeling effect.
//alignas(32) float x[ARRAY_SIZE];

// Scalar algorithm
template<typename FLOAT_T>
TIMING_RES test_scalar()
{
    unsigned long long start, end;    // Time measurements
    
    FLOAT_T *x;

    x = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), sizeof(FLOAT_T));

    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX/1000);
    }
    
    {    
        FLOAT_T sum = 0.0f;
        volatile FLOAT_T avg = 0.0f;
    
        start = get_timestamp();
      
        for(int i = 0; i < ARRAY_SIZE; i++)
        {
            sum += x[i];
        }
        
        avg = sum/(FLOAT_T)ARRAY_SIZE;
        
        end = get_timestamp();
    }
    
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}

TIMING_RES test_AVX_f_256()
{
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512__)
    unsigned long long start, end;    // Time measurements
    
    float *x;

    x = (float *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), sizeof(float));

    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/1000);
    }
   
    float sum = 0.0f;
    volatile float avg = 0.0f;
      
    // Calculate loop-peeling division
    int PEEL_COUNT = ARRAY_SIZE/8;             // Divide array size by vector length.
    int REM_COUNT = ARRAY_SIZE - PEEL_COUNT*8; // 
            
    alignas(32) float temp[8];
  
    start = get_timestamp();
      
    __m256 x_vec;
    __m256 sum_vec = _mm256_setzero_ps();
    // Instead of adding single elements, we are using SIMD to add elements
    // with STRIDE-8 distance. We then perform reduction using scalar code
    for(int i = 0; i < PEEL_COUNT; i++)
    {
        x_vec = _mm256_load_ps(&x[i*8]); // load elements with STRIDE-8
        sum_vec = _mm256_add_ps(sum_vec, x_vec); // accumulate sum of values
    }
      
    // Now the reduction operation converting a vector into a scalar value
    _mm256_store_ps(temp, sum_vec);
    for(int i = 0; i < 8; ++i)
    {
        sum += temp[i];  
    }
      
    // Calculating loop reminder
    for(int i = 0; i < REM_COUNT; i++)
    {
        sum += x[PEEL_COUNT*8 + i];
    }
      
    avg = sum/(float)ARRAY_SIZE;
      
    end = get_timestamp();
      
    // Verify the result is correct
    float test_sum = 0.0f;
    float test_avg = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        test_sum += x[i];
    }
      
    test_avg = test_sum/(float)ARRAY_SIZE;
    float normalized_res = avg/test_avg;
    float err_margin = 0.001f;
    if(    normalized_res > (1.0f + err_margin)  
        || normalized_res < (1.0f - err_margin) )
    {
        std::cout << "Result invalid: " << avg << " expected: " << test_avg << std::endl;
    }
    
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
#endif
    return 0;
}

TIMING_RES test_AVX_d_256() {
    
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512__)
    unsigned long long start, end;    // Time measurements
    
   
    double sum = 0.0f;
    volatile double avg = 0.0f;
      
    // Calculate loop-peeling division
    int PEEL_COUNT = ARRAY_SIZE/4;             // Divide array size by vector length.
    int REM_COUNT = ARRAY_SIZE - PEEL_COUNT*4; // 
            
    alignas(32) double temp[4];

    double *x;
    x = (double *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(double), 4*sizeof(double));
        
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX/1000);
    }

    start = get_timestamp();
      
    __m256d x_vec;
    __m256d sum_vec = _mm256_setzero_pd();
    // Instead of adding single elements, we are using SIMD to add elements
    // with STRIDE-8 distance. We then perform reduction using scalar code
    for(int i = 0; i < PEEL_COUNT; i++)
    {
        x_vec = _mm256_load_pd(&x[i*4]); // load elements with STRIDE-8
        sum_vec = _mm256_add_pd(sum_vec, x_vec); // accumulate sum of values
    }
      
    // Now the reduction operation converting a vector into a scalar value
    _mm256_store_pd(temp, sum_vec);
    for(int i = 0; i < 4; ++i)
    {
        sum += temp[i];  
    }
      
    // Calculating loop reminder
    for(int i = 0; i < REM_COUNT; i++)
    {
        sum += x[PEEL_COUNT*4 + i];
    }
      
    avg = sum/(double)ARRAY_SIZE;
      
    end = get_timestamp();
      
    // Verify the result is correct
    double test_sum = 0.0f;
    double test_avg = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        test_sum += x[i];
    }
      
    test_avg = test_sum/(double)ARRAY_SIZE;
    double normalized_res = avg/test_avg;
    double err_margin = 0.001f;
    if(    normalized_res > (1.0f + err_margin)  
        || normalized_res < (1.0f - err_margin) )
    {
        std::cout << "Result invalid: " << avg << " expected: " << test_avg << std::endl;
    }
    
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
#endif
    return 0;
}
TIMING_RES test_AVX_f_512()
{
#if defined(__AVX512__) || defined(__MIC__)
    unsigned long long start, end;    // Time measurements
    
   
    float sum = 0.0f;
    volatile float avg = 0.0f;
      
    // Calculate loop-peeling division
    int PEEL_COUNT = ARRAY_SIZE/16;             // Divide array size by vector length.
    int REM_COUNT = ARRAY_SIZE - PEEL_COUNT*16; // 
            
    alignas(64) float temp[16];
  
    float *x;
    x = (float *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 16*sizeof(float));
    
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/1000);
    }

    start = get_timestamp();
      
    __m512 x_vec;
    __m512 sum_vec = _mm512_setzero_ps();
    // Instead of adding single elements, we are using SIMD to add elements
    // with STRIDE-8 distance. We then perform reduction using scalar code
    for(int i = 0; i < PEEL_COUNT; i++)
    {
        x_vec = _mm512_load_ps(&x[i*16]); // load elements with STRIDE-16
        sum_vec = _mm512_add_ps(sum_vec, x_vec); // accumulate sum of values
    }
      
    // Now the reduction operation converting a vector into a scalar value
    _mm512_store_ps(temp, sum_vec);
    for(int i = 0; i < 16; ++i)
    {
        sum += temp[i];  
    }
      
    // Calculating loop reminder
    for(int i = 0; i < REM_COUNT; i++)
    {
        sum += x[PEEL_COUNT*16 + i];
    }
      
    avg = sum/(float)ARRAY_SIZE;
      
    end = get_timestamp();
      
    // Verify the result is correct
    float test_sum = 0.0f;
    float test_avg = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        test_sum += x[i];
    }
      
    test_avg = test_sum/(float)ARRAY_SIZE;
    float normalized_res = avg/test_avg;
    float err_margin = 0.001f;
    if(    normalized_res > (1.0f + err_margin)  
        || normalized_res < (1.0f - err_margin) )
    {
        std::cout << "Result invalid: " << avg << " expected: " << test_avg << std::endl;
    }
    
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
#endif
    return 0;
}

template<typename FLOAT_VEC_TYPE>
TIMING_RES test_UME_SIMD()
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_TYPE>::SCALAR_T FLOAT_T;

    const uint32_t VEC_LEN = FLOAT_VEC_TYPE::length();
    const int ALIGNMENT = FLOAT_VEC_TYPE::alignment();

    unsigned long long start, end;    // Time measurements
    
    FLOAT_T *x;

    x = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), ALIGNMENT);
    
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX/1000);
    }
   
    FLOAT_T sum = 0.0f;
    volatile FLOAT_T avg = 0.0f;
      
    // Calculate loop-peeling division
    uint32_t PEEL_COUNT = ARRAY_SIZE/VEC_LEN;             // Divide array size by vector length.
    uint32_t REM_COUNT = ARRAY_SIZE - PEEL_COUNT*VEC_LEN; // 
            
    FLOAT_T* temp;
    
    temp = (FLOAT_T*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), ALIGNMENT);

    start = get_timestamp();
      
    FLOAT_VEC_TYPE x_vec;
    FLOAT_VEC_TYPE sum_vec(0.0f);
    // Instead of adding single elements, we are using SIMD to add elements
    // with STRIDE-<VEC_LEN> distance. We then perform reduction using scalar code
    for(uint32_t i = 0; i < PEEL_COUNT; i++)
    {
        x_vec.load(&x[i*VEC_LEN]);
        //x_vec = _mm256_load_ps(&x[i*8]); // load elements with STRIDE-8
        sum_vec.adda(x_vec);
        //sum_vec = _mm256_add_ps(sum_vec, x_vec); // accumulate sum of values
    }
      
    sum_vec.store(temp);

    // TODO: replace with reduce-add
    for(uint32_t i = 0; i < VEC_LEN; ++i)
    {
        sum += temp[i];  
    }
      
    // Calculating loop reminder
    for(uint32_t i = 0; i < REM_COUNT; i++)
    {
        sum += x[PEEL_COUNT*VEC_LEN + i];
    }
      
    avg = sum/(FLOAT_T)ARRAY_SIZE;
      
    end = get_timestamp();
      
    // Verify the result is correct
    FLOAT_T test_sum = 0.0f;
    FLOAT_T test_avg = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        test_sum += x[i];
    }
      
    test_avg = test_sum/(FLOAT_T)ARRAY_SIZE;
    FLOAT_T normalized_res = avg/test_avg;
    FLOAT_T err_margin = 0.001f;
    if(    normalized_res > (1.0f + err_margin)  
        || normalized_res < (1.0f - err_margin) )
    {
            std::cout << "Result invalid: " << avg << " expected: " << test_avg << std::endl;
    }

    UME::DynamicMemory::AlignedFree(temp);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}

template<typename VEC_T>
void benchmarkUMESIMD( std::string const & resultPrefix, int iterations, TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        unsigned long long elapsed = test_UME_SIMD<VEC_T>();
        stats.update(elapsed);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}

int main()
{
    const int ITERATIONS = 1000;

    srand ((unsigned int)time(NULL));

    std::cout << "The result is amount of time it takes to calculate average of: " << ARRAY_SIZE << " elements.\n"
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD version uses following operations: \n"
        " ZERO-CONSTR, SET-CONSTR, LOAD, ADDA, STORE\n";

    TimingStatistics stats_scalar_f,
        stats_scalar_d;

    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_f.update(test_scalar<float>());
    }

    std::cout << "Scalar code (float): " << (unsigned long long)stats_scalar_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
        << " (speedup: 1.0x)"
        << std::endl;

    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_d.update(test_scalar<float>());
    }

    std::cout << "Scalar code (double): " << (unsigned long long)stats_scalar_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
        << " (speedup: "
        << stats_scalar_d.calculateSpeedup(stats_scalar_f) << ")"
        << std::endl;

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512__)
    TimingStatistics stats_avx_f, stats_avx_d;

    for (int i = 0; i < ITERATIONS; i++) {
        stats_avx_f.update(test_AVX_f_256());
    }

    std::cout << "256b intrinsic code (float): " << (unsigned long long)stats_avx_f.getAverage()
        << ", dev: " << (unsigned long long) stats_avx_f.getStdDev()
        << " (speedup: "
        << stats_avx_f.calculateSpeedup(stats_scalar_f) << ")"
        << std::endl;

    for (int i = 0; i < ITERATIONS; i++) {
        stats_avx_d.update(test_AVX_d_256());
    }

    std::cout << "256b intrinsic code (double): " << (unsigned long long)stats_avx_d.getAverage()
        << ", dev: " << (unsigned long long) stats_avx_d.getStdDev()
        << " (speedup: "
        << stats_avx_d.calculateSpeedup(stats_scalar_f) << ")"
        << std::endl;
#else
    std::cout << "256b intrinsic code:     AVX/AVX2/AVX512 disabled, cannot run measurement\n";
#endif

#if defined(__AVX512__) || defined(__MIC__)
    TimingStatistics stats_avx512_f, stats_avx512_d;

    for (int i = 0; i < ITERATIONS; i++) {
        stats_avx512_f.update(test_AVX_f_512());
    }

    std::cout << "512b intrinsic code (float): " << (unsigned long long)stats_avx512_f.getAverage()
        << ", dev: " << (unsigned long long) stats_avx512_f.getStdDev()
        << " (speedup: "
        << stats_avx512_f.calculateSpeedup(stats_scalar_f) << ")"
        << std::endl;

#else
    std::cout << "512b intrinsic code:     AVX512/KNC disabled, cannot run measurement\n";
#endif

    benchmarkUMESIMD<UME::SIMD::SIMD1_32f>("SIMD code(1x32f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_32f>("SIMD code(2x32f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_32f>("SIMD code(4x32f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_32f>("SIMD code(8x32f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_32f>("SIMD code(16x32f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD32_32f>("SIMD code(32x32f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD1_64f>("SIMD code(1x64f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_64f>("SIMD code(2x64f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_64f>("SIMD code(4x64f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_64f>("SIMD code(8x64f) :", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_64f>("SIMD code(16x64f) :", ITERATIONS, stats_scalar_f);

    return 0;
}
