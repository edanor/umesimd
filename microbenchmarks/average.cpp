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

#include <immintrin.h> 
#include <cmath>
#include <time.h>
#include <stdlib.h>

//#define UME_SIMD_SHOW_EMULATION_WARNINGS 1
#include "../UMESimd.h"

// Introducing inline assembly forces compiler to generate
#define BREAK_COMPILER_OPTIMIZATION() __asm__ ("NOP");

// define RDTSC getter function
#if defined(__i386__)
static __inline__ unsigned long long __rdtsc(void)
{
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}
#elif defined(__x86_64__)
static __inline__ unsigned long long __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}
#endif

const int ARRAY_SIZE = 600000+7; // Array size increased to show the peeling effect.
alignas(32) float x[ARRAY_SIZE];

typedef unsigned long long TIMING_RES;

// Scalar algorithm
TIMING_RES test_scalar()
{
    unsigned long long start, end;    // Time measurements
    
    srand ((unsigned int)time(NULL));
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/1000);
    }
    
    {    
        float sum = 0.0f;
        volatile float avg = 0.0f;
    
        start = __rdtsc();
      
        for(int i = 0; i < ARRAY_SIZE; i++)
        {
            sum += x[i];
        }
        
        avg = sum/(float)ARRAY_SIZE;
        
        end = __rdtsc();
    }

    return end - start;
}

TIMING_RES test_AVX_256()
{
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512__)
    unsigned long long start, end;    // Time measurements
    
    srand ((unsigned int)time(NULL));
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
  
    start = __rdtsc();
      
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
      
    end = __rdtsc();
      
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
    
    return end - start;
#endif
    return 0;
}

template<typename FLOAT_VEC_TYPE>
TIMING_RES test_UME_SIMD()
{
    const uint32_t VEC_LEN = FLOAT_VEC_TYPE::length();
    const int ALIGNMENT = FLOAT_VEC_TYPE::alignment();

    unsigned long long start, end;    // Time measurements
    
    srand ((unsigned int)time(NULL));
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX/1000);
    }
   
    float sum = 0.0f;
    volatile float avg = 0.0f;
      
    // Calculate loop-peeling division
    int PEEL_COUNT = ARRAY_SIZE/VEC_LEN;             // Divide array size by vector length.
    int REM_COUNT = ARRAY_SIZE - PEEL_COUNT*VEC_LEN; // 
            
    float* temp;
    temp = (float*) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), ALIGNMENT);

    start = __rdtsc();
      
    FLOAT_VEC_TYPE x_vec;
    FLOAT_VEC_TYPE sum_vec(0.0f);
    // Instead of adding single elements, we are using SIMD to add elements
    // with STRIDE-8 distance. We then perform reduction using scalar code
    for(int i = 0; i < PEEL_COUNT; i++)
    {
        x_vec.loada(&x[i*VEC_LEN]);
        //x_vec = _mm256_load_ps(&x[i*8]); // load elements with STRIDE-8
        sum_vec.adda(x_vec);
        //sum_vec = _mm256_add_ps(sum_vec, x_vec); // accumulate sum of values
    }
      
    // Now the reduction operation converting a vector into a scalar value
    sum_vec.storea(temp);
    
    // TODO: replace with reduce-add
    for(int i = 0; i < VEC_LEN; ++i)
    {
        sum += temp[i];  
    }
      
    // Calculating loop reminder
    for(int i = 0; i < REM_COUNT; i++)
    {
        sum += x[PEEL_COUNT*VEC_LEN + i];
    }
      
    avg = sum/(float)ARRAY_SIZE;
      
    end = __rdtsc();
      
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

    UME::DynamicMemory::AlignedFree(temp);

    return end - start;
}

int main()
{
    TIMING_RES t_scalar, t_AVX, t_UME_SIMD8_32f, t_UME_SIMD4_32f;
    float t_scalar_avg = 0.0f, 
          t_AVX_avg = 0.0f, 
          t_UME_SIMD8_32f_avg = 0.0f,
          t_UME_SIMD4_32f_avg = 0.0f;

    // Run each timing test 1000 times
    for(int i = 0; i < 100; i++)
    {
         t_scalar = test_scalar();
         t_scalar_avg = 1.0f/(1.0f + float(i)) * (float(t_scalar) - t_scalar_avg);
         
         t_AVX = test_AVX_256();
         t_AVX_avg = 1.0f/(1.0f + float(i)) * (float(t_AVX) - t_AVX_avg);

         t_UME_SIMD8_32f = test_UME_SIMD<UME::SIMD::SIMD8_32f>();
         t_UME_SIMD8_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_UME_SIMD8_32f) - t_UME_SIMD8_32f_avg);
         
         t_UME_SIMD4_32f = test_UME_SIMD<UME::SIMD::SIMD4_32f>();
         t_UME_SIMD4_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_UME_SIMD4_32f) - t_UME_SIMD4_32f_avg);
         
    }

    std::cout << "Scalar code:\n\tTiming:             " << (long)t_scalar_avg
                                                        << "\t(speedup: 1.0x)" 
                                                        <<  std::endl;
    std::cout << "256b intrinsic code:\n\tTiming:     " << (long)t_AVX_avg 
                                                        <<  "\t(speedup: " 
                                                        << float(t_scalar_avg)/float(t_AVX_avg) << ")" 
                                                        << std::endl;
    std::cout << "UME::SIMD::8_32f (256b):\n\tTiming: " << (long)t_UME_SIMD8_32f_avg 
                                                        << "\t(speedup: "
                                                        << float(t_scalar_avg)/float(t_UME_SIMD8_32f_avg) << ")" 
                                                        << std::endl;
    std::cout << "UME::SIMD::4_32f (128b):\n\tTiming: " << (long)t_UME_SIMD4_32f_avg 
                                                        << "\t(speedup: "
                                                        << float(t_scalar_avg)/float(t_UME_SIMD4_32f_avg) << ")" 
                                                        << std::endl;

    return 0;
}
