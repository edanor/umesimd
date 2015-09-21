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

typedef unsigned long long TIMING_RES;

const int ARRAY_SIZE = 600000; // Array size increased to show the peeling effect.
//alignas(32) float x[ARRAY_SIZE];

template<typename FLOAT_T>
TIMING_RES test_scalar()
{
    unsigned long long start, end; // Time measurements
    FLOAT_T a[17];
    FLOAT_T *x;

    x = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), sizeof(FLOAT_T));

    srand ((unsigned int)time(NULL));
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX/1000);
    }

    for(int i = 0; i < 17; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        a[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }
    
    // The polynomial calculated is as follows:
    // y(x) = a0       + a1*x     + a2*x^2   + a3*x^3   + a4*x^4   + a5*x^5   +
    //      + a6*x^6   + a7*x^7   + a8*x^8   + a9*x^9   + a10*x^10 + a11*x^11 +
    //      + a12*x^12 + a13*x^13 + a14*x^14 + a15*x^15 + a16*x^16
    //
    // With Estrin's scheme it can be simplified:
    // y(x) = (a0 + a1*x)  + x^2*(a2  + a3*x)  + x^4*(a4  + a5*x  + x^2*(a6  + a7*x)) +
    //      + x^8*(a8+a9*x + x^2*(a10 + a11*x) + x^4*(a12 + a13*x + x^2*(a14 + a15*x)) +
    //      + a16*x^16
    //
    
    FLOAT_T x2, x4, x8, x16;
    volatile FLOAT_T y;
    
    start = __rdtsc();

    for(int i = 0; i < ARRAY_SIZE; i++) {
        x2  = x[i]*x[i];
        x4  = x2*x2;
        x8  = x4*x4;
        x16 = x8*x8;

        y = (a[0] + a[1]*x[i]) 
            + x2*(a[2] + a[3]*x[i]) 
            + x4*(a[4] + a[5]*x[i] + x2*(a[6] + a[7]*x[i]))
            + x8*(a[8] + a[9]*x[i] + x2*(a[10]+a[11]*x[i]) + x4*(a[12] + a[13]*x[i] + x2*(a[14] + a[15]*x[i])))
            + x16*a[16];
    }

    end = __rdtsc();

    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}


template<typename FLOAT_T, typename FLOAT_VEC_TYPE>
TIMING_RES test_SIMD()
{
    unsigned long long start, end; // Time measurements
    FLOAT_T a[17];
    FLOAT_T *x;
    FLOAT_T *y;

    x = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(FLOAT_T), FLOAT_VEC_TYPE::alignment());
    y = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(FLOAT_VEC_TYPE::length()*sizeof(FLOAT_T), FLOAT_VEC_TYPE::alignment());

    srand ((unsigned int)time(NULL));
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX/1000);
    }

    for(int i = 0; i < 17; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        a[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
    }
    
    // The polynomial calculated is as follows:
    // y(x) = a0       + a1*x     + a2*x^2   + a3*x^3   + a4*x^4   + a5*x^5   +
    //      + a6*x^6   + a7*x^7   + a8*x^8   + a9*x^9   + a10*x^10 + a11*x^11 +
    //      + a12*x^12 + a13*x^13 + a14*x^14 + a15*x^15 + a16*x^16
    //
    // With Estrin's scheme it can be simplified:
    // y(x) = (a0 + a1*x)  + x^2*(a2  + a3*x)  + x^4*(a4  + a5*x  + x^2*(a6  + a7*x)) +
    //      + x^8*(a8+a9*x + x^2*(a10 + a11*x) + x^4*(a12 + a13*x + x^2*(a14 + a15*x)) +
    //      + a16*x^16
    //
    
    FLOAT_VEC_TYPE x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
    FLOAT_VEC_TYPE y_vec;
    FLOAT_VEC_TYPE t0, t1, t2, t3;

    start = __rdtsc();

    for(int i = 0; i < ARRAY_SIZE; i+= FLOAT_VEC_TYPE::length()) {
        x_vec.loada(&x[i]);
        x2_vec  = x_vec.mul(x_vec);
        x4_vec  = x2_vec.mul(x2_vec);
        x8_vec  = x4_vec.mul(x4_vec);
        x16_vec = x8_vec.mul(x8_vec);

        /*
        y_vec = (a[0] + a[1]*x_vec) 
            + x2_vec*(a[2] + a[3]*x_vec) 
            + x4_vec*(a[4] + a[5]*x_vec + x2_vec*(a[6] + a[7]*x_vec))
            + x8_vec*(a[8] + a[9]*x_vec + x2_vec*(a[10]+a[11]*x_vec) 
                    + x4_vec*(a[12] + a[13]*x_vec + x2_vec*(a[14] + a[15]*x_vec)))
            + x16_vec*a[16];
            
        // The polynomial can be reformulated
        y_vec = (x_vec*a[1] + a[0])
              + x2_vec*(x_vec*a[3] + a[2])
              + x4_vec*(x2_vec*(x_vec*a[7] + a[6]) + x_vec*a[5] + a[4]))
              + x8_vec*(x4_vec*(x2_vec*(x_vec*a[15] + a[14])+ x_vec*a[13] + a[12]) 
                        + x2_vec*(x_vec*a[11] + a[10]) + x_vec*a[9] + a[8])
              + x16_vec*a[16];
        */

        y_vec = x_vec.fmuladd(FLOAT_VEC_TYPE(a[1]), FLOAT_VEC_TYPE(a[0]));
        t0 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[3]), FLOAT_VEC_TYPE(a[2]));
        y_vec.adda(x2_vec.mul(t0));
        t0 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[7]), FLOAT_VEC_TYPE(a[6]));
        t1 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[5]), FLOAT_VEC_TYPE(a[4]));
        t2 = x2_vec.fmuladd(t0, t1);
        y_vec.adda(x4_vec.mul(t2));
        t0 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[15]), FLOAT_VEC_TYPE(a[14]));
        t1 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[13]), FLOAT_VEC_TYPE(a[12]));
        t2 = x2_vec.fmuladd(t0, t1);
        t0 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[11]), FLOAT_VEC_TYPE(a[10]));
        t1 = x_vec.fmuladd(FLOAT_VEC_TYPE(a[9]), FLOAT_VEC_TYPE(a[8]));
        t3 = x2_vec.fmuladd(t0, t1);
        t0 = x4_vec.fmuladd(t2, t3);
        y_vec.adda(x8_vec.mul(t0));
        y_vec.adda(x16_vec.mul(a[16]));

        y_vec.storea(&y[0]);
    }

    end = __rdtsc();

    UME::DynamicMemory::AlignedFree(y);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}




int main()
{
    TIMING_RES t_scalar_f,
               t_scalar_d,
               t_SIMD1_32f,
               t_SIMD2_32f,
               t_SIMD4_32f,
               t_SIMD8_32f,
               t_SIMD16_32f,
               t_SIMD32_32f;
    float t_scalar_f_avg = 0.0f,
          t_scalar_d_avg = 0.0f,
          t_SIMD1_32f_avg = 0.0f,
          t_SIMD2_32f_avg = 0.0f,
          t_SIMD4_32f_avg = 0.0f,
          t_SIMD8_32f_avg = 0.0f,
          t_SIMD16_32f_avg = 0.0f,
          t_SIMD32_32f_avg = 0.0f;

    // Run each timing test 100 times
    for(int i = 0; i < 100; i++)
    {
        t_scalar_f = test_scalar<float>();
        t_scalar_f_avg = 1.0f/(1.0f + float(i)) * (float(t_scalar_f) - t_scalar_f_avg);
        
        t_scalar_d = test_scalar<double>();
        t_scalar_d_avg = 1.0f/(1.0f + float(i)) * (float(t_scalar_d) - t_scalar_d_avg);

        t_SIMD1_32f = test_SIMD<float, UME::SIMD::SIMD1_32f>();
        t_SIMD1_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_SIMD1_32f) - t_SIMD1_32f_avg);
        
        t_SIMD2_32f = test_SIMD<float, UME::SIMD::SIMD2_32f>();
        t_SIMD2_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_SIMD2_32f) - t_SIMD2_32f_avg);

        t_SIMD4_32f = test_SIMD<float, UME::SIMD::SIMD4_32f>();
        t_SIMD4_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_SIMD4_32f) - t_SIMD4_32f_avg);

        t_SIMD8_32f = test_SIMD<float, UME::SIMD::SIMD8_32f>();
        t_SIMD8_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_SIMD8_32f) - t_SIMD8_32f_avg);

        t_SIMD16_32f = test_SIMD<float, UME::SIMD::SIMD16_32f>();
        t_SIMD16_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_SIMD16_32f) - t_SIMD16_32f_avg);

        t_SIMD32_32f = test_SIMD<float, UME::SIMD::SIMD32_32f>();
        t_SIMD32_32f_avg = 1.0f/(1.0f + float(i)) * (float(t_SIMD32_32f) - t_SIMD32_32f_avg);

    }

    std::cout << "Scalar code (float): " << (long)t_scalar_f_avg
                                         << "\t(speedup: 1.0x)"
                                         << std::endl;

    std::cout << "Scalar code (double): " << (long)t_scalar_d_avg
                                          << "\t(speedup: "
                                          << float(t_scalar_f_avg)/float(t_scalar_d_avg)
                                          << std::endl;

    std::cout << "SIMD code (1x32f): "  << (long) t_SIMD1_32f_avg
                                        << "\t(speedup: "
                                        << float(t_scalar_f_avg)/float(t_SIMD1_32f_avg)
                                        << std::endl;
    
    std::cout << "SIMD code (2x32f): "  << (long) t_SIMD2_32f_avg
                                        << "\t(speedup: "
                                        << float(t_scalar_f_avg)/float(t_SIMD2_32f_avg)
                                        << std::endl;
    
    std::cout << "SIMD code (4x32f): "  << (long) t_SIMD4_32f_avg
                                        << "\t(speedup: "
                                        << float(t_scalar_f_avg)/float(t_SIMD4_32f_avg)
                                        << std::endl;
    
    std::cout << "SIMD code (8x32f): "  << (long) t_SIMD8_32f_avg
                                        << "\t(speedup: "
                                        << float(t_scalar_f_avg)/float(t_SIMD8_32f_avg)
                                        << std::endl;

    std::cout << "SIMD code (16x32f): "  << (long) t_SIMD16_32f_avg
                                        << "\t(speedup: "
                                        << float(t_scalar_f_avg)/float(t_SIMD16_32f_avg)
                                        << std::endl;
    
    std::cout << "SIMD code (32x32f): "  << (long) t_SIMD32_32f_avg
                                        << "\t(speedup: "
                                        << float(t_scalar_f_avg)/float(t_SIMD32_32f_avg)
                                        << std::endl;
    return 0;
}