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

// Introducing inline assembly forces compiler to generate
#define BREAK_COMPILER_OPTIMIZATION() __asm__ ("NOP");

const int ARRAY_SIZE = 60000; // Array size increased to show the peeling effect.
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
        // Generate random numbers in range (0.0;1.0)
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
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

    start = get_timestamp();

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

    end = get_timestamp();

    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}

//#if defined(__AVX512F__)
// TODO: implementation for AVX512 required
#if defined(__AVX2__) || defined(__AVX__)
TIMING_RES test_avx_32f()
{
    unsigned long long start, end; // Time measurements
    float a[17];
    float *x;
    float *y;

    x = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 32);
    y = (float *)UME::DynamicMemory::AlignedMalloc(8*sizeof(float), 32);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    for (int i = 0; i < 17; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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

    __m256 x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
    __m256 y_vec;
    __m256 t0, t1, t2, t3;

    __m256 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;

    a0 = _mm256_set1_ps(a[0]);
    a1 = _mm256_set1_ps(a[1]);
    a2 = _mm256_set1_ps(a[2]);
    a3 = _mm256_set1_ps(a[3]);
    a4 = _mm256_set1_ps(a[4]);
    a5 = _mm256_set1_ps(a[5]);
    a6 = _mm256_set1_ps(a[6]);
    a7 = _mm256_set1_ps(a[7]);
    a8 = _mm256_set1_ps(a[8]);
    a9 = _mm256_set1_ps(a[9]);
    a10 = _mm256_set1_ps(a[10]);
    a11 = _mm256_set1_ps(a[11]);
    a12 = _mm256_set1_ps(a[12]);
    a13 = _mm256_set1_ps(a[13]);
    a14 = _mm256_set1_ps(a[14]);
    a15 = _mm256_set1_ps(a[15]);
    a16 = _mm256_set1_ps(a[16]);

    start = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        //x_vec.load(&x[i]);
        x_vec = _mm256_load_ps(&x[i]);

        //x2_vec = x_vec.mul(x_vec);
        x2_vec = _mm256_mul_ps(x_vec, x_vec);
        //x4_vec = x2_vec.mul(x2_vec);
        x4_vec = _mm256_mul_ps(x2_vec, x2_vec);
        //x8_vec = x4_vec.mul(x4_vec);
        x8_vec = _mm256_mul_ps(x4_vec, x4_vec);
        //x16_vec = x8_vec.mul(x8_vec);
        x16_vec = _mm256_mul_ps(x8_vec, x8_vec);

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

        //y_vec = x_vec.fmuladd(a1, a0);
        y_vec = _mm256_add_ps(_mm256_mul_ps(x_vec, a1), a0);
        //t0 = x_vec.fmuladd(a3, a2);
        t0 = _mm256_add_ps(_mm256_mul_ps(x_vec, a3), a2);
        //y_vec.adda(x2_vec.mul(t0));
        y_vec = _mm256_add_ps(y_vec, _mm256_mul_ps(x2_vec, t0));

        //t0 = x_vec.fmuladd(a7, a6);
        t0 = _mm256_add_ps(_mm256_mul_ps(x_vec, a7), a6);
        //t1 = x_vec.fmuladd(a5, a4);
        t1 = _mm256_add_ps(_mm256_mul_ps(x_vec, a5), a4);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm256_add_ps(_mm256_mul_ps(x2_vec, t0), t1);
        //y_vec.adda(x4_vec.mul(t2));
        y_vec = _mm256_add_ps(y_vec, _mm256_mul_ps(x4_vec, t2));

        //t0 = x_vec.fmuladd(a15, a14);
        t0 = _mm256_add_ps(_mm256_mul_ps(x_vec, a15), a14);
        //t1 = x_vec.fmuladd(a13, a12);
        t1 = _mm256_add_ps(_mm256_mul_ps(x_vec, a13), a12);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm256_add_ps(_mm256_mul_ps(x2_vec, t0), t1);
        //t0 = x_vec.fmuladd(a11, a10);
        t0 = _mm256_add_ps(_mm256_mul_ps(x_vec, a11), a10);
        //t1 = x_vec.fmuladd(a9, a8);
        t1 = _mm256_add_ps(_mm256_mul_ps(x_vec, a9), a8);
        //t3 = x2_vec.fmuladd(t0, t1);
        t3 = _mm256_add_ps(_mm256_mul_ps(x2_vec, t0), t1);
        //t0 = x4_vec.fmuladd(t2, t3);
        t0 = _mm256_add_ps(_mm256_mul_ps(x4_vec, t2), t3);
        //y_vec.adda(x8_vec.mul(t0));
        y_vec = _mm256_add_ps(y_vec, _mm256_mul_ps(x8_vec, t0));

        //y_vec.adda(x16_vec.mul(a[16]));
        y_vec = _mm256_add_ps(y_vec, _mm256_mul_ps(x16_vec, a16));

        //y_vec.store(&y[0]);
        _mm256_store_ps(&y[0], y_vec);
    }

    end = get_timestamp();

    UME::DynamicMemory::AlignedFree(y);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}

TIMING_RES test_avx_64f()
{
    unsigned long long start, end; // Time measurements
    double a[17];
    double *x;
    double *y;

    x = (double *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(double), 32);
    y = (double *)UME::DynamicMemory::AlignedMalloc(4 * sizeof(double), 32);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    for (int i = 0; i < 17; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        a[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
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

    __m256d x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
    __m256d y_vec;
    __m256d t0, t1, t2, t3;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;

    a0 = _mm256_set1_pd(a[0]);
    a1 = _mm256_set1_pd(a[1]);
    a2 = _mm256_set1_pd(a[2]);
    a3 = _mm256_set1_pd(a[3]);
    a4 = _mm256_set1_pd(a[4]);
    a5 = _mm256_set1_pd(a[5]);
    a6 = _mm256_set1_pd(a[6]);
    a7 = _mm256_set1_pd(a[7]);
    a8 = _mm256_set1_pd(a[8]);
    a9 = _mm256_set1_pd(a[9]);
    a10 = _mm256_set1_pd(a[10]);
    a11 = _mm256_set1_pd(a[11]);
    a12 = _mm256_set1_pd(a[12]);
    a13 = _mm256_set1_pd(a[13]);
    a14 = _mm256_set1_pd(a[14]);
    a15 = _mm256_set1_pd(a[15]);
    a16 = _mm256_set1_pd(a[16]);

    start = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i += 4) {
        //x_vec.load(&x[i]);
        x_vec = _mm256_load_pd(&x[i]);

        //x2_vec = x_vec.mul(x_vec);
        x2_vec = _mm256_mul_pd(x_vec, x_vec);
        //x4_vec = x2_vec.mul(x2_vec);
        x4_vec = _mm256_mul_pd(x2_vec, x2_vec);
        //x8_vec = x4_vec.mul(x4_vec);
        x8_vec = _mm256_mul_pd(x4_vec, x4_vec);
        //x16_vec = x8_vec.mul(x8_vec);
        x16_vec = _mm256_mul_pd(x8_vec, x8_vec);

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

        //y_vec = x_vec.fmuladd(a1, a0);
        y_vec = _mm256_add_pd(_mm256_mul_pd(x_vec, a1), a0);
        //t0 = x_vec.fmuladd(a3, a2);
        t0 = _mm256_add_pd(_mm256_mul_pd(x_vec, a3), a2);
        //y_vec.adda(x2_vec.mul(t0));
        y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(x2_vec, t0));

        //t0 = x_vec.fmuladd(a7, a6);
        t0 = _mm256_add_pd(_mm256_mul_pd(x_vec, a7), a6);
        //t1 = x_vec.fmuladd(a5, a4);
        t1 = _mm256_add_pd(_mm256_mul_pd(x_vec, a5), a4);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm256_add_pd(_mm256_mul_pd(x2_vec, t0), t1);
        //y_vec.adda(x4_vec.mul(t2));
        y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(x4_vec, t2));

        //t0 = x_vec.fmuladd(a15, a14);
        t0 = _mm256_add_pd(_mm256_mul_pd(x_vec, a15), a14);
        //t1 = x_vec.fmuladd(a13, a12);
        t1 = _mm256_add_pd(_mm256_mul_pd(x_vec, a13), a12);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm256_add_pd(_mm256_mul_pd(x2_vec, t0), t1);
        //t0 = x_vec.fmuladd(a11, a10);
        t0 = _mm256_add_pd(_mm256_mul_pd(x_vec, a11), a10);
        //t1 = x_vec.fmuladd(a9, a8);
        t1 = _mm256_add_pd(_mm256_mul_pd(x_vec, a9), a8);
        //t3 = x2_vec.fmuladd(t0, t1);
        t3 = _mm256_add_pd(_mm256_mul_pd(x2_vec, t0), t1);
        //t0 = x4_vec.fmuladd(t2, t3);
        t0 = _mm256_add_pd(_mm256_mul_pd(x4_vec, t2), t3);
        //y_vec.adda(x8_vec.mul(t0));
        y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(x8_vec, t0));

        //y_vec.adda(x16_vec.mul(a[16]));
        y_vec = _mm256_add_pd(y_vec, _mm256_mul_pd(x16_vec, a16));

        //y_vec.store(&y[0]);
        _mm256_store_pd(&y[0], y_vec);
    }

    end = get_timestamp();

    UME::DynamicMemory::AlignedFree(y);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}
#endif

#if defined(__AVX512F__)
TIMING_RES test_avx512_32f()
{
    unsigned long long start, end; // Time measurements
    float a[17];
    float *x;
    float *y;

    x = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 64);
    y = (float *)UME::DynamicMemory::AlignedMalloc(16*sizeof(float), 64);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    for (int i = 0; i < 17; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        a[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
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

    __m512 x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
    __m512 y_vec;
    __m512 t0, t1, t2, t3;

    __m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;

    a0 = _mm512_set1_ps(a[0]);
    a1 = _mm512_set1_ps(a[1]);
    a2 = _mm512_set1_ps(a[2]);
    a3 = _mm512_set1_ps(a[3]);
    a4 = _mm512_set1_ps(a[4]);
    a5 = _mm512_set1_ps(a[5]);
    a6 = _mm512_set1_ps(a[6]);
    a7 = _mm512_set1_ps(a[7]);
    a8 = _mm512_set1_ps(a[8]);
    a9 = _mm512_set1_ps(a[9]);
    a10 = _mm512_set1_ps(a[10]);
    a11 = _mm512_set1_ps(a[11]);
    a12 = _mm512_set1_ps(a[12]);
    a13 = _mm512_set1_ps(a[13]);
    a14 = _mm512_set1_ps(a[14]);
    a15 = _mm512_set1_ps(a[15]);
    a16 = _mm512_set1_ps(a[16]);

    start = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i += 16) {
        //x_vec.load(&x[i]);
        x_vec = _mm512_load_ps(&x[i]);

        //x2_vec = x_vec.mul(x_vec);
        x2_vec = _mm512_mul_ps(x_vec, x_vec);
        //x4_vec = x2_vec.mul(x2_vec);
        x4_vec = _mm512_mul_ps(x2_vec, x2_vec);
        //x8_vec = x4_vec.mul(x4_vec);
        x8_vec = _mm512_mul_ps(x4_vec, x4_vec);
        //x16_vec = x8_vec.mul(x8_vec);
        x16_vec = _mm512_mul_ps(x8_vec, x8_vec);

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

        //y_vec = x_vec.fmuladd(a1, a0);
        y_vec = _mm512_fmadd_ps(x_vec, a1, a0);
        //t0 = x_vec.fmuladd(a3, a2);
        t0 =_mm512_fmadd_ps(x_vec, a3, a2);
        //y_vec.adda(x2_vec.mul(t0));
        y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x2_vec, t0));

        //t0 = x_vec.fmuladd(a7, a6);
        t0 = _mm512_fmadd_ps(x_vec, a7, a6);
        //t1 = x_vec.fmuladd(a5, a4);
        t1 = _mm512_fmadd_ps(x_vec, a5, a4);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm512_fmadd_ps(x2_vec, t0, t1);
        //y_vec.adda(x4_vec.mul(t2));
        y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x4_vec, t2));

        //t0 = x_vec.fmuladd(a15, a14);
        t0 = _mm512_fmadd_ps(x_vec, a15, a14);
        //t1 = x_vec.fmuladd(a13, a12);
        t1 = _mm512_fmadd_ps(x_vec, a13, a12);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm512_fmadd_ps(x2_vec, t0, t1);
        //t0 = x_vec.fmuladd(a11, a10);
        t0 = _mm512_fmadd_ps(x_vec, a11, a10);
        //t1 = x_vec.fmuladd(a9, a8);
        t1 = _mm512_fmadd_ps(x_vec, a9, a8);
        //t3 = x2_vec.fmuladd(t0, t1);
        t3 = _mm512_fmadd_ps(x2_vec, t0, t1);
        //t0 = x4_vec.fmuladd(t2, t3);
        t0 = _mm512_fmadd_ps(x4_vec, t2, t3);
        //y_vec.adda(x8_vec.mul(t0));
        y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x8_vec, t0));

        //y_vec.adda(x16_vec.mul(a[16]));
        y_vec = _mm512_add_ps(y_vec, _mm512_mul_ps(x16_vec, a16));

        //y_vec.store(&y[0]);
        _mm512_store_ps(&y[0], y_vec);
    }

    end = get_timestamp();

    UME::DynamicMemory::AlignedFree(y);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}

TIMING_RES test_avx512_64f()
{
    unsigned long long start, end; // Time measurements
    double a[17];
    double *x;
    double *y;

    x = (double *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(double), 64);
    y = (double *)UME::DynamicMemory::AlignedMalloc(8 * sizeof(double), 64);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1000.0)
        x[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
    }

    for (int i = 0; i < 17; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        a[i] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
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

    __m512d x_vec, x2_vec, x4_vec, x8_vec, x16_vec;
    __m512d y_vec;
    __m512d t0, t1, t2, t3;

    __m512d a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16;

    a0 = _mm512_set1_pd(a[0]);
    a1 = _mm512_set1_pd(a[1]);
    a2 = _mm512_set1_pd(a[2]);
    a3 = _mm512_set1_pd(a[3]);
    a4 = _mm512_set1_pd(a[4]);
    a5 = _mm512_set1_pd(a[5]);
    a6 = _mm512_set1_pd(a[6]);
    a7 = _mm512_set1_pd(a[7]);
    a8 = _mm512_set1_pd(a[8]);
    a9 = _mm512_set1_pd(a[9]);
    a10 = _mm512_set1_pd(a[10]);
    a11 = _mm512_set1_pd(a[11]);
    a12 = _mm512_set1_pd(a[12]);
    a13 = _mm512_set1_pd(a[13]);
    a14 = _mm512_set1_pd(a[14]);
    a15 = _mm512_set1_pd(a[15]);
    a16 = _mm512_set1_pd(a[16]);

    start = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        //x_vec.load(&x[i]);
        x_vec = _mm512_load_pd(&x[i]);

        //x2_vec = x_vec.mul(x_vec);
        x2_vec = _mm512_mul_pd(x_vec, x_vec);
        //x4_vec = x2_vec.mul(x2_vec);
        x4_vec = _mm512_mul_pd(x2_vec, x2_vec);
        //x8_vec = x4_vec.mul(x4_vec);
        x8_vec = _mm512_mul_pd(x4_vec, x4_vec);
        //x16_vec = x8_vec.mul(x8_vec);
        x16_vec = _mm512_mul_pd(x8_vec, x8_vec);

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

        //y_vec = x_vec.fmuladd(a1, a0);
        y_vec = _mm512_fmadd_pd(x_vec, a1, a0);
        //t0 = x_vec.fmuladd(a3, a2);
        t0 = _mm512_fmadd_pd(x_vec, a3, a2);
        //y_vec.adda(x2_vec.mul(t0));
        y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x2_vec, t0));

        //t0 = x_vec.fmuladd(a7, a6);
        t0 = _mm512_fmadd_pd(x_vec, a7, a6);
        //t1 = x_vec.fmuladd(a5, a4);
        t1 = _mm512_fmadd_pd(x_vec, a5, a4);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm512_fmadd_pd(x2_vec, t0, t1);
        //y_vec.adda(x4_vec.mul(t2));
        y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x4_vec, t2));

        //t0 = x_vec.fmuladd(a15, a14);
        t0 = _mm512_fmadd_pd(x_vec, a15, a14);
        //t1 = x_vec.fmuladd(a13, a12);
        t1 = _mm512_fmadd_pd(x_vec, a13, a12);
        //t2 = x2_vec.fmuladd(t0, t1);
        t2 = _mm512_fmadd_pd(x2_vec, t0, t1);
        //t0 = x_vec.fmuladd(a11, a10);
        t0 = _mm512_fmadd_pd(x_vec, a11, a10);
        //t1 = x_vec.fmuladd(a9, a8);
        t1 = _mm512_fmadd_pd(x_vec, a9, a8);
        //t3 = x2_vec.fmuladd(t0, t1);
        t3 = _mm512_fmadd_pd(x2_vec, t0, t1);
        //t0 = x4_vec.fmuladd(t2, t3);
        t0 = _mm512_fmadd_pd(x4_vec, t2, t3);
        //y_vec.adda(x8_vec.mul(t0));
        y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x8_vec, t0));

        //y_vec.adda(x16_vec.mul(a[16]));
        y_vec = _mm512_add_pd(y_vec, _mm512_mul_pd(x16_vec, a16));

        //y_vec.store(&y[0]);
        _mm512_store_pd(&y[0], y_vec);
    }

    end = get_timestamp();

    UME::DynamicMemory::AlignedFree(y);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}
#endif

template<typename FLOAT_VEC_TYPE>
TIMING_RES test_SIMD()
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_TYPE>::SCALAR_T FLOAT_T;

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
        x[i] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
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

    FLOAT_VEC_TYPE a0(a[0]),   a1(a[1]),   a2(a[2]),   a3(a[3]), 
                   a4(a[4]),   a5(a[5]),   a6(a[6]),   a7(a[7]), 
                   a8(a[8]),   a9(a[9]),   a10(a[10]), a11(11),
                   a12(a[12]), a13(a[13]), a14(a[14]), a15(a[15]);
    start = get_timestamp();

    for(int i = 0; i < ARRAY_SIZE; i+= FLOAT_VEC_TYPE::length()) {
        x_vec.load(&x[i]);
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

        y_vec = x_vec.fmuladd(a1, a0);
        t0 = x_vec.fmuladd(a3, a2);
        y_vec.adda(x2_vec.mul(t0));

        t0 = x_vec.fmuladd(a7, a6);
        t1 = x_vec.fmuladd(a5, a4);
        t2 = x2_vec.fmuladd(t0, t1);
        y_vec.adda(x4_vec.mul(t2));

        t0 = x_vec.fmuladd(a15, a14);
        t1 = x_vec.fmuladd(a13, a12);
        t2 = x2_vec.fmuladd(t0, t1);
        t0 = x_vec.fmuladd(a11, a10);
        t1 = x_vec.fmuladd(a9, a8);
        t3 = x2_vec.fmuladd(t0, t1);
        t0 = x4_vec.fmuladd(t2, t3);
        y_vec.adda(x8_vec.mul(t0));

        y_vec.adda(x16_vec.mul(a[16]));

        y_vec.store(&y[0]);
    }

    end = get_timestamp();

    UME::DynamicMemory::AlignedFree(y);
    UME::DynamicMemory::AlignedFree(x);

    return end - start;
}

template<typename FLOAT_T>
void benchmarkSIMD(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_SIMD<FLOAT_T>());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}
#if defined(__AVX2__) || defined(__AVX__)
void benchmark_avx_32f(char * resultPrefix,
                       int iterations,
                       TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_avx_32f());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

void benchmark_avx_64f(char * resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_avx_64f());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}
#endif

#if defined(__AVX512F__)
void benchmark_avx512_32f(char * resultPrefix,
                       int iterations,
                       TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_avx512_32f());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

void benchmark_avx512_64f(char * resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_avx512_64f());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}
#endif
int main()
{
    const int ITERATIONS = 100;

    std::cout << "The result is amount of time it takes to calculate polynomial of\n" 
                 "order 16 (no zero-coefficients) of: " << ARRAY_SIZE << " elements.\n" 
                 "All timing results in nanoseconds. \n"
                 "Speedup calculated with scalar floating point result as reference.\n\n"
                 "SIMD version uses following operations: \n"
                 " ZERO-CONSTR, SET-CONSTR, LOAD, STORE, MULV, FMULADDV, ADDVA\n";

    TimingStatistics stats_scalar_f, stats_scalar_d;

    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_f.update(test_scalar<float>());
    }

    std::cout << "Scalar code (float): " << (unsigned long long) stats_scalar_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
        << " (speedup: 1.0x)\n";

    for (int i = 0; i < ITERATIONS; i++) {
        stats_scalar_d.update(test_scalar<double>());
    }

    std::cout << "Scalar code (double): " << (unsigned long long) stats_scalar_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
        << " (speedup: " << stats_scalar_d.calculateSpeedup(stats_scalar_f) << "x)\n";

#if defined(__AVX__)
    benchmark_avx_32f("AVX/AVX2 intrinsics code (8x32f): ", ITERATIONS, stats_scalar_f);
    benchmark_avx_64f("AVX/AVX2 intrinsics code (4x64f): ", ITERATIONS, stats_scalar_f);
#else
    std::cout << "AVX/AVX2 intrinsics code (8x32f): AVX/AVX2 instruction set not detected\n";
    std::cout << "AVX/AVX2 intrinsics code (4x64f): AVX/AVX2 instruction set not detected\n";
#endif

#if defined(__AVX512F__)
    benchmark_avx512_32f("AVX512 intrinsics code (16x32f): ", ITERATIONS, stats_scalar_f);
    benchmark_avx512_64f("AVX512 intrinsics code (8x64f): ", ITERATIONS, stats_scalar_f);
#else
    std::cout << "AVX512 intrinsics code (16x32f): AVX/AVX2 instruction set not detected\n";
    std::cout << "AVX512 intrinsics code (8x64f): AVX/AVX2 instruction set not detected\n";
#endif

    benchmarkSIMD<UME::SIMD::SIMD1_32f>("SIMD code (1x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD2_32f>("SIMD code (2x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD4_32f>("SIMD code (4x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD8_32f>("SIMD code (8x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD16_32f>("SIMD code (16x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD32_32f>("SIMD code (32x32f): ", ITERATIONS, stats_scalar_f);

    benchmarkSIMD<UME::SIMD::SIMD1_64f>("SIMD code (1x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD2_64f>("SIMD code (2x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD4_64f>("SIMD code (4x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD8_64f>("SIMD code (8x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkSIMD<UME::SIMD::SIMD16_64f>("SIMD code (16x64f): ", ITERATIONS, stats_scalar_f);

    return 0;
}
