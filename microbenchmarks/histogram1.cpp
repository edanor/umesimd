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

const int INPUT_SIZE = 600000; // Number of data samples
const int HIST_SIZE = 100;     // Number of histogram bins
//alignas(32) float x[ARRAY_SIZE];

template<typename FLOAT_T>
TIMING_RES test_scalar()
{
    unsigned long long start, end;    // Time measurements
    
    FLOAT_T *data;
    unsigned int *hist;

    data = (FLOAT_T *) UME::DynamicMemory::AlignedMalloc(INPUT_SIZE*sizeof(FLOAT_T), sizeof(FLOAT_T));

    // Initialize arrays with random data
    for(int i = 0; i < INPUT_SIZE; i++) {
        // Generate random numbers in range (0.0;1000.0)
        data[i] = static_cast <FLOAT_T> (rand()) / (static_cast <FLOAT_T> (RAND_MAX)/static_cast<FLOAT_T>(999));
    }
    
    hist = (unsigned int *) UME::DynamicMemory::AlignedMalloc(HIST_SIZE*sizeof(unsigned int), sizeof(unsigned int));

    for(unsigned int i = 0; i < HIST_SIZE; i++) {
        hist[i] = 0;
    }

    // This is the actual binning code
    {    
        unsigned int bin;
        start = __rdtsc();
      
        for(int i = 0; i < INPUT_SIZE; i++)
        {
            bin = (unsigned int) ((FLOAT_T(HIST_SIZE)/static_cast<FLOAT_T>(1000))*data[i]);
            hist[bin]++;
        }
        
        end = __rdtsc();
    }
    
    // Use for debugging
    //for (unsigned int i = 0; i < HIST_SIZE; i++) {
    //    std::cout << i << ": " << hist[i] << "\n";
    //}


    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}

template<typename FLOAT_VEC_T, typename UINT_VEC_T>
inline void test_UME_SIMD_float_recursive_helper(UINT_VEC_T const & index_vec, unsigned int * hist)
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::HALF_LEN_VEC_T HALF_LEN_VEC_T;
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::INT_VEC_T      INT_VEC_T;

    typedef typename UME::SIMD::SIMDTraits<HALF_LEN_VEC_T>::UINT_VEC_T HALF_LEN_UINT_VEC_T;

    // If there are no repeating indices in vector, we can increment histogram all at once.
    // Otherwise if there are colissions, we split the vector in halves and try incrementing one 
    // half at a time.
    if (index_vec.unique()) {
        UINT_VEC_T bin_vec;
        bin_vec.gather(hist, index_vec);
        bin_vec.prefinc();
        bin_vec.scatter(hist, index_vec);
    }
    else {
        //std::cout << "unpacking: " << index_vec[0] << " " << index_vec[1] << std::endl;
        HALF_LEN_UINT_VEC_T vec_l, vec_h;
        index_vec.unpack(vec_l, vec_h);
        test_UME_SIMD_float_recursive_helper<HALF_LEN_VEC_T, HALF_LEN_UINT_VEC_T>(vec_l, hist);
        test_UME_SIMD_float_recursive_helper<HALF_LEN_VEC_T, HALF_LEN_UINT_VEC_T>(vec_h, hist);
    }
}

// Specialization for SIMD1_32f. This covers boundary conditions
template<>
inline void test_UME_SIMD_float_recursive_helper<UME::SIMD::SIMD1_32f, UME::SIMD::SIMD1_32u>(UME::SIMD::SIMD1_32u const & index_vec, unsigned int * hist)
{
    unsigned int bin = index_vec[0];
    hist[bin]++;
}

template<typename FLOAT_VEC_T>
TIMING_RES test_UME_SIMD()
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::SCALAR_T   FLOAT_T;
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::INT_VEC_T  INT_VEC_T;
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::UINT_VEC_T UINT_VEC_T;

    const uint32_t VEC_LEN = FLOAT_VEC_T::length();
    const int ALIGNMENT = FLOAT_VEC_T::alignment();
    unsigned long long start, end;    // Time measurements

    FLOAT_T *data;
    unsigned int *hist;

    unsigned int *verify_hist;

    data = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(INPUT_SIZE*sizeof(FLOAT_T), ALIGNMENT);

    // Initialize arrays with random data
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Generate random numbers in range (0.0;1000.0)
        data[i] = static_cast <FLOAT_T> (rand()) / (static_cast <FLOAT_T> (RAND_MAX) / static_cast<FLOAT_T>(999));
    }

    hist = (unsigned int *)UME::DynamicMemory::AlignedMalloc(HIST_SIZE*sizeof(unsigned int), ALIGNMENT);
    verify_hist = (unsigned int *)UME::DynamicMemory::AlignedMalloc(HIST_SIZE*sizeof(unsigned int), ALIGNMENT);

    for (int i = 0; i < HIST_SIZE; i++) {
        hist[i] = 0;
        verify_hist[i] = 0;
    }

    // This is the actual binning code
    {
        // Calculate loop-peeling division
        uint32_t PEEL_COUNT = INPUT_SIZE / VEC_LEN;
        uint32_t REM_COUNT = INPUT_SIZE - PEEL_COUNT*VEC_LEN;

        FLOAT_VEC_T data_vec;
        FLOAT_VEC_T t0;
        FLOAT_VEC_T coeff_vec(float(HIST_SIZE) / static_cast<float>(1000));

        INT_VEC_T t1;
        UINT_VEC_T index_vec;

        unsigned int bin;

        start = __rdtsc();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec.loada(&data[i*VEC_LEN]);
            t0 = data_vec.mul(coeff_vec);
            t1 = t0.trunc();
            index_vec.assign(UINT_VEC_T(t1));
            // Perform histogram update
            test_UME_SIMD_float_recursive_helper<FLOAT_VEC_T, UINT_VEC_T>(index_vec, hist);
        }
        
        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((FLOAT_T(HIST_SIZE) / static_cast<FLOAT_T>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = __rdtsc();
        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((FLOAT_T(HIST_SIZE) / static_cast<FLOAT_T>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                std::cout << "Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}

int main()
{
    TIMING_RES t_scalar_f, 
               t_UME_SIMD1_32f,
               t_UME_SIMD2_32f,
               t_UME_SIMD4_32f,
               t_UME_SIMD8_32f,
               t_UME_SIMD16_32f,
               t_UME_SIMD32_32f;

    float t_scalar_f_avg = 0.0f,
          t_UME_SIMD1_32f_avg = 0.0f,
          t_UME_SIMD2_32f_avg = 0.0f,
          t_UME_SIMD4_32f_avg = 0.0f,
          t_UME_SIMD8_32f_avg = 0.0f,
          t_UME_SIMD16_32f_avg = 0.0f,
          t_UME_SIMD32_32f_avg = 0.0f;

    srand ((unsigned int)time(NULL));
    for (int i = 0; i < 20; i++)
    {
        t_scalar_f = test_scalar<float>();
        t_scalar_f_avg = 1.0f / (1.0f + float(i)) * (float(t_scalar_f) - t_scalar_f_avg);

        t_UME_SIMD1_32f = test_UME_SIMD<UME::SIMD::SIMD1_32f>();
        t_UME_SIMD1_32f_avg = 1.0f / (1.0f + float(i)) * (float(t_UME_SIMD1_32f) - t_UME_SIMD1_32f_avg);

        t_UME_SIMD2_32f = test_UME_SIMD<UME::SIMD::SIMD2_32f>();
        t_UME_SIMD2_32f_avg = 1.0f / (1.0f + float(i)) * (float(t_UME_SIMD2_32f) - t_UME_SIMD2_32f_avg);

        t_UME_SIMD4_32f = test_UME_SIMD<UME::SIMD::SIMD4_32f>();
        t_UME_SIMD4_32f_avg = 1.0f / (1.0f + float(i)) * (float(t_UME_SIMD4_32f) - t_UME_SIMD4_32f_avg);

        t_UME_SIMD8_32f = test_UME_SIMD<UME::SIMD::SIMD8_32f>();
        t_UME_SIMD8_32f_avg = 1.0f / (1.0f + float(i)) * (float(t_UME_SIMD8_32f) - t_UME_SIMD8_32f_avg);

        t_UME_SIMD16_32f = test_UME_SIMD<UME::SIMD::SIMD16_32f>();
        t_UME_SIMD16_32f_avg = 1.0f / (1.0f + float(i)) * (float(t_UME_SIMD16_32f) - t_UME_SIMD16_32f_avg);

        t_UME_SIMD32_32f = test_UME_SIMD<UME::SIMD::SIMD32_32f>();
        t_UME_SIMD32_32f_avg = 1.0f / (1.0f + float(i)) * (float(t_UME_SIMD32_32f) - t_UME_SIMD32_32f_avg);
    }

    std::cout << "The result is amount of time it takes to calculate histogram of: " << INPUT_SIZE << " elements with " << HIST_SIZE << "-bin histogram.\n"
        "All timing results in clock cycles. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD versions use following operations: \n"
        "float 32b: LOADA, MULV, TRUNC\n"
        "int   32b:  ITOU\n"
        "uint  32b:  ASSIGNV, UNIQUE, GATHER, SCATTER, PREFINC, UNPACK\n\n";

    std::cout << "Scalar code (float): " << (long)t_scalar_f_avg
        << " (speedup: 1.0x)"
        << std::endl;

    std::cout << "SIMD code (1x32f): " << (long)t_UME_SIMD1_32f_avg
        << " (speedup: "
        << float(t_scalar_f_avg) / float(t_UME_SIMD1_32f_avg) << ")"
        << std::endl;

    std::cout << "SIMD code (2x32f): " << (long)t_UME_SIMD2_32f_avg
        << " (speedup: "
        << float(t_scalar_f_avg) / float(t_UME_SIMD2_32f_avg) << ")"
        << std::endl;

    std::cout << "SIMD code (4x32f): " << (long)t_UME_SIMD4_32f_avg
        << " (speedup: "
        << float(t_scalar_f_avg) / float(t_UME_SIMD4_32f_avg) << ")"
        << std::endl;

    std::cout << "SIMD code (8x32f): " << (long)t_UME_SIMD8_32f_avg
        << " (speedup: "
        << float(t_scalar_f_avg) / float(t_UME_SIMD8_32f_avg) << ")"
        << std::endl;

    std::cout << "SIMD code (16x32f): " << (long)t_UME_SIMD16_32f_avg
        << " (speedup: "
        << float(t_scalar_f_avg) / float(t_UME_SIMD16_32f_avg) << ")"
        << std::endl;

    std::cout << "SIMD code (32x32f): " << (long)t_UME_SIMD32_32f_avg
        << " (speedup: "
        << float(t_scalar_f_avg) / float(t_UME_SIMD32_32f_avg) << ")"
        << std::endl;

    return 0;
}