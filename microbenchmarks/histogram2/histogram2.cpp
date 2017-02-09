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

#include <iostream>
#include <memory>

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h> 
#endif

#include <cmath>
#include <time.h>
#include <stdlib.h>
#include <string>

//#define UME_SIMD_SHOW_EMULATION_WARNINGS 1
#include "../../UMESimd.h"
#include "../utilities/TimingStatistics.h"

const int INPUT_SIZE = 1000000; // Number of data samples
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
        start = get_timestamp();
      
        for(int i = 0; i < INPUT_SIZE; i++)
        {
            bin = (unsigned int) ((FLOAT_T(HIST_SIZE)/static_cast<FLOAT_T>(1000))*data[i]);
            hist[bin]++;
        }
        
        end = get_timestamp();
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}

template<typename FLOAT_VEC_T, typename UINT_VEC_T>
UME_FORCE_INLINE void test_UME_SIMD_float_recursive_helper(UINT_VEC_T const & index_vec, unsigned int * hist)
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::HALF_LEN_VEC_T HALF_LEN_VEC_T;

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
        HALF_LEN_UINT_VEC_T vec_l, vec_h;
        index_vec.unpack(vec_l, vec_h);

        test_UME_SIMD_float_recursive_helper<HALF_LEN_VEC_T, HALF_LEN_UINT_VEC_T>(vec_l, hist);
        test_UME_SIMD_float_recursive_helper<HALF_LEN_VEC_T, HALF_LEN_UINT_VEC_T>(vec_h, hist);
    }
}

// Specialization for SIMD1_32f. This covers boundary conditions
template<>
UME_FORCE_INLINE void test_UME_SIMD_float_recursive_helper<UME::SIMD::SIMD1_32f, UME::SIMD::SIMD1_32u>(UME::SIMD::SIMD1_32u const & index_vec, unsigned int * hist)
{
    unsigned int bin = index_vec[0];
    hist[bin]++;
}

#if defined(__AVX2__) || defined(__AVX512F__)
TIMING_RES test_AVX_f_256()
{
    const uint32_t VEC_LEN = 8;
    const int ALIGNMENT = 32;
    unsigned long long start, end;    // Time measurements

    float *data;
    unsigned int *hist;

    unsigned int *verify_hist;

    data = (float*)UME::DynamicMemory::AlignedMalloc(INPUT_SIZE*sizeof(float), ALIGNMENT);

    // Initialize arrays with random data
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Generate random numbers in range (0.0;1000.0)
        data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / static_cast<float>(999));
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

        __m256 data_vec;
        __m256 t0;
        __m256 coeff_vec = _mm256_set1_ps(float(HIST_SIZE) / static_cast<float>(1000));

        __m256i index_vec;

        unsigned int bin;

        start = get_timestamp();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec = _mm256_load_ps(&data[i*VEC_LEN]);
            t0 = _mm256_mul_ps(data_vec, coeff_vec);
            index_vec = _mm256_cvttps_epi32(t0);
            // Perform histogram update

            // AVX2 does not offer 'conflict detection' instructions.
            // Need to emulate using scalar code.
            alignas(16) int32_t temp_index[8];
            _mm256_store_si256((__m256i*)temp_index, index_vec);

            for (unsigned int i = 0; i < VEC_LEN; i++) {
                hist[temp_index[i]]++;
            }
        }

        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((float(HIST_SIZE) / static_cast<float>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = get_timestamp();

        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((float(HIST_SIZE) / static_cast<float>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                std::cout << VEC_LEN << ": Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}

TIMING_RES test_AVX_d_256()
{
    const uint32_t VEC_LEN = 4;
    const int ALIGNMENT = 32;
    unsigned long long start, end;    // Time measurements

    double *data;
    unsigned int *hist;

    unsigned int *verify_hist;

    data = (double*)UME::DynamicMemory::AlignedMalloc(INPUT_SIZE*sizeof(double), ALIGNMENT);

    // Initialize arrays with random data
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Generate random numbers in range (0.0;1000.0)
        data[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) / static_cast<double>(999));
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

        __m256d data_vec;
        __m256d t0;
        __m256d coeff_vec = _mm256_set1_pd(double(HIST_SIZE) / static_cast<double>(1000));

        __m128i index_vec;

        unsigned int bin;

        start = get_timestamp();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec = _mm256_load_pd(&data[i*VEC_LEN]);
            t0 = _mm256_mul_pd(data_vec, coeff_vec);
            index_vec = _mm256_cvttpd_epi32(t0);
            // Perform histogram update

            // AVX2 does not offer 'conflict detection' instructions.
            // Need to emulate using scalar code.
            alignas(16) int32_t temp_index[4];
            _mm_store_si128((__m128i*)temp_index, index_vec);

            for (unsigned int i = 0; i < VEC_LEN; i++) {
                hist[temp_index[i]]++;
            }
        }

        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((double(HIST_SIZE) / static_cast<double>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = get_timestamp();

        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((double(HIST_SIZE) / static_cast<double>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                std::cout << VEC_LEN << ": Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}
#endif

#if defined(__AVX512F__)
TIMING_RES test_AVX512_f()
{
    const uint32_t VEC_LEN = 16;
    const int ALIGNMENT = 64;
    unsigned long long start, end;    // Time measurements

    float *data;
    unsigned int *hist;

    unsigned int *verify_hist;

    data = (float*)UME::DynamicMemory::AlignedMalloc(INPUT_SIZE*sizeof(float), ALIGNMENT);

    // Initialize arrays with random data
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Generate random numbers in range (0.0;1000.0)
        data[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX) / static_cast<float>(999));
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

        __m512 data_vec;
        __m512 t0;
        __m512 coeff_vec = _mm512_set1_ps(float(HIST_SIZE) / static_cast<float>(1000));

        __m512i index_vec;
        alignas(ALIGNMENT) uint32_t indices[VEC_LEN];

        unsigned int bin;

        start = get_timestamp();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec = _mm512_load_ps(&data[i*VEC_LEN]);
            t0 = _mm512_mul_ps(data_vec, coeff_vec);
            index_vec = _mm512_cvttps_epi32(t0);
            // Perform histogram update

            // check for conflicts
            __m512i t0 = _mm512_conflict_epi32(index_vec);
            __mmask16 m0 = 0xFFFF & _mm512_cmpeq_epu32_mask(t0, _mm512_setzero_epi32());

            // Perform histogram update. Use scalar emulation
            if(m0 == 0xFFFF)
            {
                __m512i bin_vec = _mm512_i32gather_epi32(index_vec, (const int *)hist, 4);
                bin_vec = _mm512_add_epi32(bin_vec, _mm512_set1_epi32(1));
                _mm512_i32scatter_epi32((int *)hist, index_vec, bin_vec, 4);
            }
            else
            {
                _mm512_store_epi32(indices, index_vec);
                for (unsigned int i = 0; i < VEC_LEN; i++) {
                    hist[indices[i]]++;
                }
            }
        }

        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((float(HIST_SIZE) / static_cast<float>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = get_timestamp();

        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((float(HIST_SIZE) / static_cast<float>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                std::cout << VEC_LEN << ": Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}

TIMING_RES test_AVX512_d()
{
    const uint32_t VEC_LEN = 8;
    const int ALIGNMENT = 64;
    unsigned long long start, end;    // Time measurements

    double *data;
    unsigned int *hist;

    unsigned int *verify_hist;

    data = (double*)UME::DynamicMemory::AlignedMalloc(INPUT_SIZE*sizeof(double), ALIGNMENT);

    // Initialize arrays with random data
    for (int i = 0; i < INPUT_SIZE; i++) {
        // Generate random numbers in range (0.0;1000.0)
        data[i] = static_cast <double> (rand()) / (static_cast <double> (RAND_MAX) / static_cast<double>(999));
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

        __m512d data_vec;
        __m512d t0;
        __m512d coeff_vec = _mm512_set1_pd(double(HIST_SIZE) / static_cast<double>(1000));

        __m256i index_vec;
        __m256i bin_vec;
        alignas(ALIGNMENT) uint32_t indices[VEC_LEN];

        unsigned int bin;

        start = get_timestamp();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec = _mm512_load_pd(&data[i*VEC_LEN]);
            t0 = _mm512_mul_pd(data_vec, coeff_vec);
            index_vec = _mm512_cvttpd_epi32(t0);

            // check for conflicts
            __m512i t0 = _mm512_conflict_epi32(_mm512_castsi256_si512(index_vec));
            __mmask16 m0 = 0xFF & _mm512_cmpeq_epu32_mask(t0, _mm512_setzero_epi32());

            // Perform histogram update. Use scalar emulation
            if(m0 == 0xFF)
            {
                bin_vec = _mm256_i32gather_epi32((const int *)hist, index_vec, 4);
                bin_vec = _mm256_add_epi32(bin_vec, _mm256_set1_epi32(1));
                //_mm512_i32scatter_epi32(hist, index_vec, bin_vec, 4);
                _mm512_mask_i32scatter_epi32(
                    (int*)hist,
                    0xFF,
                    _mm512_castsi256_si512(index_vec),
                    _mm512_castsi256_si512(bin_vec),
                    4);
            }
            else
            {
                _mm256_store_si256((__m256i*)indices, index_vec);
                for (unsigned int i = 0; i < VEC_LEN; i++) {
                    hist[indices[i]]++;
                }
            }
        }

        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((double(HIST_SIZE) / static_cast<double>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = get_timestamp();

        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((double(HIST_SIZE) / static_cast<double>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                std::cout << VEC_LEN << ": Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}
#endif

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
        UINT_VEC_T bin_vec;
        alignas(FLOAT_VEC_T::alignment()) uint32_t indices[VEC_LEN];

        unsigned int bin;

        start = get_timestamp();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec.loada(&data[i*VEC_LEN]);
            t0 = data_vec.mul(coeff_vec);
            t1 = t0.trunc();
            index_vec.assign(UINT_VEC_T(t1));
            // Perform histogram update
            // test_UME_SIMD_float_recursive_helper<FLOAT_VEC_T, UINT_VEC_T>(index_vec, hist);
            if (index_vec.unique()) {
                bin_vec.gather(hist, index_vec);
                bin_vec.prefinc();
                bin_vec.scatter(hist, index_vec);
            }
            else {
                index_vec.storea(indices);
                for (unsigned int i = 0; i < VEC_LEN; i++) {
                    hist[indices[i]]++;
                }
            }
        }
        
        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((FLOAT_T(HIST_SIZE) / static_cast<FLOAT_T>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = get_timestamp();

        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((FLOAT_T(HIST_SIZE) / static_cast<FLOAT_T>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                std::cout << VEC_LEN << ": Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
}

/* 
    This testing code should be unified with single precision
template<typename DOUBLE_VEC_T>
TIMING_RES test_UME_SIMD_double()
{
    typedef typename UME::SIMD::SIMDTraits<DOUBLE_VEC_T>::SCALAR_T   FLOAT_T;
    typedef typename UME::SIMD::SIMDTraits<DOUBLE_VEC_T>::INT_VEC_T  INT_VEC_T;
    typedef typename UME::SIMD::SIMDTraits<DOUBLE_VEC_T>::UINT_VEC_T UINT_VEC_T;

    const uint32_t VEC_LEN = DOUBLE_VEC_T::length();
    const int ALIGNMENT = DOUBLE_VEC_T::alignment();
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

        DOUBLE_VEC_T data_vec;
        DOUBLE_VEC_T t0;
        DOUBLE_VEC_T coeff_vec(float(HIST_SIZE) / static_cast<float>(1000));

        INT_VEC_T t1;
        UME::SIMD::SIMDVec_u<uint32_t, VEC_LEN> index_vec;
        UME::SIMD::SIMDVec_u<uint32_t, VEC_LEN> bin_vec;
        alignas(UME::SIMD::SIMDVec_u<uint32_t, VEC_LEN>::alignment()) uint32_t indices[VEC_LEN];

        unsigned int bin;

        start = get_timestamp();

        for (uint32_t i = 0; i < PEEL_COUNT; i++) {
            // Calculate indices
            data_vec.loada(&data[i*VEC_LEN]);
            t0 = data_vec.mul(coeff_vec);
            t1 = t0.trunc();
            index_vec.assign(UME::SIMD::SIMDVec_u<uint32_t, VEC_LEN>(UINT_VEC_T(t1)));
            // Perform histogram update
            // test_UME_SIMD_float_recursive_helper<FLOAT_VEC_T, UINT_VEC_T>(index_vec, hist);
            if (index_vec.unique()) {
                bin_vec.gather(hist, index_vec);
                bin_vec.prefinc();
                bin_vec.scatter(hist, index_vec);
            }
            else {
                index_vec.storea(indices);
                for (unsigned int i = 0; i < VEC_LEN; i++) {
                    hist[indices[i]]++;
                }
            }
        }

        // Calculate reminder elements using scalar code
        for (uint32_t i = 0; i < REM_COUNT; i++) {
            bin = (unsigned int)((FLOAT_T(HIST_SIZE) / static_cast<FLOAT_T>(1000)) * data[PEEL_COUNT*VEC_LEN + i]);
            hist[bin]++;
        }

        end = get_timestamp();

        // Verify results
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            unsigned int bin = (unsigned int)((FLOAT_T(HIST_SIZE) / static_cast<FLOAT_T>(1000))*data[i]);
            verify_hist[bin]++;
        }

        for (int i = 0; i < HIST_SIZE; i++) {
            if (hist[i] != verify_hist[i]) {
                //std::cout << VEC_LEN << ": Invalid result at index " << i << " expected: " << verify_hist[i] << ", actual: " << hist[i] << "\n";
            }
        }
    }

    UME::DynamicMemory::AlignedFree(data);
    UME::DynamicMemory::AlignedFree(hist);

    return end - start;
} */

#if defined(__AVX2__) || defined(__AVX512F__)
void benchmarkAVX256_f(int iterations, TimingStatistics & reference)
{
    TimingStatistics stats;
    for (int i = 0; i < iterations; i++)
    {
        unsigned long long elapsed = test_AVX_f_256();
        stats.update(elapsed);
    }

    std::cout << "AVX2 (float): " << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}
#endif

#if defined(__AVX512F__)
void benchmarkAVX512_f(int iterations, TimingStatistics & reference)
{
    TimingStatistics stats;
    for (int i = 0; i < iterations; i++)
    {
        unsigned long long elapsed = test_AVX512_f();
        stats.update(elapsed);
    }

    std::cout << "AVX512 (float): " << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}
#endif

#if defined(__AVX2__) || defined(__AVX512F__)
void benchmarkAVX256_d(int iterations, TimingStatistics & reference)
{
    TimingStatistics stats;
    for (int i = 0; i < iterations; i++)
    {
        unsigned long long elapsed = test_AVX_d_256();
        stats.update(elapsed);
    }

    std::cout << "AVX2 (double): " << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}
#endif

#if defined(__AVX512F__)
void benchmarkAVX512_d(int iterations, TimingStatistics & reference)
{
    TimingStatistics stats;
    for (int i = 0; i < iterations; i++)
    {
        unsigned long long elapsed = test_AVX512_d();
        stats.update(elapsed);
    }

    std::cout << "AVX512 (double): " << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats.confidence95()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}
#endif

template<typename VEC_T>
void benchmarkUMESIMD(std::string const & resultPrefix, int iterations, TimingStatistics & reference)
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

template<typename VEC_T>
void benchmarkUMESIMD_double(std::string const & resultPrefix, int iterations, TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        //unsigned long long elapsed = test_UME_SIMD_double<VEC_T>();
        //stats.update(elapsed);
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
    const int ITERATIONS = 100;

    TimingStatistics stats_scalar_f, stats_scalar_d;

    srand ((unsigned int)time(NULL));

    std::cout << "The result is amount of time it takes to calculate histogram of: " << INPUT_SIZE << " elements with " << HIST_SIZE << "-bin histogram.\n"
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD versions use following operations: \n"
        "float 32b: LOADA, MULV, TRUNC\n"
        "int   32b:  ITOU\n"
        "uint  32b:  ASSIGNV, UNIQUE, GATHERV, SCATTERV, PREFINC, UNPACK\n\n";

    for (int i = 0; i < ITERATIONS; i++)
    {
        stats_scalar_f.update(test_scalar<float>());
    }

    std::cout << "Scalar code (float): " << (unsigned long long) stats_scalar_f.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_f.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats_scalar_f.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats_scalar_f.confidence95()
        << " (speedup: 1.0x)"
        << std::endl;

#if defined(__AVX2__) || defined(__AVX512F__)
    benchmarkAVX256_f(ITERATIONS, stats_scalar_f);
#else
    std::cout << "AVX2 (float): disabled, cannot run measurements.\n";
#endif
#if defined(__AVX512F__)
    benchmarkAVX512_f(ITERATIONS, stats_scalar_f);
#else
    std::cout << "AVX512 (float): disabled, cannot run measurements.\n";
#endif

    benchmarkUMESIMD<UME::SIMD::SIMD1_32f>("SIMD code (1x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_32f>("SIMD code (2x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_32f>("SIMD code (4x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_32f>("SIMD code (8x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_32f>("SIMD code (16x32f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD<UME::SIMD::SIMD32_32f>("SIMD code (32x32f): ", ITERATIONS, stats_scalar_f);

    for (int i = 0; i < ITERATIONS; i++)
    {
        stats_scalar_d.update(test_scalar<double>());
    }

    std::cout << "Scalar code (double): " << (unsigned long long) stats_scalar_d.getAverage()
        << ", dev: " << (unsigned long long) stats_scalar_d.getStdDev()
        << ", 90% confidence: " << (unsigned long long) stats_scalar_d.confidence90()
        << ", 95% confidence: " << (unsigned long long) stats_scalar_d.confidence95()
        << " (speedup: " << stats_scalar_d.getAverage()/stats_scalar_f.getAverage() << ")"
        << std::endl;

#if defined(__AVX2__) || defined(__AVX512F__)
    benchmarkAVX256_d(ITERATIONS, stats_scalar_f);
#else
    std::cout << "AVX2 (double): disabled, cannot run measurements.\n";
#endif
#if defined(__AVX512F__)
    benchmarkAVX512_d(ITERATIONS, stats_scalar_f);
#else
    std::cout << "AVX512 (double): disabled, cannot run measurements.\n";
#endif

    benchmarkUMESIMD_double<UME::SIMD::SIMD1_64f>("SIMD code (1x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD_double<UME::SIMD::SIMD2_64f>("SIMD code (2x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD_double<UME::SIMD::SIMD4_64f>("SIMD code (4x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD_double<UME::SIMD::SIMD8_64f>("SIMD code (8x64f): ", ITERATIONS, stats_scalar_f);
    benchmarkUMESIMD_double<UME::SIMD::SIMD16_64f>("SIMD code (16x64f): ", ITERATIONS, stats_scalar_f);

    return 0;
}
