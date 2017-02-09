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

#ifndef UME_FIR_VERTICAL_INTEL_H_
#define UME_FIR_VERTICAL_INTEL_H_

#include <string>

#if defined(__AVX2__) || defined(__AVX512F__)
UME_NEVER_INLINE TIMING_RES test_intel_FIR8_float()
{
    const unsigned int FIR_ORDER = 8;

    unsigned long long start, end; // Time measurements
    alignas(32) float coeffs[FIR_ORDER];
    alignas(32) float state[FIR_ORDER];
    int state_begin = 0;
    float *x;
    float *y;

    x = (float *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 32);
    y = (float *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 32);

    //srand ((unsigned int)time(NULL));
    srand(0);
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    for(uint32_t i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        // Inverse the coefficients to simplify the initialization
        coeffs[FIR_ORDER - i - 1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        state[i] = static_cast<float>(0);
    }

    start = get_timestamp();

    alignas(64) uint32_t indices[8] = {0, 7, 6, 5, 4, 3, 2, 1};
    
    __m256 coeff_vec = _mm256_load_ps(coeffs);
    __m256i index_vec = _mm256_load_si256((const __m256i*)indices);
    __m256 state_vec, vec;
    
    __m256i permute_vec = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
    
#if defined ENABLE_DEBUG
        std::cout << "Coefficients: \n";
        for (uint32_t i = 0; i < FIR_ORDER; i++) {
            std::cout << coeffs[i] << " ";
        }
        std::cout <<"\nFIR evolution: \n";
#endif

    for(int i = 0; i < ARRAY_SIZE; i++) {
        // update state
        state[state_begin] = x[i];
        state_begin = (state_begin + 1) % FIR_ORDER;
        // calculate output
        // state_vec.gather(state, index_vec);
        state_vec = _mm256_load_ps(state);
        state_vec = _mm256_permutevar8x32_ps(state_vec, index_vec);
        // vec = state_vec.mul(coeff_vec);
        vec = _mm256_mul_ps(state_vec, coeff_vec);
        // y[i] = vec.hadd();
        alignas(32) float raw[8];
        _mm256_store_ps(raw, vec);
        y[i] = 0.0f;
        for(int j = 0; j < 8; j++) y[i] += raw[j];
        //index_vec = index_vec.template swizzle<15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>();
        index_vec = _mm256_permutevar8x32_epi32(permute_vec, index_vec);
        
#if defined ENABLE_DEBUG
        std::cout << "\nx: " << x[i];
        std::cout << "\nstate: \n";
        for (uint32_t j = 0; j < FIR_ORDER; j++) {
            std::cout << state[(state_begin + j) % FIR_ORDER] << " ";
        }
        std::cout << "\nstate vec: \n";
        for (uint32_t j = 0; j < FIR_ORDER; j++) {
            std::cout << state_vec[j] << " ";
        }
        std::cout << "\ny: " << y[i];
#endif
    }
    end = get_timestamp();

    // Perform reduction to avoid dead-code removals.
    volatile float red = static_cast<float>(0);
    for(int i = 0; i < ARRAY_SIZE; i++) {
        red += y[i];
    }
    // cast to void to avoid reduction
    (void)red;
    
    UME::DynamicMemory::AlignedFree(x);
    UME::DynamicMemory::AlignedFree(y);

    return end - start;
}
#endif

void benchmarkIntel_FIR8_float(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
#if defined(__AVX2__) || defined(__AVX512F__)
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_intel_FIR8_float());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
        
#else
    std::cout << resultPrefix << ": AVX2/AVX512 not detected, cannot run measurments.\n";
#endif
}

#if defined(__AVX512F__)
UME_NEVER_INLINE TIMING_RES test_intel_FIR16_float()
{
    const unsigned int FIR_ORDER = 16;

    unsigned long long start, end; // Time measurements
    alignas(64) float coeffs[FIR_ORDER];
    alignas(64) float state[FIR_ORDER];
    int state_begin = 0;
    float *x;
    float *y;

    x = (float *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 64);
    y = (float *) UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE*sizeof(float), 64);

    //srand ((unsigned int)time(NULL));
    srand(0);
    // Initialize arrays with random data
    for(int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1.0)
        x[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }

    for(uint32_t i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        // Inverse the coefficients to simplify the initialization
        coeffs[FIR_ORDER - i - 1] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        state[i] = static_cast<float>(0);
    }

    start = get_timestamp();

    alignas(64) uint32_t indices[16] = {0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    
    __m512 coeff_vec = _mm512_load_ps(coeffs);
    __m512i index_vec = _mm512_load_si512(indices);
    __m512 state_vec, vec;
    
    __m512i permute_vec = _mm512_setr_epi32(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    
#if defined ENABLE_DEBUG
        std::cout << "Coefficients: \n";
        for (uint32_t i = 0; i < FIR_ORDER; i++) {
            std::cout << coeffs[i] << " ";
        }
        std::cout <<"\nFIR evolution: \n";
#endif

    for(int i = 0; i < ARRAY_SIZE; i++) {
        // update state
        state[state_begin] = x[i];
        state_begin = (state_begin + 1) % FIR_ORDER;
        // calculate output
        // state_vec.gather(state, index_vec);
        state_vec = _mm512_load_ps(state);
        state_vec = _mm512_permutexvar_ps(index_vec, state_vec);
        // vec = state_vec.mul(coeff_vec);
        vec = _mm512_mul_ps(state_vec, coeff_vec);
        // y[i] = vec.hadd();
        y[i] = _mm512_reduce_add_ps(vec);
        //index_vec = index_vec.template swizzle<15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>();
        index_vec = _mm512_permutexvar_epi32(index_vec, permute_vec);
        
#if defined ENABLE_DEBUG
        std::cout << "\nx: " << x[i];
        std::cout << "\nstate: \n";
        for (uint32_t j = 0; j < FIR_ORDER; j++) {
            std::cout << state[(state_begin + j) % FIR_ORDER] << " ";
        }
        std::cout << "\nstate vec: \n";
        for (uint32_t j = 0; j < FIR_ORDER; j++) {
            std::cout << state_vec[j] << " ";
        }
        std::cout << "\ny: " << y[i];
#endif
    }
    end = get_timestamp();

    // Perform reduction to avoid dead-code removals.
    volatile float red = static_cast<float>(0);
    for(int i = 0; i < ARRAY_SIZE; i++) {
        red += y[i];
    }
    // cast to void to avoid reduction
    (void)red;
    
    UME::DynamicMemory::AlignedFree(x);
    UME::DynamicMemory::AlignedFree(y);

    return end - start;
}
#endif

void benchmarkIntel_FIR16_float(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
#if defined(__AVX512F__)
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_intel_FIR16_float());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
        
#else
    std::cout << resultPrefix << ": AVX512 not detected, cannot run measurments.\n";
#endif
}

#endif
