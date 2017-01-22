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

#ifndef UME_FIR_VERTICAL_UMESIMD_H_
#define UME_FIR_VERTICAL_UMESIMD_H_

using namespace UME::SIMD;

template<typename SCALAR_UINT_T, uint32_t FIR_ORDER>
class VerticalFIRHelper {
    const SCALAR_UINT_T indices[FIR_ORDER];
    const SCALAR_UINT_T permutation[4];
    
    VerticalFIRHelper() {
        indices[0] = 0;
        permutation[0] = FIR_ORDER - 1;
        for(uint32_t i = 1; i < FIR_ORDER; i++) {
            indices[i] = FIR_ORDER-i;
            permutation[i] = i-1;
        }
    }

};

template<typename SCALAR_UINT_T>
class VerticalFIRHelper<SCALAR_UINT_T, 4> {
public:
    const SCALAR_UINT_T indices[4];
    const SCALAR_UINT_T permutation[4];
    
    VerticalFIRHelper() : indices{0, 3, 2, 1}, permutation{3, 0, 1, 2} {
    }
};

template<typename SCALAR_UINT_T>
class VerticalFIRHelper<SCALAR_UINT_T, 8> {
public:
    const SCALAR_UINT_T indices[8];
    const SCALAR_UINT_T permutation[8];
    
    VerticalFIRHelper() : indices{0, 7, 6, 5, 4, 3, 2, 1}, permutation{7, 0, 1, 2, 3, 4, 5, 6} {
    }
};

template<typename SCALAR_UINT_T>
class VerticalFIRHelper<SCALAR_UINT_T, 16> {
public:
    const SCALAR_UINT_T indices[16];
    const SCALAR_UINT_T permutation[16];

    VerticalFIRHelper() : indices{0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}, permutation{15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14} {
    }
};

template<typename FLOAT_T, uint32_t FIR_ORDER>
UME_NEVER_INLINE TIMING_RES test_ume_FIR()
{
    typedef typename SIMDTraits<SIMDVec<FLOAT_T, FIR_ORDER>>::SCALAR_UINT_T SCALAR_UINT_T;
    const int ALIGNMENT = SIMDVec<FLOAT_T, FIR_ORDER>::alignment();

    unsigned long long start, end; // Time measurements
    alignas(ALIGNMENT) FLOAT_T coeffs[FIR_ORDER];
    alignas(ALIGNMENT) FLOAT_T state[FIR_ORDER];
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

    for(uint32_t i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        // Inverse the coefficients to simplify the initialization
        coeffs[FIR_ORDER - i - 1] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        state[i] = static_cast<FLOAT_T>(0);
    }

    start = get_timestamp();

    VerticalFIRHelper<SCALAR_UINT_T, FIR_ORDER> helper;
    SIMDVec<FLOAT_T, FIR_ORDER> coeff_vec(coeffs);
    SIMDVec<SCALAR_UINT_T, FIR_ORDER> index_vec(helper.indices);
    SIMDVec<FLOAT_T, FIR_ORDER> state_vec, vec;
    SIMDSwizzle<FIR_ORDER> perm(helper.permutation);

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
        state_vec.gather(state, index_vec);
        vec = state_vec.mul(coeff_vec);
        y[i] = vec.hadd();
        index_vec.swizzlea(perm);

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

template<typename FLOAT_T, uint32_t FIR_ORDER>
void benchmarkSIMD(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_ume_FIR<FLOAT_T, FIR_ORDER>());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

template<typename FLOAT_T>
UME_NEVER_INLINE TIMING_RES test_ume_FIR4()
{
    typedef typename SIMDTraits<SIMDVec<FLOAT_T, 4>>::SCALAR_UINT_T SCALAR_UINT_T;
    const unsigned int FIR_ORDER = 4;

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

    for(uint32_t i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        // Inverse the coefficients to simplify the initialization
        coeffs[FIR_ORDER - i - 1] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        state[i] = static_cast<FLOAT_T>(0);
    }

    start = get_timestamp();

    VerticalFIRHelper<SCALAR_UINT_T, FIR_ORDER> helper;
    SIMDVec<FLOAT_T, FIR_ORDER> coeff_vec(coeffs);
    SIMDVec<SCALAR_UINT_T, FIR_ORDER> index_vec(helper.indices);
    SIMDVec<FLOAT_T, FIR_ORDER> state_vec, vec;

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
        state_vec.gather(state, index_vec);
        vec = state_vec.mul(coeff_vec);
        y[i] = vec.hadd();
        index_vec = index_vec.template swizzle<3, 0, 1, 2>();

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


template<typename FLOAT_T>
UME_NEVER_INLINE TIMING_RES test_ume_FIR8()
{
    typedef typename SIMDTraits<SIMDVec<FLOAT_T, 8>>::SCALAR_UINT_T SCALAR_UINT_T;
    const unsigned int FIR_ORDER = 8;

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

    for(uint32_t i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        // Inverse the coefficients to simplify the initialization
        coeffs[FIR_ORDER - i - 1] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        state[i] = static_cast<FLOAT_T>(0);
    }

    start = get_timestamp();

    VerticalFIRHelper<SCALAR_UINT_T, FIR_ORDER> helper;
    SIMDVec<FLOAT_T, FIR_ORDER> coeff_vec(coeffs);
    SIMDVec<SCALAR_UINT_T, FIR_ORDER> index_vec(helper.indices);
    SIMDVec<FLOAT_T, FIR_ORDER> state_vec, vec;

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
        state_vec.gather(state, index_vec);
        vec = state_vec.mul(coeff_vec);
        y[i] = vec.hadd();
        index_vec = index_vec.template swizzle<7, 0, 1, 2, 3, 4, 5, 6>();

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


template<typename FLOAT_T>
UME_NEVER_INLINE TIMING_RES test_ume_FIR16()
{
    typedef typename SIMDTraits<SIMDVec<FLOAT_T, 16>>::SCALAR_UINT_T SCALAR_UINT_T;
    const unsigned int FIR_ORDER = 16;

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

    for(uint32_t i = 0; i < FIR_ORDER; i++)
    {
        // Generate random coefficients in range (0.0; 1.0)
        // Inverse the coefficients to simplify the initialization
        coeffs[FIR_ORDER - i - 1] = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        state[i] = static_cast<FLOAT_T>(0);
    }

    start = get_timestamp();

    VerticalFIRHelper<SCALAR_UINT_T, FIR_ORDER> helper;
    SIMDVec<FLOAT_T, FIR_ORDER> coeff_vec(coeffs);
    SIMDVec<SCALAR_UINT_T, FIR_ORDER> index_vec(helper.indices);
    SIMDVec<FLOAT_T, FIR_ORDER> state_vec, vec;

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
        state_vec.gather(state, index_vec);
        vec = state_vec.mul(coeff_vec);
        y[i] = vec.hadd();
        index_vec = index_vec.template swizzle<15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14>();

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

template<typename FLOAT_T>
void benchmarkSIMD_FIR4(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_ume_FIR4<FLOAT_T>());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}


template<typename FLOAT_T>
void benchmarkSIMD_FIR8(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_ume_FIR8<FLOAT_T>());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}


template<typename FLOAT_T>
void benchmarkSIMD_FIR16(std::string const & resultPrefix,
                   int iterations,
                   TimingStatistics & reference)
{
    TimingStatistics stats;

    for (int i = 0; i < iterations; i++)
    {
        stats.update(test_ume_FIR16<FLOAT_T>());
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

#endif

