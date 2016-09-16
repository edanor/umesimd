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

#ifndef QUADRATIC_SOLVER_SIMD_H_
#define QUADRATIC_SOLVER_SIMD_H_

template <typename SCALAR_FLOAT_T, typename FLOAT_VEC_T, typename INT_VEC_T>
UME_NEVER_INLINE void QuadSolveSIMD(
#if defined(_MSC_VER)
    const SCALAR_FLOAT_T* __restrict a,
    const SCALAR_FLOAT_T* __restrict b,
    const SCALAR_FLOAT_T* __restrict c,
    SCALAR_FLOAT_T* __restrict x1,
    SCALAR_FLOAT_T* __restrict x2,
    int* __restrict roots
#else
    const SCALAR_FLOAT_T* __restrict__ a,
    const SCALAR_FLOAT_T* __restrict__ b,
    const SCALAR_FLOAT_T* __restrict__ c,
    SCALAR_FLOAT_T* __restrict__ x1,
    SCALAR_FLOAT_T* __restrict__ x2,
    int* __restrict__ roots
#endif
    )
{
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::MASK_T MASK_T;

    FLOAT_VEC_T one(1.0f);
    FLOAT_VEC_T va(&a[0]);
    FLOAT_VEC_T vb(&b[0]);
    FLOAT_VEC_T zero(0.0f);
    FLOAT_VEC_T a_inv = one / va;
    FLOAT_VEC_T b2 = vb * vb;
    FLOAT_VEC_T eps(std::numeric_limits<SCALAR_FLOAT_T>::epsilon());
    FLOAT_VEC_T vc(&c[0]);
    FLOAT_VEC_T negone(-1.0f);
    FLOAT_VEC_T ac = va * vc;
    FLOAT_VEC_T sign = negone.blend(vb >= zero, one);
    FLOAT_VEC_T negfour(-4.0f);
    FLOAT_VEC_T delta = negfour.fmuladd(ac, b2);
    FLOAT_VEC_T r1 = sign.fmuladd(delta.sqrt(), vb);
    MASK_T mask0 = delta < zero;
    MASK_T mask2 = delta >= eps;
    r1 = r1.mul(-0.5f);
    FLOAT_VEC_T r2 = vc / r1;
    r1 = a_inv * r1;
    FLOAT_VEC_T r3 = vb * a_inv * (-0.5f);
    FLOAT_VEC_T two(2.0f);
    FLOAT_VEC_T nr = one.blend(mask2, two);
    nr = nr.blend(mask0, zero);
    r3 = r3.blend(mask0, zero);
    r1 = r3.blend(mask2, r1);
    r2 = r3.blend(mask2, r2);

    INT_VEC_T int_roots(nr);
    UME::SIMD::SIMDVec<int, INT_VEC_T::length()> int_roots2(int_roots);
    int_roots2.store(roots);
    r1.store(x1);
    r2.store(x2);
}

template<typename FLOAT_T, uint32_t LENGTH>
UME_NEVER_INLINE TIMING_RES run_SIMD()
{
    typedef typename UME::SIMD::SIMDVec<FLOAT_T, LENGTH>                FLOAT_VEC_T;
    typedef typename UME::SIMD::SIMDTraits<FLOAT_VEC_T>::INT_VEC_T      INT_VEC_T;

    unsigned long long start, end; // Time measurements

                                   // Align everything to a cacheline boundary
    FLOAT_T *a = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);
    FLOAT_T *b = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);
    FLOAT_T *c = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);

    int *roots = (int *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(int), 64);
    FLOAT_T *x1 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);
    FLOAT_T *x2 = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);

    srand((unsigned int)time(NULL));

    // Initialize arrays with random data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1.0)
        FLOAT_T t0 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        a[i] = FLOAT_T(10.0) * (t0 - FLOAT_T(0.5));
        t0 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        b[i] = FLOAT_T(10.0) * (t0 - FLOAT_T(0.5));
        t0 = static_cast <FLOAT_T> (rand()) / static_cast <FLOAT_T> (RAND_MAX);
        c[i] = FLOAT_T(50.0) * (t0 - FLOAT_T(0.5));
        x1[i] = FLOAT_T(0.0);
        x2[i] = FLOAT_T(0.0);
        roots[i] = 0;
    }

    volatile FLOAT_T x1_dump = FLOAT_T(0), x2_dump = FLOAT_T(0);
    volatile int root_dump = 0;

    start = __rdtsc();
    for (int i = 0; i < ARRAY_SIZE; i += LENGTH) {
        QuadSolveSIMD<FLOAT_T, FLOAT_VEC_T, INT_VEC_T>(&a[i], &b[i], &c[i], &x1[i], &x2[i], &roots[i]);
        
        // Useful for debugging
        //for (int k = 0; k < LENGTH; k++) {
        //    FLOAT_T t0 = FLOAT_T(0.0f), t1 = FLOAT_T(0.0f);
        //    int t2 = QuadSolveNaive<FLOAT_T>(a[i + k], b[i + k], c[i + k], t0, t1);
        //    if (roots[i + k] != t2)
        //    {
        //        std::cout << "Result invalid! (roots: " << roots[i + k] << " expected: " << t2 << std::endl;
        //    }
        //}
    }
    end = __rdtsc();

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // Use all generated results to prevent compiler optimizations
        root_dump += roots[i];
        x1_dump += x1[i];
        x2_dump += x2[i];

        // Verify the result using reference solver
        FLOAT_T t0 = FLOAT_T(0.0f), t1 = FLOAT_T(0.0f);
        int t2 = QuadSolveNaive<FLOAT_T>(a[i], b[i], c[i], t0, t1);
        if (roots[i] != t2)
        {
            std::cout << "Result invalid! (roots: " << roots[i] << " expected: " << t2 << std::endl;
        }
    }

    UME::DynamicMemory::AlignedFree(a);
    UME::DynamicMemory::AlignedFree(b);
    UME::DynamicMemory::AlignedFree(c);
    UME::DynamicMemory::AlignedFree(roots);
    UME::DynamicMemory::AlignedFree(x1);
    UME::DynamicMemory::AlignedFree(x2);

    return end - start;
}

#endif
