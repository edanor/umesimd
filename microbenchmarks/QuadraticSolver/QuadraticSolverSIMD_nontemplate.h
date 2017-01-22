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

#ifndef QUADRATIC_SOLVER_SIMD_NONTEMPLATE_H_
#define QUADRATIC_SOLVER_SIMD_NONTEMPLATE_H_

UME_NEVER_INLINE void QuadSolveSIMD8_32f(
#if defined(_MSC_VER)
    const float* __restrict a,
    const float* __restrict b,
    const float* __restrict c,
    float* __restrict x1,
    float* __restrict x2,
    int* __restrict roots
#else
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ x1,
    float* __restrict__ x2,
    int* __restrict__ roots
#endif
    )
{
    using namespace UME::SIMD;

    SIMD8_32f one(1.0f);
    SIMD8_32f va(&a[0]);
    SIMD8_32f vb(&b[0]);
    SIMD8_32f zero(0.0f);
    SIMD8_32f a_inv = one / va;
    SIMD8_32f b2 = vb * vb;
    SIMD8_32f eps(std::numeric_limits<float>::epsilon());
    SIMD8_32f vc(&c[0]);
    SIMD8_32f negone(-1.0f);
    SIMD8_32f ac = va * vc;
    SIMD8_32f sign = negone.blend(vb >= zero, one);
    SIMD8_32f negfour(-4.0f);
    SIMD8_32f delta = negfour.fmuladd(ac, b2);
    SIMD8_32f r1 = sign.fmuladd(delta.sqrt(), vb);
    SIMDMask8 mask0 = delta < zero;
    SIMDMask8 mask2 = delta >= eps;
    r1 = r1.mul(-0.5f);
    SIMD8_32f r2 = vc / r1;
    r1 = a_inv * r1;
    SIMD8_32f r3 = vb * a_inv * (-0.5f);
    SIMD8_32f two(2.0f);
    SIMD8_32f nr = one.blend(mask2, two);
    nr = nr.blend(mask0, zero);
    r3 = r3.blend(mask0, zero);
    r1 = r3.blend(mask2, r1);
    r2 = r3.blend(mask2, r2);

    SIMD8_32i int_roots(nr);
    int_roots.sstore(roots);
    r1.sstore(x1);
    r2.sstore(x2);
}

UME_NEVER_INLINE TIMING_RES run_SIMD_nontemplate()
{
    unsigned long long start, end; // Time measurements

                                   // Align everything to a cacheline boundary
    float *a = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(float), 64);
    float *b = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(float), 64);
    float *c = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(float), 64);

    int *roots = (int *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(int), 64);
    float *x1 = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(float), 64);
    float *x2 = (float *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(float), 64);

    srand((unsigned int)time(NULL));

    // Initialize arrays with random data
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        // Generate random numbers in range (0.0;1.0)
        float t0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        a[i] = float(10.0) * (t0 - float(0.5));
        t0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        b[i] = float(10.0) * (t0 - float(0.5));
        t0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        c[i] = float(50.0) * (t0 - float(0.5));
        x1[i] = float(0.0);
        x2[i] = float(0.0);
        roots[i] = 0;
    }

    volatile float x1_dump = float(0), x2_dump = float(0);
    volatile int root_dump = 0;

    start = get_timestamp();
    for (int i = 0; i < ARRAY_SIZE; i += 8) {
        QuadSolveSIMD8_32f(&a[i], &b[i], &c[i], &x1[i], &x2[i], &roots[i]);

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
    end = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // Use all generated results to prevent compiler optimizations
        root_dump += roots[i];
        x1_dump += x1[i];
        x2_dump += x2[i];

        // Verify the result using reference solver
        float t0 = float(0.0f), t1 = float(0.0f);
        int t2 = QuadSolveNaive<float>(a[i], b[i], c[i], t0, t1);
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
