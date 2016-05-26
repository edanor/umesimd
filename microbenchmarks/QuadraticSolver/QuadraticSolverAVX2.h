#ifndef QUADRATIC_SOLVER_AVX2_H_
#define QUADRATIC_SOLVER_AVX2_H_

   // This piece of code comes from VecGeom project and was authored by Guilherme Amadio. 
   // VecGeom is a vectorized geometry library for particle-detector simulation. The original code is available at:
   // https://gitlab.cern.ch/VecGeom/VecGeom/blob/veccore-autovec/VecCore/examples/quadratic.cc
   //
   // This code is not a part of UME::SIMD library code and is used purely for
   // performance measurement reference.
   //
   // Modifications have been made to original files to fit them for benchmarking
   // of UME::SIMD.

#ifdef __AVX2__
// explicit AVX2 code using intrinsics

UME_FORCE_INLINE void QuadSolveAVX2(
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    float* __restrict__ x1,
    float* __restrict__ x2,
    int* __restrict__ roots)
{
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 va = _mm256_load_ps(a);
    __m256 vb = _mm256_load_ps(b);
    __m256 zero = _mm256_set1_ps(0.0f);
    __m256 a_inv = _mm256_div_ps(one, va);
    __m256 b2 = _mm256_mul_ps(vb, vb);
    __m256 eps = _mm256_set1_ps(std::numeric_limits<float>::epsilon());
    __m256 vc = _mm256_load_ps(c);
    __m256 negone = _mm256_set1_ps(-1.0f);
    __m256 ac = _mm256_mul_ps(va, vc);
    __m256 sign = _mm256_blendv_ps(negone, one, _mm256_cmp_ps(vb, zero, _CMP_GE_OS));
#ifdef __FMA__
    __m256 delta = _mm256_fmadd_ps(_mm256_set1_ps(-4.0f), ac, b2);
    __m256 r1 = _mm256_fmadd_ps(sign, _mm256_sqrt_ps(delta), vb);
#else
    __m256 delta = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(-4.0f), ac), b2);
    __m256 r1 = _mm256_add_ps(_mm256_mul_ps(sign, _mm256_sqrt_ps(delta)), vb);
#endif
    __m256 mask0 = _mm256_cmp_ps(delta, zero, _CMP_LT_OS);
    __m256 mask2 = _mm256_cmp_ps(delta, eps, _CMP_GE_OS);
    r1 = _mm256_mul_ps(_mm256_set1_ps(-0.5f), r1);
    __m256 r2 = _mm256_div_ps(vc, r1);
    r1 = _mm256_mul_ps(a_inv, r1);
    __m256 r3 = _mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_mul_ps(vb, a_inv));
    __m256 nr = _mm256_blendv_ps(one, _mm256_set1_ps(2), mask2);
    nr = _mm256_blendv_ps(nr, _mm256_set1_ps(0), mask0);
    r3 = _mm256_blendv_ps(r3, zero, mask0);
    r1 = _mm256_blendv_ps(r3, r1, mask2);
    r2 = _mm256_blendv_ps(r3, r2, mask2);
    _mm256_store_si256((__m256i*)roots, _mm256_cvtps_epi32(nr));
    _mm256_store_ps(x1, r1);
    _mm256_store_ps(x2, r2);
}

UME_NEVER_INLINE TIMING_RES run_AVX2()
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
        a[i] = 10.0f * (t0 - 0.5f);
        t0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        b[i] = 10.0f * (t0 - 0.5f);
        t0 = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        c[i] = 50.0f * (t0 - 0.5f);
        x1[i] = 0.0f;
        x2[i] = 0.0f;
        roots[i] = 0;
    }

    volatile float x1_dump = 0.0f, x2_dump = 0.0f;
    volatile int root_dump = 0;
    start = __rdtsc();

    for (int i = 0; i < ARRAY_SIZE; i+=8) {
        QuadSolveAVX2(&a[i], &b[i], &c[i], &x1[i], &x2[i], &roots[i]);
    }
    end = __rdtsc();

    for (int i = 0; i < ARRAY_SIZE; i++) {
        // Use all generated results to prevent compiler optimizations
        root_dump += roots[i];
        x1_dump += x1[i];
        x2_dump += x2[i];

        // Verify the result using reference solver
        float t0 = 0.0f, t1 = 0.0f;
        int t2 = QuadSolveNaive<float>(a[i], b[i], c[i], t0, t1);
        if (roots[i] != t2)
        {
            std::cout << "Result invalid! (roots: " << roots[i] << " expected: " << t2;
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


#endif
