#ifndef QUADRATIC_SOLVER_OPTIMIZED_H_
#define QUADRATIC_SOLVER_OPTIMIZED_H_

// This piece of code comes from VecGeom project and was authored by Guilherme Amadio. 
// VecGeom is a vectorized geometry library for particle-detector simulation. The original code is available at:
// https://gitlab.cern.ch/VecGeom/VecGeom/blob/veccore-autovec/VecCore/examples/quadratic.cc
//
// This code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
//
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

template <typename T>
UME_NEVER_INLINE void QuadSolveOptimized(const T& a, const T& b, const T& c, T &x1, T &x2, int& roots) {
    T a_inv = T(1.0) / a;
    T delta = b * b - T(4.0) * a * c;
    T s = (b >= 0) ? T(1.0) : T(-1.0);

    roots = delta > std::numeric_limits<T>::epsilon() ? 2 : delta < T(0.0) ? 0 : 1;

    switch (roots) {
    case 2:
        x1 = T(-0.5) * (b + s * std::sqrt(delta));
        x2 = c / x1;
        x1 *= a_inv;
        return;

    case 0: return;

    case 1:
        x1 = x2 = T(-0.5) * b * a_inv;
        return;

    default: return;
    }
}

template<typename FLOAT_T>
UME_NEVER_INLINE TIMING_RES run_scalar_optimized()
{
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

    for (int i = 0; i < ARRAY_SIZE; i++) {
        QuadSolveOptimized<FLOAT_T>(a[i], b[i], c[i], x1[i], x2[i], roots[i]);
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
