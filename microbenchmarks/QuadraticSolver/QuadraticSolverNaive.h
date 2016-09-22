#ifndef QUADRATIC_SOLVER_NAIVE_H_
#define QUADRATIC_SOLVER_NAIVE_H_

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
UME_NEVER_INLINE int QuadSolveNaive(T a, T b, T c, T &x1, T &x2) {
    T delta = b * b - T(4.0) * a * c;

    if (delta < 0.0)
        return 0;

    if (delta < std::numeric_limits<T>::epsilon()) {
        x1 = x2 = -T(0.5) * b / a;
        return 1;
    }

    if (b >= 0.0) {
        x1 = -T(0.5) * (b + std::sqrt(delta)) / a;
        x2 = c / (a * x1);
    }
    else {
        x2 = -T(0.5) * (b - std::sqrt(delta)) / a;
        x1 = c / (a * x2);
    }

    return 2;
}

template<typename FLOAT_T, typename INT_T>
UME_NEVER_INLINE TIMING_RES run_scalar_naive()
{
    unsigned long long start, end; // Time measurements

                                   // Align everything to a cacheline boundary
    FLOAT_T *a = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);
    FLOAT_T *b = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);
    FLOAT_T *c = (FLOAT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(FLOAT_T), 64);

    INT_T *roots = (INT_T *)UME::DynamicMemory::AlignedMalloc(ARRAY_SIZE * sizeof(INT_T), 64);
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
        roots[i] = INT_T(0);
    }

    volatile FLOAT_T root_dump = FLOAT_T(0), x1_dump = FLOAT_T(0), x2_dump = FLOAT_T(0);

    start = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i++) {
        roots[i] = QuadSolveNaive<FLOAT_T>(a[i], b[i], c[i], x1[i], x2[i]);
    }
    end = get_timestamp();

    for (int i = 0; i < ARRAY_SIZE; i++) {
        root_dump += roots[i];
        x1_dump += x1[i];
        x2_dump += x2[i];
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
