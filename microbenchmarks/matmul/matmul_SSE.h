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

#ifndef MATMUL_SSE_H_
#define MATMUL_SSE_H_

template<int MAT_RANK>
RESULTS<float> test_sse_32f()
{
    unsigned long long start, end; // Time measurements
    float *A, *B, *B_T, *C;

    // All arrays should be padded, so that rows start at optimal alignment.
    // Making each row of A and column of B padded, also simplifies 
    int SIMD_STRIDE = 4;
    int PADDING = SIMD_STRIDE - (MAT_RANK % SIMD_STRIDE);

    // Allocate alligned to a single scalar
    A = (float *)UME::DynamicMemory::AlignedMalloc((MAT_RANK + PADDING)*MAT_RANK*sizeof(float), 16);
    // B doesn't have to be padded, since we will transpose it at the beginning of computation
    B = (float *)UME::DynamicMemory::AlignedMalloc(MAT_RANK*MAT_RANK*sizeof(float), 16);
    B_T = (float *)UME::DynamicMemory::AlignedMalloc((MAT_RANK + PADDING)*MAT_RANK*sizeof(float), 16);
    C = (float *)UME::DynamicMemory::AlignedMalloc((MAT_RANK)*MAT_RANK*sizeof(float), 16);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < MAT_RANK; i++)
    {
        for (int j = 0; j < MAT_RANK; j++)
        {
            // Generate random numbers in range (0.0;1.0)
            A[i*(MAT_RANK + PADDING) + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            B[i*MAT_RANK + j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            C[i*(MAT_RANK)+j] = float(0);
        }
        for (int j = MAT_RANK; j < MAT_RANK + PADDING; j++) {
            A[i*(MAT_RANK + PADDING) + j] = 0.0f;
            //C[i*(MAT_RANK) + j] = 0.0f;
        }
    }
    /*
    std::cout << "\n\nA = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << A[i*(MAT_RANK +PADDING) + j] << " ";
    }
    std::cout << ";\n";
    }

    std::cout << "\n\nB = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B[i*MAT_RANK + j] << " ";
    }
    std::cout << ";\n";
    }*/


    start = __rdtsc();

    // Transpose B matrix to a row-major form
    for (int i = 0; i < MAT_RANK; i++) {
        for (int j = 0; j < MAT_RANK;j++)
        {
            B_T[i*(MAT_RANK + PADDING) + j] = B[j*MAT_RANK + i];
        }
        for (int j = MAT_RANK; j < MAT_RANK + PADDING; j++) {
            B_T[i*(MAT_RANK + PADDING) + j] = 0.0f;
        }
    }

    // For each row in C
    alignas(16) float raw[4];
    for (int i = 0; i < MAT_RANK; i++) {
        // For each column in C
        for (int j = 0; j < MAT_RANK; j++) {
            __m128 t0 = _mm_setzero_ps();
            // Traverse single row of A and single column of B
            for (int k = 0; k < MAT_RANK + PADDING; k += SIMD_STRIDE) {
                __m128 t1 = _mm_load_ps(&A[i*(MAT_RANK + PADDING) + k]);
                __m128 t2 = _mm_load_ps(&B_T[j*(MAT_RANK + PADDING) + k]);
                __m128 t3 = _mm_mul_ps(t1, t2);
                __m128 t4 = _mm_add_ps(t0, t3);
                t0 = t4;
            }
            _mm_store_ps(raw, t0);
            C[i*(MAT_RANK)+j] = raw[0] + raw[1] + raw[2] + raw[3];
        }
    }

    end = __rdtsc();

    /*
    std::cout << "\n\nB': \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B_T[i*(MAT_RANK + PADDING) + j] << " ";
    }
    std::cout << std::endl;
    }

    std::cout << "\n\nC = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << C[i*(MAT_RANK + PADDING) + j] << " ";
    }
    std::cout << ";\n";
    }*/

    float error = calculate_RMS_error_SIMD<float, MAT_RANK, 4>(A, B, C);
    //std::cout << "AVX/AVX2 float RMS error: " << error << std::endl;

    UME::DynamicMemory::AlignedFree(A);
    UME::DynamicMemory::AlignedFree(B);
    UME::DynamicMemory::AlignedFree(B_T);
    UME::DynamicMemory::AlignedFree(C);

    RESULTS<float> retval;
    retval.elapsed = end - start;
    retval.RMS_error = error;

    return retval;
}

template<int MAT_RANK>
RESULTS<double> test_sse_64f()
{
    unsigned long long start, end; // Time measurements
    double *A, *B, *B_T, *C;

    // All arrays should be padded, so that rows start at optimal alignment.
    // Making each row of A and column of B padded, also simplifies 
    int SIMD_STRIDE = 2;
    int PADDING = SIMD_STRIDE - (MAT_RANK % SIMD_STRIDE);

    // Allocate alligned to a single scalar
    A = (double *)UME::DynamicMemory::AlignedMalloc((MAT_RANK + PADDING)*MAT_RANK*sizeof(double), 16);
    // B doesn't have to be padded, since we will transpose it at the beginning of computation
    B = (double *)UME::DynamicMemory::AlignedMalloc(MAT_RANK*MAT_RANK*sizeof(double), 16);
    B_T = (double *)UME::DynamicMemory::AlignedMalloc((MAT_RANK + PADDING)*MAT_RANK*sizeof(double), 16);
    C = (double *)UME::DynamicMemory::AlignedMalloc((MAT_RANK)*MAT_RANK*sizeof(double), 16);

    srand((unsigned int)time(NULL));
    // Initialize arrays with random data
    for (int i = 0; i < MAT_RANK; i++)
    {
        for (int j = 0; j < MAT_RANK; j++)
        {
            // Generate random numbers in range (0.0;1.0)
            A[i*(MAT_RANK + PADDING) + j] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            B[i*MAT_RANK + j] = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);
            C[i*(MAT_RANK)+j] = double(0);
        }
        for (int j = MAT_RANK; j < MAT_RANK + PADDING; j++) {
            A[i*(MAT_RANK + PADDING) + j] = 0.0;
            //C[i*(MAT_RANK) + j] = 0.0f;
        }
    }
    /*
    std::cout << "\n\nA = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << A[i*(MAT_RANK +PADDING) + j] << " ";
    }
    std::cout << ";\n";
    }

    std::cout << "\n\nB = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B[i*MAT_RANK + j] << " ";
    }
    std::cout << ";\n";
    }*/


    start = __rdtsc();

    // Transpose B matrix to a row-major form
    for (int i = 0; i < MAT_RANK; i++) {
        for (int j = 0; j < MAT_RANK;j++)
        {
            B_T[i*(MAT_RANK + PADDING) + j] = B[j*MAT_RANK + i];
        }
        for (int j = MAT_RANK; j < MAT_RANK + PADDING; j++) {
            B_T[i*(MAT_RANK + PADDING) + j] = 0.0;
        }
    }

    // For each row in C
    alignas(32) double raw[4];
    for (int i = 0; i < MAT_RANK; i++) {
        // For each column in C
        for (int j = 0; j < MAT_RANK; j++) {
            __m128d t0 = _mm_setzero_pd();
            // Traverse single row of A and single column of B
            for (int k = 0; k < MAT_RANK + PADDING; k += SIMD_STRIDE) {
                __m128d t1 = _mm_load_pd(&A[i*(MAT_RANK + PADDING) + k]);
                __m128d t2 = _mm_load_pd(&B_T[j*(MAT_RANK + PADDING) + k]);
                __m128d t3 = _mm_mul_pd(t1, t2);
                __m128d t4 = _mm_add_pd(t0, t3);
                t0 = t4;
            }
            _mm_store_pd(raw, t0);
            C[i*(MAT_RANK)+j] = raw[0] + raw[1] + raw[2] + raw[3];
        }
    }

    end = __rdtsc();

    /*
    std::cout << "\n\nB': \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << B_T[i*(MAT_RANK + PADDING) + j] << " ";
    }
    std::cout << std::endl;
    }

    std::cout << "\n\nC = [ \n";
    for (int i = 0; i < MAT_RANK; i++) {
    for (int j = 0; j < MAT_RANK; j++) {
    std::cout << C[i*(MAT_RANK + PADDING) + j] << " ";
    }
    std::cout << ";\n";
    }*/

    double error = calculate_RMS_error_SIMD<double, MAT_RANK, 2>(A, B, C);
    //std::cout << "AVX/AVX2 float RMS error: " << error << std::endl;

    UME::DynamicMemory::AlignedFree(A);
    UME::DynamicMemory::AlignedFree(B);
    UME::DynamicMemory::AlignedFree(B_T);
    UME::DynamicMemory::AlignedFree(C);

    RESULTS<double> retval;
    retval.elapsed = end - start;
    retval.RMS_error = error;

    return retval;
}


template<int MAT_RANK>
void benchmark_sse_32f(std::string const & resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;
    Statistics<float> errors;

    for (int i = 0; i < iterations; i++)
    {
        RESULTS<float> results = test_sse_32f<MAT_RANK>();
        stats.update(results.elapsed);
        errors.update(results.RMS_error);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", RMS error: " << errors.getAverage()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

template<int MAT_RANK>
void benchmark_sse_64f(std::string const & resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;
    Statistics<double> errors;

    for (int i = 0; i < iterations; i++)
    {
        RESULTS<double> results = test_sse_64f<MAT_RANK>();
        stats.update(results.elapsed);
        errors.update(results.RMS_error);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << ", error: " << errors.getAverage()
        << " (speedup: " << stats.calculateSpeedup(reference) << "x)\n";
}

#endif
