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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "../../UMESimd.h"
#include "../utilities/UMEBitmap.h"
#include "../utilities/TimingStatistics.h"

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
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#endif

#include "mandel_intel.h"

typedef unsigned long long TIMING_RES;

template<typename VEC_T>
void mandel_umesimd(
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T x1, 
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T y1, 
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T x2, 
    typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T y2, 
    int width, 
    int height, 
    int maxIters, 
    uint8_t * image)
{
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::INT_VEC_T    INT_VEC_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T       MASK_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T     SCALAR_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_INT_T SCALAR_INT_T;

    constexpr int ALIGNMENT = VEC_T::alignment();
    constexpr int VEC_LEN = VEC_T::length();

    SCALAR_T dx = (x2 - x1) / width;
    SCALAR_T dy = (y2 - y1) / height;
    // round up width to next multiple of 8
    SCALAR_INT_T roundedWidth = (width + 7) & ~7UL;

    SCALAR_T constants[] = { dx, dy, x1, y1, 1.0f, 4.0f };
    VEC_T ymm0 = VEC_T(constants[0]);   // all dx
    VEC_T ymm1 = VEC_T(constants[1]); // all dy
    VEC_T ymm2 = VEC_T(constants[2]); // all x1
    VEC_T ymm3 = VEC_T(constants[3]); // all y1
    VEC_T ymm4 = VEC_T(constants[4]); // all 1's (iter increments)
    VEC_T ymm5 = VEC_T(constants[5]); // all 4's (comparisons)

    // Define increment for maximum allowed length of vector
    alignas(ALIGNMENT) SCALAR_T incr[32] = { 
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
        8.0f,  9.0f,  10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f}; // used to reset the i position when j increases
    VEC_T ymm6 = VEC_T(SCALAR_T(0)); // zero out j counter (ymm0 is just a dummy)

    alignas(ALIGNMENT) SCALAR_INT_T raw_outputs[VEC_LEN];

    for (int j = 0; j < height; j += 1)
    {
        VEC_T ymm7;  // i counter set to 0,1,2,..,7
        ymm7.loada(incr);
        for (int i = 0; i < roundedWidth; i += VEC_LEN)
        {
            VEC_T ymm8 = ymm7 * ymm0;  // x0 = (i+k)*dx 
            ymm8 = ymm8 + ymm2;         // x0 = x1+(i+k)*dx
            VEC_T ymm9 = ymm6 * ymm1;  // y0 = j*dy
            ymm9 = ymm9 + ymm3;         // y0 = y1+j*dy
            VEC_T ymm10 = VEC_T(SCALAR_T(0));  // zero out iteration counter (ymm0 is just a dummy)
            VEC_T ymm11 = ymm10, ymm12 = ymm10;        // set initial xi=0, yi=0

            unsigned int test = 0;
            int iter = 0;
            do
            {
                VEC_T ymm13 = ymm11 * ymm11; // xi*xi
                VEC_T ymm14 = ymm12 * ymm12; // yi*yi
                VEC_T ymm15 = ymm13 + ymm14; // xi*xi+yi*yi

                MASK_T mask = ymm15 < ymm5;        // xi*xi+yi*yi < 4 in each slot
                                                                       // now ymm15 has all 1s in the non overflowed locations
                test = mask.hlor();      // lower 8 bits are comparisons
                VEC_T ymm16 = VEC_T(SCALAR_T(0)); 
                ymm16.assign(mask, ymm4); // get 1.0f or 0.0f in each field as counters
                ymm10 = ymm10 + ymm16;        // counters for each pixel iteration

                ymm15 = ymm11 * ymm12;        // xi*yi

                ymm11 = ymm13 - ymm14;        // xi*xi-yi*yi
                ymm11 = ymm11 + ymm8;         // xi <- xi*xi-yi*yi+x0 done!
                ymm12 = ymm15 + ymm15;        // 2*xi*yi
                ymm12 = ymm12 + ymm9;         // yi <- 2*xi*yi+y0            

                ++iter;
            } while ((test != 0) && (iter < maxIters));

            // convert iterations to output values
            INT_VEC_T ymm10i = INT_VEC_T(ymm10);

            // write only where needed
            ymm10i.storea((SCALAR_INT_T*)raw_outputs);
            int top = (i + VEC_LEN - 1) < width ? VEC_LEN : width & (VEC_LEN-1);
            for (int k = 0; k < top; ++k)
                image[i + k + j*width] = (uint8_t)raw_outputs[k];

            // next i position - increment each slot by 8
            ymm7 = ymm7 + SCALAR_T(VEC_LEN);
        }
        ymm6 = ymm6 + ymm4; // increment j counter
    }
}

#ifdef __x86_64__
#include <cpuid.h>

static inline int
is_avx_supported(void)
{
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    return ecx & bit_AVX ? 1 : 0;
}
#endif // __x86_64__

template<typename SIMD_T>
void benchmarkUMESIMD(int width,
                      int height,
                      char * filename, 
                      char * resultPrefix, 
                      int iterations,
                      TimingStatistics & reference)
{
    TimingStatistics stats;

    UME::Bitmap bmp(width, height, UME::PIXEL_TYPE_RGB);
    uint8_t* image = bmp.GetRasterData();

    uint8_t *raw_image;

    raw_image = (uint8_t *)UME::DynamicMemory::AlignedMalloc(width*height*sizeof(uint8_t), SIMD_T::alignment());

    for (int i = 0; i < iterations; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(uint8_t));

        start = __rdtsc();
        mandel_umesimd<SIMD_T>(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, 4096, raw_image);
        end = __rdtsc();

        stats.update(end - start);

        // Rewrite algorithm output to BMP format
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                image[3 * (h*width + w) + 0] = raw_image[h*width + w];
                image[3 * (h*width + w) + 1] = raw_image[h*width + w];
                image[3 * (h*width + w) + 2] = raw_image[h*width + w];
            }
        }

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile(filename);
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;

    UME::DynamicMemory::AlignedFree(raw_image);
}

alignas(32) unsigned short g_raw_image[640 * 640];
int
main(int argc, char *argv[])
{
    int ITERATIONS = 100;

    /* Config */
    int width = 640;
    int height = 640;

    TimingStatistics stats_scalar_32f,
                     stats_scalar_64f,
                     stats_sse,
                     stats_avx,
                     stats_avx2,
                     stats_SIMD1_32f,
                     stats_SIMD2_32f,
                     stats_SIMD4_32f,
                     stats_SIMD4_32f_MFI,
                     stats_SIMD8_32f,
                     stats_SIMD16_32f,
                     stats_SIMD32_32f,
                     stats_SIMD1_64f,
                     stats_SIMD2_64f,
                     stats_SIMD4_64f,
                     stats_SIMD8_64f,
                     stats_SIMD16_64f;

    alignas(64) unsigned short raw_image[640 * 640];

    UME::Bitmap bmp(width, height, UME::PIXEL_TYPE_RGB);
    uint8_t* image = bmp.GetRasterData();

    std::cout << "The result is amount of time it takes to calculate mandelbrot algorithm.\n"
        "All timing results in clock cycles. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD version uses following operations: \n"
        "   32f vectors: SET-CONSTR, LOAD-CONSTR, LOADA, MULV (operator*), ADDV (operator+), \n"
        "              CMPLTV (operator<), SUBV (operator-), ASSIGNV (operator=), MASSIGNV,\n"
        "              ADDS (operator+ RHS scalar), FTOI\n"
        "   32i vectors: STOREA\n"
        "   masks:       HLOR\n"
        " Algorithm parameters are:\n"
        "     image width: " << width << "\n"
        "     image height: " << height << "\n"
        "     # of iterations: " << 4096 << "\n"
        "     # of executions per measurement: " << ITERATIONS << "\n\n";

    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(unsigned short));

        start = __rdtsc();
        MandelbrotCPU2(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, 4096, raw_image);
        end = __rdtsc();

        stats_scalar_32f.update(end - start);

        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                image[3 * k * 640 + 3 * j] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 1] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 2] = (uint8_t)(raw_image[k * 640 + j]);
            }
        }

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_basic_32f.bmp");
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << "Scalar code (float): " << (unsigned long long)stats_scalar_32f.getAverage()
        << ", dev: " << (unsigned long long)stats_scalar_32f.getStdDev()
        << " (speedup: 1.0x )" 
        << std::endl;

    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(unsigned short));

        start = __rdtsc();
        MandelbrotCPU2_64f(0.29768, 0.48364, 0.29778, 0.48354, width, height, 4096, raw_image);
        end = __rdtsc();

        stats_scalar_64f.update(end - start);

        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                image[3 * k * 640 + 3 * j] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 1] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 2] = (uint8_t)(raw_image[k * 640 + j]);
            }
        }

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_basic_64f.bmp");
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << "Scalar code (double): " << (unsigned long long)stats_scalar_64f.getAverage()
        << ", dev: " << (unsigned long long)stats_scalar_64f.getStdDev()
        << " (speedup: "
        << stats_scalar_64f.calculateSpeedup(stats_scalar_32f) << ")\n";

#if defined __SSE__
    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(unsigned short));
        start = __rdtsc();
        //MandelbrotSSE2(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, 4096, raw_image);
        end = __rdtsc();

        stats_sse.update(end - start);
        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                image[3 * k * 640 + 3 * j] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 1] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 2] = (uint8_t)(raw_image[k * 640 + j]);
            }
        }
        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_intel_sse.bmp");
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << "SSE intrinsics code (float): " << (unsigned long long)stats_sse.getAverage()
        << ", dev: " << (unsigned long long)stats_sse.getStdDev()
        << " (speedup: "
        << stats_sse.calculateSpeedup(stats_scalar_32f) << ")\n";
#else
    std::cout << "SSE intrinsics code (float): not used\n";
#endif

#if defined __AVX__
    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(unsigned short));
        start = __rdtsc();
        MandelbrotAVX(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, 4096, raw_image);
        end = __rdtsc();

        stats_avx2.update(end - start);
        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                image[3 * k * 640 + 3 * j] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 1] = (uint8_t)(raw_image[k * 640 + j]);
                image[3 * k * 640 + 3 * j + 2] = (uint8_t)(raw_image[k * 640 + j]);
            }
        }
        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_intel_avx.bmp");
        bmp.ClearTarget(0, 255, 0);
}

    std::cout << "AVX intrinsics code (float): " << (unsigned long long)stats_avx2.getAverage()
        << ", dev: " << (unsigned long long)stats_avx2.getStdDev()
        << " (speedup: "
        << stats_avx2.calculateSpeedup(stats_scalar_32f) << ")\n";
#else
    std::cout << "AVX intrinsics code (float): not used\n";
#endif


    benchmarkUMESIMD<UME::SIMD::SIMD1_32f>(width, height, "mandel_umesimd_1_32f.bmp", "SIMD code (1x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_32f>(width, height, "mandel_umesimd_2_32f.bmp", "SIMD code (2x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_32f>(width, height, "mandel_umesimd_4_32f.bmp", "SIMD code (4x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_32f>(width, height, "mandel_umesimd_8_32f.bmp", "SIMD code (8x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_32f>(width, height, "mandel_umesimd_16_32f.bmp", "SIMD code (16x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD32_32f>(width, height, "mandel_umesimd_32_32f.bmp", "SIMD code (32x32f): ", ITERATIONS, stats_scalar_32f);

    benchmarkUMESIMD<UME::SIMD::SIMD1_64f>(width, height, "mandel_umesimd_1_64f.bmp", "SIMD code (1x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_64f>(width, height, "mandel_umesimd_2_64f.bmp", "SIMD code (2x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_64f>(width, height, "mandel_umesimd_4_64f.bmp", "SIMD code (4x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_64f>(width, height, "mandel_umesimd_8_64f.bmp", "SIMD code (8x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_64f>(width, height, "mandel_umesimd_16_64f.bmp", "SIMD code (16x64f): ", ITERATIONS, stats_scalar_32f);

    return 0;
}
