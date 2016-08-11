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

#if !defined(__GNUG__)
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
#endif

typedef unsigned long long TIMING_RES;

struct Color{
    uint8_t r, g, b;
    Color() {}
    Color(uint8_t R, uint8_t G, uint8_t B) : r(R), g(G), b(B) {}
    void setRgb(uint8_t R, uint8_t G, uint8_t B) {
        r = R;
        g = G;
        b = B;
    }
};

Color getColor(int value) {
    if (value < 65535 && value > 0) {
        int i = value % 16;
        Color mapping[16];
        mapping[0].setRgb(66, 30, 15);
        mapping[1].setRgb(25, 7, 26);
        mapping[2].setRgb(9, 1, 47);
        mapping[3].setRgb(4, 4, 73);
        mapping[4].setRgb(0, 7, 100);
        mapping[5].setRgb(12, 44, 138);
        mapping[6].setRgb(24, 82, 177);
        mapping[7].setRgb(57, 125, 209);
        mapping[8].setRgb(134, 181, 229);
        mapping[9].setRgb(211, 236, 248);
        mapping[10].setRgb(241, 233, 191);
        mapping[11].setRgb(248, 201, 95);
        mapping[12].setRgb(255, 170, 0);
        mapping[13].setRgb(204, 128, 0);
        mapping[14].setRgb(153, 87, 0);
        mapping[15].setRgb(106, 52, 3);
        return mapping[i];
    }
    return Color(0, 0, 0);
}

#include "mandel_intel.h"
#include "mandel_ume.h"
#include "mandel_openmp.h"



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

alignas(32) unsigned short g_raw_image[640 * 640];
int
main(int argc, char *argv[])
{
    int ITERATIONS = 20;

    /* Config */
    int width = 640;
    int height = 640;
    int depth = 4096;

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
        "     # of iterations: " << depth << "\n"
        "     # of executions per measurement: " << ITERATIONS << "\n\n";

    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(unsigned short));

        start = __rdtsc();
        MandelbrotCPU2(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, depth, raw_image);
        end = __rdtsc();

        stats_scalar_32f.update(end - start);

        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                int value = raw_image[k*640 + j];
                Color c = getColor(value);
                image[3 * k * 640 + 3 * j] = c.r;
                image[3 * k * 640 + 3 * j + 1] = c.g;
                image[3 * k * 640 + 3 * j + 2] = c.b;
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
        MandelbrotCPU2_64f(0.29768, 0.48364, 0.29778, 0.48354, width, height, depth, raw_image);
        end = __rdtsc();

        stats_scalar_64f.update(end - start);

        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                int value = raw_image[k*640 + j];
                Color c = getColor(value);
                image[3 * k * 640 + 3 * j] = c.r;
                image[3 * k * 640 + 3 * j + 1] = c.g;
                image[3 * k * 640 + 3 * j + 2] = c.b;
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
        //MandelbrotSSE2(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, depth, raw_image);
        end = __rdtsc();

        stats_sse.update(end - start);
        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                int value = raw_image[k*640 + j];
                Color c = getColor(value);
                image[3 * k * 640 + 3 * j] = c.r;
                image[3 * k * 640 + 3 * j + 1] = c.g;
                image[3 * k * 640 + 3 * j + 2] = c.b;
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
        MandelbrotAVX(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, depth, raw_image);
        end = __rdtsc();

        stats_avx2.update(end - start);
        for (int k = 0; k < 640; k++) {
            for (int j = 0; j < 640; j++) {
                int value = raw_image[k*640 + j];
                Color c = getColor(value);
                image[3 * k * 640 + 3 * j] = c.r;
                image[3 * k * 640 + 3 * j + 1] = c.g;
                image[3 * k * 640 + 3 * j + 2] = c.b;
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

    benchmarkOpenMP(width, height, depth, "mandel openmp.bmp", "Openmp: ", ITERATIONS, stats_scalar_32f);

    benchmarkUMESIMD<UME::SIMD::SIMD1_32f>(width, height, depth, "mandel_umesimd_1_32f.bmp", "SIMD code (1x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_32f>(width, height, depth, "mandel_umesimd_2_32f.bmp", "SIMD code (2x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_32f>(width, height, depth, "mandel_umesimd_4_32f.bmp", "SIMD code (4x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_32f>(width, height, depth, "mandel_umesimd_8_32f.bmp", "SIMD code (8x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_32f>(width, height, depth, "mandel_umesimd_16_32f.bmp", "SIMD code (16x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD32_32f>(width, height, depth, "mandel_umesimd_32_32f.bmp", "SIMD code (32x32f): ", ITERATIONS, stats_scalar_32f);

    benchmarkUMESIMD<UME::SIMD::SIMD1_64f>(width, height, depth, "mandel_umesimd_1_64f.bmp", "SIMD code (1x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_64f>(width, height, depth, "mandel_umesimd_2_64f.bmp", "SIMD code (2x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_64f>(width, height, depth, "mandel_umesimd_4_64f.bmp", "SIMD code (4x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_64f>(width, height, depth, "mandel_umesimd_8_64f.bmp", "SIMD code (8x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_64f>(width, height, depth, "mandel_umesimd_16_64f.bmp", "SIMD code (16x64f): ", ITERATIONS, stats_scalar_32f);

    return 0;
}
