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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mandelbrot.h"
#include "mandelbrot_ume.h"

#include "../utilities/UMEBitmap.h"
#include "../utilities/TimingStatistics.h"

void mandel_basic_32f(unsigned char *image, const struct spec *s);
void mandel_basic_64f(unsigned char *image, const struct spec *s);
void mandel_avx(unsigned char *image, const struct spec *s);
void mandel_sse2(unsigned char *image, const struct spec *s);

template<typename SIMD_T>
void benchmarkUMESIMD(struct spec & spec,
                      std::string const & filename, 
                      std::string const & resultPrefix, 
                      int iterations,
                      TimingStatistics & reference)
{
    TimingStatistics stats;

    UME::Bitmap bmp(spec.width, spec.height, UME::PIXEL_TYPE_RGB);
    uint8_t* image = bmp.GetRasterData();

    for (int i = 0; i < iterations; i++) {
        TIMING_RES start, end;

        start = get_timestamp();
        mandel_umesimd<SIMD_T>(image, &spec);
        end = get_timestamp();

        stats.update(end - start);

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile(filename);
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}

template<typename SIMD_T>
void benchmarkUMESIMD_MFI(struct spec & spec,
    char * filename,
    char * resultPrefix,
    int iterations,
    TimingStatistics & reference)
{
    TimingStatistics stats;

    UME::Bitmap bmp(spec.width, spec.height, UME::PIXEL_TYPE_RGB);
    uint8_t* image = bmp.GetRasterData();

    for (int i = 0; i < iterations; i++) {
        TIMING_RES start, end;

        start = get_timestamp();
        mandel_umesimd_MFI<SIMD_T>(image, &spec);
        end = get_timestamp();

        stats.update(end - start);

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile(filename);
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << resultPrefix << (unsigned long long) stats.getAverage()
        << ", dev: " << (unsigned long long) stats.getStdDev()
        << " (speedup: "
        << stats.calculateSpeedup(reference) << ")"
        << std::endl;
}

int main()
{
    int ITERATIONS = 100;

    /* Config */
    struct spec spec;
    spec.width = 640;
    spec.height = 640;
    spec.depth = 256;
    spec.xlim[0] = -2.5;
    spec.xlim[1] = 1.5;
    spec.ylim[0] = -1.5;
    spec.ylim[1] = 1.5;
    spec.iterations = 256;

    TimingStatistics stats_scalar_32f,
                     stats_scalar_64f,
                     stats_sse2,
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

    UME::Bitmap bmp(spec.width, spec.height, UME::PIXEL_TYPE_RGB);
    uint8_t* image = bmp.GetRasterData();

    std::cout << "The result is amount of time it takes to calculate mandelbrot algorithm.\n"
        "All timing results in nanoseconds. \n"
        "Speedup calculated with scalar floating point result as reference.\n\n"
        "SIMD version uses following operations: \n"
        "   32f vectors: FULL-CONSTR, LOAD-CONSTR, ADDS (operator+ RHS scalar), \n"
        "              FMULADDV, MULV (operator*), SUBV (operator-), ADDV (operator+)\n"
        "              ASSIGNV (operator=), CMPLTV (operator<), MADDV, SQRT, ROUND\n"
        "              FTOI\n"
        "   32i vectors: STORE\n"
        "   masks:       HLOR\n"
        " Algorithm parameters are:\n"
        "     image width: " << spec.width << "\n"
        "     image height: " << spec.height << "\n"
        "     iteration depth: " << spec.depth << "\n"
        "     # of iterations: " << spec.iterations << "\n"
        "     # of executions per measurement: " << ITERATIONS << "\n\n";

    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        start = get_timestamp();
        mandel_basic_32f(image, &spec);
        end = get_timestamp();

        stats_scalar_32f.update(end - start);

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

        start = get_timestamp();
        mandel_basic_64f(image, &spec);
        end = get_timestamp();

        stats_scalar_64f.update(end - start);

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_basic_64f.bmp");
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << "Scalar code (double): " << (unsigned long long)stats_scalar_64f.getAverage()
        << ", dev: " << (unsigned long long)stats_scalar_64f.getStdDev()
        << " (speedup: "
        << stats_scalar_64f.calculateSpeedup(stats_scalar_32f) << ")\n";

#if defined __SSE2__ | _M_IX86_FP == 2 
    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        start = get_timestamp();
        mandel_sse2(image, &spec);
        end = get_timestamp();

        stats_sse2.update(end - start);

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_sse2.bmp");
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << "SSE2 intrinsic code (float): " << (unsigned long long)stats_sse2.getAverage()
        << ", dev: " << (unsigned long long)stats_sse2.getStdDev()
        << " (speedup: "
        << stats_sse2.calculateSpeedup(stats_scalar_32f)<< ")\n";
#endif

#if defined __AVX__
    for (int i = 0; i < ITERATIONS; i++) {
        TIMING_RES start, end;

        start = get_timestamp();
        mandel_avx(image, &spec);
        end = get_timestamp();

        stats_avx.update(end - start);

        // Saving to file to make sure the results generated are correct
        bmp.SaveToFile("mandel_avx.bmp");
        bmp.ClearTarget(0, 255, 0);
    }

    std::cout << "AVX intrinsic code (float): " << (unsigned long long)stats_avx.getAverage()
        << ", dev: " << (unsigned long long)stats_avx.getStdDev()
        << " (speedup: "
        << stats_avx.calculateSpeedup(stats_scalar_32f) << ")\n";
#endif

    benchmarkUMESIMD<UME::SIMD::SIMD1_32f>(spec, "mandel_umesimd_1_32f.bmp", "SIMD code (1x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_32f>(spec, "mandel_umesimd_2_32f.bmp", "SIMD code (2x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_32f>(spec, "mandel_umesimd_4_32f.bmp", "SIMD code (4x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_32f>(spec, "mandel_umesimd_8_32f.bmp", "SIMD code (8x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_32f>(spec, "mandel_umesimd_16_32f.bmp", "SIMD code (16x32f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD32_32f>(spec, "mandel_umesimd_32_32f.bmp", "SIMD code (32x32f): ", ITERATIONS, stats_scalar_32f);
    
    benchmarkUMESIMD<UME::SIMD::SIMD1_64f>(spec, "mandel_umesimd_1_64f.bmp", "SIMD code (1x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD2_64f>(spec, "mandel_umesimd_2_64f.bmp", "SIMD code (2x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD4_64f>(spec, "mandel_umesimd_4_64f.bmp", "SIMD code (4x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD8_64f>(spec, "mandel_umesimd_8_64f.bmp", "SIMD code (8x64f): ", ITERATIONS, stats_scalar_32f);
    benchmarkUMESIMD<UME::SIMD::SIMD16_64f>(spec, "mandel_umesimd_16_64f.bmp", "SIMD code (16x64f): ", ITERATIONS, stats_scalar_32f);

    return 0;
}
