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
#ifndef UME_MANDEL_OPENMP_H_
#define UME_MANDEL_OPENMP_H_

void mandel_openmp(
    float x1, 
    float y1, 
    float x2, 
    float y2, 
    int width, 
    int height, 
    int maxIters, 
    uint16_t * image)
{
    // Code for openMP implementation. MandelbrotCPU2 (mandel_intel.h) is a good starting point.
}

void benchmarkOpenMP(int width,
                      int height,
                      int depth,
                      std::string const & filename, 
                      std::string const & resultPrefix, 
                      int iterations,
                      TimingStatistics & reference)
{
    TimingStatistics stats;

    UME::Bitmap bmp(width, height, UME::PIXEL_TYPE_RGB);
    uint8_t* image = bmp.GetRasterData();

    unsigned short *raw_image;

    // Using 64B alignment in this case will not incure visible memory penalty, but will guarantee proper alignment.
    raw_image = (unsigned short *)UME::DynamicMemory::AlignedMalloc(width*height*sizeof(unsigned short), 64);

    for (int i = 0; i < iterations; i++) {
        TIMING_RES start, end;

        memset(raw_image, 0, width*height *sizeof(uint16_t));

        start = get_timestamp();
        mandel_openmp(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, depth, raw_image);
        end = get_timestamp();

        stats.update(end - start);

        // Rewrite algorithm output to BMP format
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int value = raw_image[h*width + w];
                Color c = getColor(value);
                image[3 * (h*width + w) + 0] = c.r;
                image[3 * (h*width + w) + 1] = c.g;
                image[3 * (h*width + w) + 2] = c.b;
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

#endif