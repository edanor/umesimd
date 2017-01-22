// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
//
// Authors: Przemyslaw Karpinski, Mathieu Gravey
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


#ifndef UME_MANDEL_OPENMP_H_
#define UME_MANDEL_OPENMP_H_

#define N 16

void mandel_openmp(
    float x1, 
    float y1, 
    float x2, 
    float y2, 
    uint32_t width, 
    uint32_t height, 
    uint32_t maxIters, 
    uint16_t * image)
{
    float dx = (x2 - x1) / width, dy = (y2 - y1) / height;
    for (uint32_t index = 0; index < height*width; index+=N)
    {   
        uint32_t i[N];
        uint32_t j[N];
        float cx[N];
        float cy[N];
        float x[N];
        float y[N];
        uint32_t val[N];
        #pragma omp simd
        for (uint32_t subIndex = 0; subIndex < N; ++subIndex)
        {
            i[subIndex]=(index+subIndex)%width;
            j[subIndex]=(index+subIndex)/width;
            cx[subIndex] = x1 + dx*i[subIndex];
            cy[subIndex] = y1 + dy*j[subIndex];
            x[subIndex] = cx[subIndex];
            y[subIndex] = cy[subIndex];
            val[subIndex]=0;
        }

        unsigned char todo=16;
        uint32_t count = 1;
        uint32_t stepSize=16;
        while ((todo>2) && (count < maxIters))
        {
            todo=false;
            #pragma omp simd reduction(+:todo)
            for (uint32_t subIndex = 0; subIndex < N; ++subIndex)
            {
                bool localTodo;
                #pragma unroll (4)
                for (uint32_t k = 0; k < stepSize ; ++k)
                {
                    float x2 = x[subIndex] * x[subIndex];
                    float y2 = y[subIndex] * y[subIndex];
                    localTodo = (x2 + y2 < 4.0f);
                    float xy = x[subIndex]*y[subIndex];
                    x[subIndex] +=(localTodo) * (x2 - y2 + cx[subIndex] - x[subIndex]);
                    y[subIndex] +=(localTodo) * (2 * xy + cy[subIndex] - y[subIndex]);
                    val[subIndex]=val[subIndex]+localTodo;
                }
                todo+=localTodo;
            }
            count+=stepSize;
        }
        uint32_t stopCount=count;
        for (uint32_t subIndex = 0; subIndex < N; ++subIndex)
        {
           float x2 = x[subIndex] * x[subIndex];
           float y2 = y[subIndex] * y[subIndex];
           bool localTodo = (x2 + y2 < 4.0f);
           count=stopCount;
           while ((localTodo) && (count < maxIters))
           {
            float x2 = x[subIndex] * x[subIndex];
            float y2 = y[subIndex] * y[subIndex];
            if (!(x2 + y2 < 4.0f)) break;
            float xy = x[subIndex]*y[subIndex];
            x[subIndex] +=(x2 - y2 + cx[subIndex]);
            y[subIndex] += (2 * xy + cy[subIndex]);
            val[subIndex]=val[subIndex]+1;
            count++;
        }
    }


        #pragma omp simd
        for (uint32_t subIndex = 0; subIndex < N; ++subIndex)
        {
            image[index+subIndex] = val[subIndex];
        }
    }
}

void benchmarkOpenMP(int width,
  int height,
  int depth,
  char * filename, 
  char * resultPrefix, 
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

        start = __rdtsc();
        mandel_openmp(0.29768f, 0.48364f, 0.29778f, 0.48354f, width, height, depth, raw_image);
        end = __rdtsc();

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

#undef N

#endif