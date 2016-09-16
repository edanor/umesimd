// 
// This piece of code comes from https://github.com/skeeto/mandel-simd .
// this code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
// 
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

#include <cmath>
#include "mandelbrot.h"

void mandel_basic_32f(unsigned char *image, const struct spec *s)
{
    float xdiff = s->xlim[1] - s->xlim[0];
    float ydiff = s->ylim[1] - s->ylim[0];
    float iter_scale = 1.0f / s->iterations;
    float depth_scale = float(s->depth - 1);
    // #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x++) {
            float cr = x * xdiff / s->width + s->xlim[0];
            float ci = y * ydiff / s->height + s->ylim[0];
            float zr = cr;
            float zi = ci;
            int k = 0;
            float mk = 0.0f;
            while (++k < s->iterations) {
                float zr1 = zr * zr - zi * zi + cr;
                float zi1 = zr * zi + zr * zi + ci;
                zr = zr1;
                zi = zi1;
                mk += 1.0f;
                if (zr * zr + zi * zi >= 4.0f)
                    break;
            }
            mk *= iter_scale;
            mk = sqrtf(mk);
            mk *= depth_scale;
            int pixel = int(mk);
            image[y * s->width * 3 + x * 3 + 0] = pixel;
            image[y * s->width * 3 + x * 3 + 1] = pixel;
            image[y * s->width * 3 + x * 3 + 2] = pixel;
        }
    }
}


void mandel_basic_64f(unsigned char *image, const struct spec *s)
{
    double xdiff = s->xlim[1] - s->xlim[0];
    double ydiff = s->ylim[1] - s->ylim[0];
    double iter_scale = 1.0 / s->iterations;
    double depth_scale = double(s->depth - 1);
    // #pragma omp parallel for schedule(dynamic, 1)
    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x++) {
            double cr = x * xdiff / s->width + s->xlim[0];
            double ci = y * ydiff / s->height + s->ylim[0];
            double zr = cr;
            double zi = ci;
            long k = 0;
            double mk = 0.0;
            while (++k < s->iterations) {
                double zr1 = zr * zr - zi * zi + cr;
                double zi1 = zr * zi + zr * zi + ci;
                zr = zr1;
                zi = zi1;
                mk += 1.0;
                if (zr * zr + zi * zi >= 4.0)
                    break;
            }
            mk *= iter_scale;
            mk = sqrt(mk);
            mk *= depth_scale;
            int pixel = int(mk);
            image[y * s->width * 3 + x * 3 + 0] = pixel;
            image[y * s->width * 3 + x * 3 + 1] = pixel;
            image[y * s->width * 3 + x * 3 + 2] = pixel;
        }
    }
}
