// 
// This piece of code comes from https://github.com/skeeto/mandel-simd .
// this code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
// 
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

#ifndef MANDELBROT_H_
#define MANDELBROT_H_

struct spec {
    /* Image Specification */
    int width;
    int height;
    int depth;
    /* Fractal Specification */
    float xlim[2];
    float ylim[2];
    int iterations;
};

#endif
