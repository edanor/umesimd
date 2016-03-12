// 
// This piece of code comes from https://github.com/skeeto/mandel-simd .
// this code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
// 
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

#include <immintrin.h>
#include "mandelbrot.h"



#if defined __AVX__

void
mandel_avx(unsigned char *image, const struct spec *s)
{
    __m256 xmin = _mm256_set1_ps(s->xlim[0]);
    __m256 ymin = _mm256_set1_ps(s->ylim[0]);
    __m256 xscale = _mm256_set1_ps((s->xlim[1] - s->xlim[0]) / s->width);
    __m256 yscale = _mm256_set1_ps((s->ylim[1] - s->ylim[0]) / s->height);
    __m256 threshold = _mm256_set1_ps(4);
    __m256 one = _mm256_set1_ps(1);
    __m256 iter_scale = _mm256_set1_ps(1.0f / s->iterations);
    __m256 depth_scale = _mm256_set1_ps(float(s->depth - 1));

    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x += 8) {
            __m256 mx = _mm256_set_ps(float(x + 7), float(x + 6), float(x + 5), float(x + 4),
                                      float(x + 3), float(x + 2), float(x + 1), float(x + 0));
            __m256 my = _mm256_set1_ps(float(y));
            __m256 cr = _mm256_add_ps(_mm256_mul_ps(mx, xscale), xmin);
            __m256 ci = _mm256_add_ps(_mm256_mul_ps(my, yscale), ymin);
            __m256 zr = cr;
            __m256 zi = ci;
            int k = 1;
            __m256 mk = _mm256_set1_ps(float(k));
            while (++k < s->iterations) {
                /* Compute z1 from z0 */
                __m256 zr2 = _mm256_mul_ps(zr, zr);
                __m256 zi2 = _mm256_mul_ps(zi, zi);
                __m256 zrzi = _mm256_mul_ps(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm256_add_ps(_mm256_sub_ps(zr2, zi2), cr);
                zi = _mm256_add_ps(_mm256_add_ps(zrzi, zrzi), ci);

                /* Increment k */
                zr2 = _mm256_mul_ps(zr, zr);
                zi2 = _mm256_mul_ps(zi, zi);
                __m256 mag2 = _mm256_add_ps(zr2, zi2);
                __m256 mask = _mm256_cmp_ps(mag2, threshold, _CMP_LT_OS);
                mk = _mm256_add_ps(_mm256_and_ps(mask, one), mk);

                /* Early bailout? */
                if (_mm256_testz_ps(mask, _mm256_set1_ps(-1)))
                    break;
            }
            mk = _mm256_mul_ps(mk, iter_scale);
            mk = _mm256_sqrt_ps(mk);
            mk = _mm256_mul_ps(mk, depth_scale);
            __m256i pixels = _mm256_cvtps_epi32(mk);
            unsigned char *dst = image + y * s->width * 3 + x * 3;
            //unsigned char *src = (unsigned char *)&pixels;
            alignas(32) unsigned char src[32];
            _mm256_store_si256((__m256i*)src, pixels);
            for (int i = 0; i < 8; i++) {
                dst[i * 3 + 0] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4];
                dst[i * 3 + 2] = src[i * 4];
            }
        }
    }
}

#endif
