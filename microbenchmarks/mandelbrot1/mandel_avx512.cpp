// 
// This piece of code comes from https://github.com/skeeto/mandel-simd .
// this code is not a part of UME::SIMD library code and is used purely for
// performance measurement reference.
// 
// Modifications have been made to original files to fit them for benchmarking
// of UME::SIMD.

#include <immintrin.h>
#include "mandelbrot.h"

#if defined __AVX512F__

void
mandel_avx512(unsigned char *image, const struct spec *s)
{
    __m512 xmin = _mm512_set1_ps(s->xlim[0]);
    __m512 ymin = _mm512_set1_ps(s->ylim[0]);
    __m512 xscale = _mm512_set1_ps((s->xlim[1] - s->xlim[0]) / s->width);
    __m512 yscale = _mm512_set1_ps((s->ylim[1] - s->ylim[0]) / s->height);
    __m512 threshold = _mm512_set1_ps(4);
    __m512 one = _mm512_set1_ps(1);
    __m512 iter_scale = _mm512_set1_ps(1.0f / s->iterations);
    __m512 depth_scale = _mm512_set1_ps(float(s->depth - 1));

    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x += 16) {
            __m512 mx = _mm512_set_ps(float(x + 15), float(x + 14), float(x + 13), float(x + 12),
                                      float(x + 11), float(x + 10), float(x + 9), float(x + 8),
                                      float(x + 7), float(x + 6), float(x + 5), float(x + 4),
                                      float(x + 3), float(x + 2), float(x + 1), float(x + 0));
            __m512 my = _mm512_set1_ps(float(y));
            __m512 cr = _mm512_add_ps(_mm512_mul_ps(mx, xscale), xmin);
            __m512 ci = _mm512_add_ps(_mm512_mul_ps(my, yscale), ymin);
            __m512 zr = cr;
            __m512 zi = ci;
            int k = 1;
            __m512 mk = _mm512_set1_ps(float(k));
            while (++k < s->iterations) {
                /* Compute z1 from z0 */
                __m512 zr2 = _mm512_mul_ps(zr, zr);
                __m512 zi2 = _mm512_mul_ps(zi, zi);
                __m512 zrzi = _mm512_mul_ps(zr, zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + ci */
                zr = _mm512_add_ps(_mm512_sub_ps(zr2, zi2), cr);
                zi = _mm512_add_ps(_mm512_add_ps(zrzi, zrzi), ci);

                /* Increment k */
                zr2 = _mm512_mul_ps(zr, zr);
                zi2 = _mm512_mul_ps(zi, zi);
                __m512 mag2 = _mm512_add_ps(zr2, zi2);
                __mmask16 mask = _mm512_cmp_ps_mask(mag2, threshold, _CMP_LT_OQ);
                mk = _mm512_mask_add_ps(mk, mask, one, mk);

                /* Early bailout? */
                if (mask == 0)
                    break;
            }
            mk = _mm512_mul_ps(mk, iter_scale);
            mk = _mm512_sqrt_ps(mk);
            mk = _mm512_mul_ps(mk, depth_scale);
            __m512i pixels = _mm512_cvtps_epi32(mk);
            unsigned char *dst = image + y * s->width * 3 + x * 3;
            //unsigned char *src = (unsigned char *)&pixels;
            alignas(64) unsigned char src[64];
            _mm512_store_si512(src, pixels);
            for (int i = 0; i < 16; i++) {
                dst[i * 3 + 0] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4];
                dst[i * 3 + 2] = src[i * 4];
            }
        }
    }
}

#endif
