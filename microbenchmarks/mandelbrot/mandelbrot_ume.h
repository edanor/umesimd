#ifndef MANDELBROT_UME_H_
#define MANDELBROT_UME_H_

#include "mandelbrot.h"
#include "../../UMESimd.h"

template<typename VEC_T>
void mandel_umesimd(unsigned char *image, const struct spec *s)
{
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T  SCALAR_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T    MASK_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::INT_VEC_T INT_VEC_T;

    VEC_T xmin(s->xlim[0]);
    VEC_T ymin(s->ylim[0]);
    VEC_T xscale((s->xlim[1] - s->xlim[0]) / s->width);
    VEC_T yscale((s->ylim[1] - s->ylim[0]) / s->height);
    VEC_T threshold(4.0f);
    VEC_T one(1.0f);
    VEC_T iter_scale(1.0f / s->iterations);
    VEC_T depth_scale(SCALAR_T(s->depth - 1));

    // Initialize vector of incremental values: 0, 1, 2, 3, ... up to VEC_LEN
    SCALAR_T initializer1[VEC_T::length()];
    for (unsigned int i = 0; i < VEC_T::length(); i++) initializer1[i] = SCALAR_T(i);
    VEC_T initial_increments(initializer1);

    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x += VEC_T::length()) {
            VEC_T mx = initial_increments + SCALAR_T(x);
            VEC_T my = VEC_T(float(y));
            VEC_T cr = mx.fmuladd(xscale, xmin);
            VEC_T ci = my.fmuladd(yscale, ymin);
            VEC_T zr = cr;
            VEC_T zi = ci;
            int k = 1;
            VEC_T mk = VEC_T(SCALAR_T(k));
            while (++k < s->iterations) {
                /* Compute z1 from z0 */
                VEC_T zr2 = zr * zr;
                VEC_T zi2 = zi * zi;
                VEC_T zrzi = zr * zi;
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + c1 */
                zr = zr2 - zi2 + cr;
                zi = zrzi + zrzi + ci;

                /* Increment k */
                zr2 = zr * zr;
                zi2 = zi * zi;
                VEC_T mag2 = zr2 + zi2;
                MASK_T mask = mag2 < threshold;
                mk = mk.add(mask, one);

                /* Early bailout? */
                if (mask.hlor() == 0) break;
            }
            mk = mk * iter_scale;
            mk = mk.sqrt();
            mk = mk * depth_scale;
            INT_VEC_T pixels = INT_VEC_T(mk.round());
            unsigned char *dst = image + y * s->width * 3 + x * 3;
            unsigned char src[VEC_T::length()*4];
            pixels.store((int32_t*)src);
            for (unsigned int i = 0; i < VEC_T::length(); i++) {
                dst[i * 3 + 0] = src[i * 4];
                dst[i * 3 + 1] = src[i * 4];
                dst[i * 3 + 2] = src[i * 4];
            }
        }
    }
}

#endif
