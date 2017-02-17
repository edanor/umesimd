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

#ifndef MANDELBROT_UME_H_
#define MANDELBROT_UME_H_

#include "mandelbrot.h"
#include <ume/simd>

template<typename VEC_T>
void mandel_umesimd(unsigned char *image, const struct spec *s)
{
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T  SCALAR_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_INT_T SCALAR_INT_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T    MASK_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::INT_VEC_T INT_VEC_T;

    const unsigned int VEC_LEN = VEC_T::length();
    VEC_T xmin(s->xlim[0]);
    VEC_T ymin(s->ylim[0]);
    VEC_T xscale((s->xlim[1] - s->xlim[0]) / s->width);
    VEC_T yscale((s->ylim[1] - s->ylim[0]) / s->height);
    VEC_T threshold(4.0f);
    VEC_T one(1.0f);
    VEC_T iter_scale(1.0f / s->iterations);
    VEC_T depth_scale(SCALAR_T(s->depth - 1));

    // Initialize vector of incremental values: 0, 1, 2, 3, ... up to VEC_LEN
    SCALAR_T initializer1[VEC_LEN];
    for (unsigned int i = 0; i < VEC_LEN; i++) initializer1[i] = SCALAR_T(i);
    VEC_T initial_increments(initializer1);

    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x += VEC_LEN) {
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
            alignas(VEC_T::alignment())unsigned char src[VEC_LEN * sizeof(SCALAR_INT_T)];
            pixels.storea((SCALAR_INT_T*)src);
            for (unsigned int i = 0; i < VEC_LEN; i++) {
                dst[i * 3 + 0] = src[i * sizeof(SCALAR_INT_T)];
                dst[i * 3 + 1] = src[i * sizeof(SCALAR_INT_T)];
                dst[i * 3 + 2] = src[i * sizeof(SCALAR_INT_T)];
            }
        }
    }
}

template<typename VEC_T>
void mandel_umesimd_MFI(unsigned char *image, const struct spec *s)
{
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_T  SCALAR_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::SCALAR_INT_T SCALAR_INT_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::MASK_T    MASK_T;
    typedef typename UME::SIMD::SIMDTraits<VEC_T>::INT_VEC_T INT_VEC_T;

    const unsigned int VEC_LEN = VEC_T::length();
    VEC_T xmin(s->xlim[0]);
    VEC_T ymin(s->ylim[0]);
    VEC_T xscale((s->xlim[1] - s->xlim[0]) / s->width);
    VEC_T yscale((s->ylim[1] - s->ylim[0]) / s->height);
    VEC_T threshold(4.0f);
    VEC_T one(1.0f);
    VEC_T iter_scale(1.0f / s->iterations);
    VEC_T depth_scale(SCALAR_T(s->depth - 1));

    // Initialize vector of incremental values: 0, 1, 2, 3, ... up to VEC_LEN
    SCALAR_T initializer1[VEC_LEN];
    for (unsigned int i = 0; i < VEC_LEN; i++) initializer1[i] = SCALAR_T(i);
    VEC_T initial_increments(initializer1);

    for (int y = 0; y < s->height; y++) {
        for (int x = 0; x < s->width; x += VEC_LEN) {
            VEC_T mx = initial_increments.add(SCALAR_T(x));
            VEC_T my = VEC_T(float(y));
            VEC_T cr = mx.fmuladd(xscale, xmin);
            VEC_T ci = my.fmuladd(yscale, ymin);
            VEC_T zr = cr;
            VEC_T zi = ci;
            int k = 1;
            VEC_T mk = VEC_T(SCALAR_T(k));
            while (++k < s->iterations) {
                /* Compute z1 from z0 */
                VEC_T zr2 = zr.mul(zr);
                VEC_T zi2 = zi.mul(zi);
                VEC_T zrzi = zr.mul(zi);
                /* zr1 = zr0 * zr0 - zi0 * zi0 + cr */
                /* zi1 = zr0 * zi0 + zr0 * zi0 + c1 */
                zr.assign((zr2.sub(zi2)).add(cr));
                zi.assign((zrzi.add(zrzi)).add(ci));

                /* Increment k */
                zr2.assign(zr.mul(zr));
                zi2.assign(zi.mul(zi));
                VEC_T mag2 = zr2.add(zi2);
                MASK_T mask = mag2.cmplt(threshold);
                mk.assign(mk.add(mask, one));
                /* Early bailout? */
                if (mask.hlor() == 0) break;
            }
            mk.mula(iter_scale);
            mk.sqrta();
            mk.mula(depth_scale);
            INT_VEC_T pixels = INT_VEC_T(mk.round());
            unsigned char *dst = image + y * s->width * 3 + x * 3;
            alignas(VEC_T::alignment()) unsigned char src[VEC_LEN * sizeof(SCALAR_INT_T)];
            pixels.storea((SCALAR_INT_T*)src);
            for (unsigned int i = 0; i < VEC_LEN; i++) {
                dst[i * 3 + 0] = src[i * sizeof(SCALAR_INT_T)];
                dst[i * 3 + 1] = src[i * sizeof(SCALAR_INT_T)];
                dst[i * 3 + 2] = src[i * sizeof(SCALAR_INT_T)];
            }
        }
    }
}

#endif
