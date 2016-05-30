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

#ifndef UME_SIMD_VECTOR_EMULATION_H_
#define UME_SIMD_VECTOR_EMULATION_H_

#include "UMEInline.h"
#include "UMEBasicTypes.h"

namespace UME
{
namespace SIMD
{
    //   All functions in this namespace will have one purpose: emulation of single function in different backends.
    //   While scalar emulation is already handling primitive cases, there exists a need for emulation of more
    //   complex functions, and still benefit from vectorization. Functions present in this namespace are non-specialized
    //   implementations that are allowed to use all primitive operations (defined in SCALAR_EMULATION namespace), but
    //   might result in high performance, portable kernels that can be called inside plugin specializations. For the sake of
    //   performance comparison with pure scalar version, none of these functions should be called directly in the interface.
    namespace VECTOR_EMULATION
    {
        // SIN - single precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T sinf(FLOAT_VEC_T const & xx)
        {
            FLOAT_VEC_T s;

            const float ONEOPIO4F = 4.0f / (3.1415927f);

            const float DP1F = (float)0.78515625;
            const float DP2F = (float)2.4187564849853515625e-4;
            const float DP3F = (float)3.77489497744594108e-8;

            INT_VEC_T j;

            /* make argument positive */
            FLOAT_VEC_T x_pos = xx.abs();

            j = INT_VEC_T(ONEOPIO4F * x_pos); /* integer part of x/PIO4 */

            j = (j + 1) & (~1);
            const FLOAT_VEC_T y = FLOAT_VEC_T(j);

            // Extended precision modular arithmetic
            const FLOAT_VEC_T x = ((x_pos - y * DP1F) - y * DP2F) - y * DP3F;

            INT_VEC_T signS = (j & 4);
            j -= 2;

            const INT_VEC_T signC = (j & 4);
            const INT_VEC_T poly = j & 2;

            FLOAT_VEC_T ls, lc;

            FLOAT_VEC_T z = x * x;

            ls = (((-1.9515295891E-4f * z
                + 8.3321608736E-3f) * z
                - 1.6666654611E-1f) * z * x)
                + x;

            lc = ((2.443315711809948E-005f * z
                - 1.388731625493765E-003f) * z
                + 4.166664568298827E-002f) * z * z
                - 0.5f * z + 1.0f;

            //swap
            MASK_T mask_poly = (poly == 0);
            const FLOAT_VEC_T tmp = lc;
            ls.assign(mask_poly, tmp);

            MASK_T mask_signS = (signS != 0);
            ls.assign(mask_signS, -ls);

            MASK_T mask_xx = (xx < 0);
            ls.assign(mask_xx, -ls);

            s = ls;
            return s;
        }

        // SIN - double precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T sind(FLOAT_VEC_T const & xx)
        {
            FLOAT_VEC_T s, c;
            const double ONEOPIO4 = 4.0 / (3.14159265358979323846);

            const double C1sin = 1.58962301576546568060E-10;
            const double C2sin = -2.50507477628578072866E-8;
            const double C3sin = 2.75573136213857245213E-6;
            const double C4sin = -1.98412698295895385996E-4;
            const double C5sin = 8.33333333332211858878E-3;
            const double C6sin = -1.66666666666666307295E-1;

            const double C1cos = -1.13585365213876817300E-11;
            const double C2cos = 2.08757008419747316778E-9;
            const double C3cos = -2.75573141792967388112E-7;
            const double C4cos = 2.48015872888517045348E-5;
            const double C5cos = -1.38888888888730564116E-3;
            const double C6cos = 4.16666666666665929218E-2;

            const double DP1D = 7.853981554508209228515625E-1;
            const double DP2D = 7.94662735614792836714E-9;
            const double DP3D = 3.06161699786838294307E-17;

            INT_VEC_T j;

            FLOAT_VEC_T x = xx.abs();
            j = INT_VEC_T(ONEOPIO4 * x); // always positive, so (int) == std::floor
            j = (j + 1) & (~1);
            const FLOAT_VEC_T y = FLOAT_VEC_T(j);
            // Extended precision modular arithmetic
            x = ((x - y * DP1D) - y * DP2D) - y * DP3D;

            const FLOAT_VEC_T signS = (j & 4);

            j -= 2;

            const FLOAT_VEC_T signC = (j & 4);
            const FLOAT_VEC_T poly = j & 2;

            FLOAT_VEC_T zz = x * x;

            FLOAT_VEC_T px1(C1sin);
            px1 *= zz;
            px1 += C2sin;
            px1 *= zz;
            px1 += C3sin;
            px1 *= zz;
            px1 += C4sin;
            px1 *= zz;
            px1 += C5sin;
            px1 *= zz;
            px1 += C6sin;
            s = x + x * zz *px1;

            FLOAT_VEC_T px2(C1cos);
            px2 *= zz;
            px2 += C2cos;
            px2 *= zz;
            px2 += C3cos;
            px2 *= zz;
            px2 += C4cos;
            px2 *= zz;
            px2 += C5cos;
            px2 *= zz;
            px2 += C6cos;
            c = 1.0 - zz * .5 + zz * zz * px2;

            //swap
            MASK_T maskPoly = (poly == 0);

            const FLOAT_VEC_T tmp = c;
            s.assign(maskPoly, tmp);

            MASK_T maskSignS = (signS != 0);
            s.nega(maskSignS);

            MASK_T maskXX = (xx < 0);
            s.nega(maskXX);

            return s;
        }

        // MSIN - single precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T sinf(MASK_T const & mask, FLOAT_VEC_T const & xx) {
            FLOAT_VEC_T t0 = xx;
            FLOAT_VEC_T t1 = sinf<FLOAT_VEC_T, INT_VEC_T, MASK_T>(xx);
            t0.assign(mask, t1);
            return t0;
        }

        // MSIN - double precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T sind(MASK_T const & mask, FLOAT_VEC_T const & xx) {
            FLOAT_VEC_T t0 = xx;
            FLOAT_VEC_T t1 = sind<FLOAT_VEC_T, INT_VEC_T, MASK_T>(xx);
            t0.assign(mask, t1);
            return t0;
        }

        // COS - single precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T cosf(FLOAT_VEC_T const & xx)
        {
            FLOAT_VEC_T c;

            const float ONEOPIO4F = 4.0f / (3.1415927f);

            const float DP1F = (float)0.78515625;
            const float DP2F = (float)2.4187564849853515625e-4;
            const float DP3F = (float)3.77489497744594108e-8;

            INT_VEC_T j;

            /* make argument positive */
            FLOAT_VEC_T x_pos = xx.abs();

            j = INT_VEC_T(ONEOPIO4F * x_pos); /* integer part of x/PIO4 */

            j = (j + 1) & (~1);
            const FLOAT_VEC_T y = FLOAT_VEC_T(j);

            // Extended precision modular arithmetic
            const FLOAT_VEC_T x = ((x_pos - y * DP1F) - y * DP2F) - y * DP3F;

            INT_VEC_T signS = (j & 4);
            j -= 2;

            const INT_VEC_T signC = (j & 4);
            const INT_VEC_T poly = j & 2;

            FLOAT_VEC_T ls, lc;

            FLOAT_VEC_T z = x * x;

            ls = (((-1.9515295891E-4f * z
                + 8.3321608736E-3f) * z
                - 1.6666654611E-1f) * z * x)
                + x;

            lc = ((2.443315711809948E-005f * z
                - 1.388731625493765E-003f) * z
                + 4.166664568298827E-002f) * z * z
                - 0.5f * z + 1.0f;

            //swap
            MASK_T mask_poly = (poly == 0);
            lc.assign(mask_poly, ls);

            MASK_T mask_signC = (signC == 0);
            lc.assign(mask_signC, -lc);

            c = lc;
            return c;
        }

        // COS - double precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T cosd(FLOAT_VEC_T const & xx)
        {
            FLOAT_VEC_T s, c;

            const double ONEOPIO4 = 4.0 / (3.14159265358979323846);

            const double C1sin = 1.58962301576546568060E-10;
            const double C2sin = -2.50507477628578072866E-8;
            const double C3sin = 2.75573136213857245213E-6;
            const double C4sin = -1.98412698295895385996E-4;
            const double C5sin = 8.33333333332211858878E-3;
            const double C6sin = -1.66666666666666307295E-1;

            const double C1cos = -1.13585365213876817300E-11;
            const double C2cos = 2.08757008419747316778E-9;
            const double C3cos = -2.75573141792967388112E-7;
            const double C4cos = 2.48015872888517045348E-5;
            const double C5cos = -1.38888888888730564116E-3;
            const double C6cos = 4.16666666666665929218E-2;

            const double DP1D = 7.853981554508209228515625E-1;
            const double DP2D = 7.94662735614792836714E-9;
            const double DP3D = 3.06161699786838294307E-17;

            INT_VEC_T j;

            FLOAT_VEC_T x = xx.abs();
            j = INT_VEC_T(ONEOPIO4 * x); // always positive, so (int) == std::floor
            j = (j + 1) & (~1);
            const FLOAT_VEC_T y = FLOAT_VEC_T(j);
            // Extended precision modular arithmetic
            x = ((x - y * DP1D) - y * DP2D) - y * DP3D;

            const FLOAT_VEC_T signS = (j & 4);

            j -= 2;

            const FLOAT_VEC_T signC = (j & 4);
            const FLOAT_VEC_T poly = j & 2;

            FLOAT_VEC_T zz = x * x;

            FLOAT_VEC_T px1(C1sin);
            px1 *= zz;
            px1 += C2sin;
            px1 *= zz;
            px1 += C3sin;
            px1 *= zz;
            px1 += C4sin;
            px1 *= zz;
            px1 += C5sin;
            px1 *= zz;
            px1 += C6sin;
            s = x + x * zz *px1;

            FLOAT_VEC_T px2(C1cos);
            px2 *= zz;
            px2 += C2cos;
            px2 *= zz;
            px2 += C3cos;
            px2 *= zz;
            px2 += C4cos;
            px2 *= zz;
            px2 += C5cos;
            px2 *= zz;
            px2 += C6cos;
            c = 1.0 - zz * .5 + zz * zz * px2;

            //swap
            MASK_T maskPoly = (poly == 0);

            c.assign(maskPoly, s);

            MASK_T maskSignC = (signC == 0);
            c.nega(maskSignC);

            return c;
        }

        // MCOS - single precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T cosf(MASK_T const & mask, FLOAT_VEC_T const & xx) {
            FLOAT_VEC_T t0 = xx;
            FLOAT_VEC_T t1 = cosf<FLOAT_VEC_T, INT_VEC_T, MASK_T>(xx);
            t0.assign(mask, t1);
            return t0;
        }

        // MCOS - double precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T cosd(MASK_T const & mask, FLOAT_VEC_T const & xx) {
            FLOAT_VEC_T t0 = xx;
            FLOAT_VEC_T t1 = cosd<FLOAT_VEC_T, INT_VEC_T, MASK_T>(xx);
            t0.assign(mask, t1);
            return t0;
        }

        // SINCOS - single precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE void sincosf(FLOAT_VEC_T const & xx, FLOAT_VEC_T & s, FLOAT_VEC_T &c)
        {
            const float ONEOPIO4F = 4.0f / (3.1415927f);

            const float DP1F = (float)0.78515625;
            const float DP2F = (float)2.4187564849853515625e-4;
            const float DP3F = (float)3.77489497744594108e-8;

            INT_VEC_T j;

            /* make argument positive */
            FLOAT_VEC_T x_pos = xx.abs();

            j = INT_VEC_T(ONEOPIO4F * x_pos); /* integer part of x/PIO4 */

            j = (j + 1) & (~1);
            const FLOAT_VEC_T y = FLOAT_VEC_T(j);

            // Extended precision modular arithmetic
            const FLOAT_VEC_T x = ((x_pos - y * DP1F) - y * DP2F) - y * DP3F;

            INT_VEC_T signS = (j & 4);
            j -= 2;

            const INT_VEC_T signC = (j & 4);
            const INT_VEC_T poly = j & 2;

            FLOAT_VEC_T ls, lc;

            FLOAT_VEC_T z = x * x;

            ls = (((-1.9515295891E-4f * z
                + 8.3321608736E-3f) * z
                - 1.6666654611E-1f) * z * x)
                + x;

            lc = ((2.443315711809948E-005f * z
                - 1.388731625493765E-003f) * z
                + 4.166664568298827E-002f) * z * z
                - 0.5f * z + 1.0f;

            //swap
            MASK_T mask_poly = (poly == 0);
            const FLOAT_VEC_T tmp = lc;
            lc.assign(mask_poly, ls);
            ls.assign(mask_poly, tmp);

            MASK_T mask_signC = (signC == 0);
            lc.assign(mask_signC, -lc);

            MASK_T mask_signS = (signS != 0);
            ls.assign(mask_signS, -ls);

            MASK_T mask_xx = (xx < 0);
            ls.assign(mask_xx, -ls);

            c = lc;
            s = ls;
        }

        // SINCOS - double precision version
        template<typename FLOAT_VEC_T, typename INT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE void sincosd(FLOAT_VEC_T const & xx, FLOAT_VEC_T & s, FLOAT_VEC_T & c) {
            const double ONEOPIO4 = 4.0 / (3.14159265358979323846);

            const double C1sin = 1.58962301576546568060E-10;
            const double C2sin = -2.50507477628578072866E-8;
            const double C3sin = 2.75573136213857245213E-6;
            const double C4sin = -1.98412698295895385996E-4;
            const double C5sin = 8.33333333332211858878E-3;
            const double C6sin = -1.66666666666666307295E-1;

            const double C1cos = -1.13585365213876817300E-11;
            const double C2cos = 2.08757008419747316778E-9;
            const double C3cos = -2.75573141792967388112E-7;
            const double C4cos = 2.48015872888517045348E-5;
            const double C5cos = -1.38888888888730564116E-3;
            const double C6cos = 4.16666666666665929218E-2;

            const double DP1D = 7.853981554508209228515625E-1;
            const double DP2D = 7.94662735614792836714E-9;
            const double DP3D = 3.06161699786838294307E-17;

            INT_VEC_T j;

            FLOAT_VEC_T x = xx.abs();
            j = INT_VEC_T(ONEOPIO4 * x); // always positive, so (int) == std::floor
            j = (j + 1) & (~1);
            const FLOAT_VEC_T y = FLOAT_VEC_T(j);
            // Extended precision modular arithmetic
            x = ((x - y * DP1D) - y * DP2D) - y * DP3D;

            const FLOAT_VEC_T signS = (j & 4);

            j -= 2;

            const FLOAT_VEC_T signC = (j & 4);
            const FLOAT_VEC_T poly = j & 2;

            FLOAT_VEC_T zz = x * x;

            FLOAT_VEC_T px1(C1sin);
            px1 *= zz;
            px1 += C2sin;
            px1 *= zz;
            px1 += C3sin;
            px1 *= zz;
            px1 += C4sin;
            px1 *= zz;
            px1 += C5sin;
            px1 *= zz;
            px1 += C6sin;
            s = x + x * zz *px1;

            FLOAT_VEC_T px2(C1cos);
            px2 *= zz;
            px2 += C2cos;
            px2 *= zz;
            px2 += C3cos;
            px2 *= zz;
            px2 += C4cos;
            px2 *= zz;
            px2 += C5cos;
            px2 *= zz;
            px2 += C6cos;
            c = 1.0 - zz * .5 + zz * zz * px2;

            //swap
            MASK_T maskPoly = (poly == 0);

            const FLOAT_VEC_T tmp = c;
            c.assign(maskPoly, s);
            s.assign(maskPoly, tmp);

            MASK_T maskSignC = (signC == 0);
            c.nega(maskSignC);

            MASK_T maskSignS = (signS != 0);
            s.nega(maskSignS);

            MASK_T maskXX = (xx < 0);
            s.nega(maskXX);
        }

        // MSINCOS - single precision version
        template<typename FLOAT_VEC_T, typename MASK_TYPE>
        UME_FORCE_INLINE void sincosf(MASK_TYPE const & mask, FLOAT_VEC_T const & xx, FLOAT_VEC_T & s, FLOAT_VEC_T & c) {
            FLOAT_VEC_T masked_s, masked_c;
            s = xx;
            c = xx;
            sincosf<FLOAT_VEC_T>(xx, masked_s, masked_c);
            s.assign(mask, masked_s);
            c.assign(mask, masked_c);
        }

        // MSINCOS - double precision version
        template<typename FLOAT_VEC_T, typename MASK_TYPE>
        UME_FORCE_INLINE void sincosd(MASK_TYPE const & mask, FLOAT_VEC_T const & xx, FLOAT_VEC_T & s, FLOAT_VEC_T & c) {
            FLOAT_VEC_T masked_s, masked_c;
            s = xx;
            c = xx;
            sincosd<FLOAT_VEC_T>(xx, masked_s, masked_c);
            s.assign(mask, masked_s);
            c.assign(mask, masked_c);
        }
    }
}
}

#endif