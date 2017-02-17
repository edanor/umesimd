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
        // EXP - single precision version
        template<typename FLOAT_VEC_T, typename UINT_VEC_T>
        UME_FORCE_INLINE FLOAT_VEC_T expf(FLOAT_VEC_T const & initial_x) {
            const float MAXLOGF = 88.72283905206835f;
            const float MINLOGF = -88.f;

            const float C1F =   0.693359375f;
            const float C2F =  -2.12194440e-4f;

            const float PX1expf = 1.9875691500E-4f;
            const float PX2expf =1.3981999507E-3f;
            const float PX3expf =8.3334519073E-3f;
            const float PX4expf =4.1665795894E-2f;
            const float PX5expf =1.6666665459E-1f;
            const float PX6expf =5.0000001201E-1f;

            const float LOG2EF = 1.44269504088896341f;

            FLOAT_VEC_T x = initial_x;
            FLOAT_VEC_T z = (LOG2EF * x +0.5f ).floor(); /* floor() truncates toward -infinity. */

            x -= z * C1F;
            x -= z * C2F;
            const UINT_VEC_T n = UINT_VEC_T ( z );

            const FLOAT_VEC_T x2 = x * x;

            z = x*PX1expf;
            z += PX2expf;
            z *= x;
            z += PX3expf;
            z *= x;
            z += PX4expf;
            z *= x;
            z += PX5expf;
            z *= x;
            z += PX6expf;
            z *= x2;
            z += x + 1.0f;

            /* multiply by power of 2 */
            alignas(FLOAT_VEC_T::alignment()) float raw[FLOAT_VEC_T::length()];
            ((n + 0x7f) << 23).store((uint32_t*)&raw[0]);
            FLOAT_VEC_T z_0(raw);
            z *= z_0;

            z[initial_x > MAXLOGF] = std::numeric_limits<float>::infinity();
            z[initial_x < MINLOGF] = 0.0f;

            return z;
        }
        // EXP - double precision version

        template<typename FLOAT_VEC_T, typename UINT_VEC_T>
        UME_FORCE_INLINE FLOAT_VEC_T expd(FLOAT_VEC_T const & initial_x) {
            const double EXP_LIMIT = 708;

            const double PX1exp = 1.26177193074810590878E-4;
            const double PX2exp = 3.02994407707441961300E-2;
            const double PX3exp = 9.99999999999999999910E-1;
            const double QX1exp = 3.00198505138664455042E-6;
            const double QX2exp = 2.52448340349684104192E-3;
            const double QX3exp = 2.27265548208155028766E-1;
            const double QX4exp = 2.00000000000000000009E0;

            const double LOG2E = 1.4426950408889634073599; // 1/log(2)

            FLOAT_VEC_T x = initial_x;
            FLOAT_VEC_T px = ( LOG2E * x +0.5 ).floor();

            x -= px * 6.93145751953125E-1;
            x -= px * 1.42860682030941723212E-6;

            const UINT_VEC_T n = UINT_VEC_T ( x );

            const FLOAT_VEC_T xx = x * x;

            // px = x * P(x**2).
            px = PX1exp;
            px *= xx;
            px += PX2exp;
            px *= xx;
            px += PX3exp;
            px *= x;

            // Evaluate Q(x**2).
            FLOAT_VEC_T qx(QX1exp);
            qx *= xx;
            qx += QX2exp;
            qx *= xx;
            qx += QX3exp;
            qx *= xx;
            qx += QX4exp;

            // e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
            x = px / (qx - px);
            x = 1.0 + 2.0 * x;

            /* multiply by power of 2 */
            alignas(FLOAT_VEC_T::alignment()) double raw[FLOAT_VEC_T::length()];
            ((n + 1023) << 52).store((uint64_t*)&raw[0]);
            UINT_VEC_T x_0((uint64_t*)raw);

            x *= x_0;

            x[initial_x > EXP_LIMIT] = std::numeric_limits<double>::infinity();
            x[initial_x < -EXP_LIMIT] =0.;

            return x;
        }
        // MEXP - single precision version
        template<typename FLOAT_VEC_T, typename UINT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T expf(MASK_T const & mask, FLOAT_VEC_T const & initial_x) {
            FLOAT_VEC_T t0 = initial_x;
            FLOAT_VEC_T t1 = expf<FLOAT_VEC_T, UINT_VEC_T>(initial_x);
            t0.assign(mask, t1);
            return t0;
        }
        // MEXP - double precision version
        template<typename FLOAT_VEC_T, typename UINT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T expd(MASK_T const & mask, FLOAT_VEC_T const & initial_x) {
            FLOAT_VEC_T t0 = initial_x;
            FLOAT_VEC_T t1 = expd<FLOAT_VEC_T, UINT_VEC_T>(initial_x);
            t0.assign(mask, t1);
            return t0;
        }
        // LOG - single precision
        template<typename FLOAT_VEC_T, typename UINT_VEC_T>
        UME_FORCE_INLINE FLOAT_VEC_T logf(FLOAT_VEC_T const & initial_x) {
            const float MAXNUMF = 3.4028234663852885981170418348451692544e38f;
            const float LOGF_UPPER_LIMIT = MAXNUMF;
            const float LOGF_LOWER_LIMIT = 0;
            const float PX1logf = 7.0376836292E-2f;
            const float PX2logf	= -1.1514610310E-1f;
            const float PX3logf	= 1.1676998740E-1f;
            const float PX4logf	= -1.2420140846E-1f;
            const float PX5logf	= 1.4249322787E-1f;
            const float PX6logf	= -1.6668057665E-1f;
            const float PX7logf	= 2.0000714765E-1f;
            const float PX8logf	= -2.4999993993E-1f;
            const float PX9logf	= 3.3333331174E-1f;
            const float SQRTHF = 0.707106781186547524f;

            FLOAT_VEC_T fe;
            //x = details::getMantExponentf( x, fe);
            ///////////////
            alignas(FLOAT_VEC_T::alignment()) float raw[FLOAT_VEC_T::length()];
            initial_x.storea(raw);
            UINT_VEC_T n;
            n.loada((uint32_t*)&raw[0]);
            UINT_VEC_T e = (n >> 23)-127;
            fe = FLOAT_VEC_T(e);

            // fractional part
            const uint32_t p05f = 0x3f000000; // //sp2uint32(0.5);
            n.banda(0x807fffff);// ~0x7f800000;
            n |= p05f;

            n.storea((uint32_t*)&raw[0]);
            FLOAT_VEC_T x;
            x.loada(raw);
            //////////////

            fe.postinc(x > SQRTHF);
            x.adda(x <= SQRTHF, x);
            x -= 1.0f;

            const FLOAT_VEC_T x2 = x*x;

            //FLOAT_VEC_T res = details::get_log_poly(x);
            FLOAT_VEC_T res = x*PX1logf;
            res += PX2logf;
            res *= x;
            res += PX3logf;
            res *= x;
            res += PX4logf;
            res *= x;
            res += PX5logf;
            res *= x;
            res += PX6logf;
            res *= x;
            res += PX7logf;
            res *= x;
            res += PX8logf;
            res *= x;
            res += PX9logf;

            res *= x2*x;

            res += -2.12194440e-4f * fe;
            res +=  -0.5f * x2;

            res= x + res;

            res += 0.693359375f * fe;

            res[initial_x > LOGF_UPPER_LIMIT] = std::numeric_limits<float>::infinity();
            res[initial_x < LOGF_LOWER_LIMIT] = -std::numeric_limits<float>::quiet_NaN();

            return res;
        }
        // LOG - double precision
        template<typename FLOAT_VEC_T, typename UINT_VEC_T>
        UME_FORCE_INLINE FLOAT_VEC_T logd(FLOAT_VEC_T const & initial_x) {
            const double LOG_UPPER_LIMIT = 1e307;
            const double LOG_LOWER_LIMIT = 0;

            const double SQRTH = 0.70710678118654752440;

            /* separate mantissa from exponent */
            FLOAT_VEC_T fe;
            //x = details::getMantExponent(x,fe);
            ///////
            alignas(FLOAT_VEC_T::alignment()) double raw[FLOAT_VEC_T::length()];
            initial_x.storea(raw);
            UINT_VEC_T n;
            n.loada((uint64_t*)&raw[0]);

            // Shift to the right up to the beginning of the exponent.
            // Then with a mask, cut off the sign bit
            UINT_VEC_T le = (n >> 52);

            // chop the head of the number: an int contains more than 11 bits (32)
            UINT_VEC_T e = le;
            fe = e-1023;

            // This puts to 11 zeroes the exponent
            n.banda(0x800FFFFFFFFFFFFFULL);
            // build a mask which is 0.5, i.e. an exponent equal to 1022
            // which means *2, see the above +1.
            const uint64_t p05 = 0x3FE0000000000000ULL; //dp2uint64(0.5);
            n |= p05;

            n.storea((uint64_t*)&raw[0]);
            FLOAT_VEC_T x;
            x.loada(&raw[0]);

            ///////
            // blending
            fe.postinc(x > SQRTH);
            x.adda(x <= SQRTH, x);
            x -= 1.0;

            /* rational form */
            //FLOAT_VEC_T px =  details::get_log_px(x);
            //////
            const double PX1log = 1.01875663804580931796E-4;
            const double PX2log = 4.97494994976747001425E-1;
            const double PX3log = 4.70579119878881725854E0;
            const double PX4log = 1.44989225341610930846E1;
            const double PX5log = 1.79368678507819816313E1;
            const double PX6log = 7.70838733755885391666E0;

            FLOAT_VEC_T px(PX1log);
            px *= x;
            px += PX2log;
            px *= x;
            px += PX3log;
            px *= x;
            px += PX4log;
            px *= x;
            px += PX5log;
            px *= x;
            px += PX6log;

            /////

            //for the final formula
            const FLOAT_VEC_T x2 = x*x;
            px *= x;
            px *= x2;

            //const FLOAT_VEC_T qx = details::get_log_qx(x);
            //////

            const double QX1log = 1.12873587189167450590E1;
            const double QX2log = 4.52279145837532221105E1;
            const double QX3log = 8.29875266912776603211E1;
            const double QX4log = 7.11544750618563894466E1;
            const double QX5log = 2.31251620126765340583E1;

            FLOAT_VEC_T qx = x;
            qx += QX1log;
            qx *=x;
            qx += QX2log;
            qx *=x;
            qx += QX3log;
            qx *=x;
            qx += QX4log;
            qx *=x;
            qx += QX5log;
            /////

            FLOAT_VEC_T res = px / qx ;

            res -= fe * 2.121944400546905827679e-4;
            res -= 0.5 * x2  ;

            res = x + res;
            res += fe * 0.693359375;

            res[initial_x > LOG_UPPER_LIMIT] = std::numeric_limits<double>::infinity();
            res[initial_x < LOG_LOWER_LIMIT] = -std::numeric_limits<double>::quiet_NaN();

            return res;

        }
        // MLOG - single precision version
        template<typename FLOAT_VEC_T, typename UINT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T logf(MASK_T const & mask, FLOAT_VEC_T const & initial_x) {
            FLOAT_VEC_T t0 = initial_x;
            FLOAT_VEC_T t1 = logf<FLOAT_VEC_T, UINT_VEC_T>(initial_x);
            t0.assign(mask, t1);
            return t0;
        }
        // MLOG - double precision version
        template<typename FLOAT_VEC_T, typename UINT_VEC_T, typename MASK_T>
        UME_FORCE_INLINE FLOAT_VEC_T logd(MASK_T const & mask, FLOAT_VEC_T const & initial_x) {
            FLOAT_VEC_T t0 = initial_x;
            FLOAT_VEC_T t1 = logd<FLOAT_VEC_T, UINT_VEC_T>(initial_x);
            t0.assign(mask, t1);
            return t0;
        }
    
        // LOG2
        // LOG10

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

        // TAN
        // MTAN
        // CTAN
        // MCTAN
    }
}
}

#endif
