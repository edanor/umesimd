#include "../../UMESimd.h"

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923   // pi/2
#endif

#ifndef M_PI_4
#define M_PI_4     0.785398163397448309616  // pi/4
#endif

#ifndef M_1_PI
#define M_1_PI     0.318309886183790671538  // 1/pi
#endif

#ifndef M_2_PI
#define M_2_PI     0.636619772367581343076  // 2/pi
#endif

const double TWOPI = 2.*M_PI;
const double PI = M_PI;
const double PIO2 = M_PI_2;
const double PIO4 = M_PI_4;
//const double ONEOPIO4 = 4. / M_PI;
const double ONEOPIO4 = 4.0 / (3.14159265358979323846);
const float ONEOPIO4F = 4.0f / (3.1415927f);

const double MOREBITS = 6.123233995736765886130E-17;

const double MAXNUMF = 3.4028234663852885981170418348451692544e38f;

const float DP1F = (float)0.78515625;
const float DP2F = (float)2.4187564849853515625e-4;
const float DP3F = (float)3.77489497744594108e-8;

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

const float T24M1 = 16777215.;

template<typename FLOAT_SCALAR_T>
inline FLOAT_SCALAR_T DP1() { return -1.0; }
template<typename FLOAT_SCALAR_T>
inline FLOAT_SCALAR_T DP2() { return -1.0; }
template<typename FLOAT_SCALAR_T>
inline FLOAT_SCALAR_T DP3() { return -1.0; }

template<> inline float DP1<float>() { return DP1F; }
template<> inline float DP2<float>() { return DP2F; }
template<> inline float DP3<float>() { return DP3F; }

template<> inline double DP1<double>() { return DP1D; }
template<> inline double DP2<double>() { return DP2D; }
template<> inline double DP3<double>() { return DP3D; }

template<typename FLOAT_VEC_T>
inline void sincosf(FLOAT_VEC_T const & xx, FLOAT_VEC_T & s, FLOAT_VEC_T &c )
{
    typedef typename SIMDTraits<FLOAT_VEC_T>::SCALAR_T  FLOAT_SCALAR_T;
    typedef typename SIMDTraits<FLOAT_VEC_T>::INT_VEC_T INT_VEC_T;
    typedef typename SIMDTraits<FLOAT_VEC_T>::MASK_T MASK_T;

    INT_VEC_T j;

    /* make argument positive */
    FLOAT_VEC_T x_pos = xx.abs();

    j = INT_VEC_T(FLOAT_SCALAR_T(ONEOPIO4F) * x_pos); /* integer part of x/PIO4 */

    j = (j + 1) & (~1);
    const FLOAT_VEC_T y = FLOAT_VEC_T(j);

    // Extended precision modular arithmetic
    const FLOAT_VEC_T x = ((x_pos - y * FLOAT_SCALAR_T(DP1<FLOAT_SCALAR_T>())) - y * FLOAT_SCALAR_T(DP2<FLOAT_SCALAR_T>())) - y * FLOAT_SCALAR_T(DP3<FLOAT_SCALAR_T>());

    INT_VEC_T signS = (j&4);
    j-=2;

    const INT_VEC_T signC = (j&4);
    const INT_VEC_T poly = j&2;

    FLOAT_VEC_T ls,lc;

    FLOAT_VEC_T z = x * x;

    ls = (((FLOAT_SCALAR_T(-1.9515295891E-4f) * z
        + FLOAT_SCALAR_T(8.3321608736E-3f)) * z
        - FLOAT_SCALAR_T(1.6666654611E-1f)) * z * x)
        + x;

    lc = ((FLOAT_SCALAR_T(2.443315711809948E-005f) * z
        - FLOAT_SCALAR_T(1.388731625493765E-003f)) * z
        + FLOAT_SCALAR_T(4.166664568298827E-002f)) * z * z
        - FLOAT_SCALAR_T(0.5f) * z + FLOAT_SCALAR_T(1.0f);

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

    c=lc;
    s=ls;
  }

template<typename DOUBLE_VEC_T>
inline void fast_sincos(DOUBLE_VEC_T const & xx, DOUBLE_VEC_T & s, DOUBLE_VEC_T &c) {
    typedef typename SIMDTraits<DOUBLE_VEC_T>::INT_VEC_T INT_VEC_T;
    typedef typename SIMDTraits<DOUBLE_VEC_T>::MASK_T MASK_T;

    INT_VEC_T j;

    DOUBLE_VEC_T x = xx.abs();
    j = INT_VEC_T(ONEOPIO4 * x); // always positive, so (int) == std::floor
    j = (j + 1) & (~1);
    const DOUBLE_VEC_T y = DOUBLE_VEC_T(j);
    // Extended precision modular arithmetic
    x = ((x - y * DP1D) - y * DP2D) - y * DP3D;

    const DOUBLE_VEC_T signS = (j & 4);

    j -= 2;

    const DOUBLE_VEC_T signC = (j & 4);
    const DOUBLE_VEC_T poly = j & 2;

    DOUBLE_VEC_T zz = x * x;

    DOUBLE_VEC_T px1(C1sin);
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

    DOUBLE_VEC_T px2(C1cos);
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

    const DOUBLE_VEC_T tmp = c;
    c.assign(maskPoly, s);
    s.assign(maskPoly, tmp);

    MASK_T maskSignC = (signC == 0);
    c.nega(maskSignC);

    MASK_T maskSignS = (signS != 0);
    s.nega(maskSignS);

    MASK_T maskXX = (xx < 0);
    s.nega(maskXX);
}


/*
const double UME_PI = 3.14159265358979323846;       // PI
const double UME_PI_2 = 1.57079632679489661923;     // PI/2
const double UME_PI_4 = 0.785398163397448309616;    // PI/4
const double UME_1_PI = 0.318309886183790671538;    // 1/PI
const double UME_2_PI = 0.636619772367581343076;    // 2/PI
const double UME_TWO_PI = 2.0 * UME_PI;             // 2*PI

template<typename VEC_TYPE>
UME_FORCE_INLINE VEC_TYPE sin(VEC_TYPE const & a) {
    UME_EMULATION_WARNING();
    VEC_TYPE retval;
    for (uint32_t i = 0; i < VEC_TYPE::length(); i++) {
        retval.insert(i, std::sin(a[i]));
    }
    return retval;
}*/