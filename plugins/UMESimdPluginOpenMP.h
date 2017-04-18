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

#ifndef UME_SIMD_PLUGIN_OPENMP_H_
#define UME_SIMD_PLUGIN_OPENMP_H_

#include <type_traits>

#include "../UMESimdInterface.h"

namespace UME
{
namespace SIMD
{
    // Specialize SuggestedVecLen
    template<> struct SuggestedVecLen<uint8_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<int8_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<uint16_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<int16_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<uint32_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<int32_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<uint64_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<int64_t> { static const int value = 1; };
    template<> struct SuggestedVecLen<float> { static const int value = 1; };
    template<> struct SuggestedVecLen<double> { static const int value = 1; };

    // forward declarations of simd types classes;
    template<uint32_t VEC_LEN>                             class SIMDVecMask;
    template<uint32_t SMASK_LEN>                           class SIMDSwizzle;
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>  class SIMDVec_u;
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>   class SIMDVec_i;
    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN> class SIMDVec_f;
}
}

#include "openmp/UMESimdMask.h"
#include "openmp/UMESimdSwizzle.h"
#include "openmp/UMESimdVecUint.h"
#include "openmp/UMESimdVecInt.h"
#include "openmp/UMESimdVecFloat.h"
#include "openmp/UMESimdCastOperators.h"

namespace UME
{
namespace SIMD
{
    // Mask types
    typedef SIMDVecMask<1>      SIMDMask1;
    typedef SIMDVecMask<2>      SIMDMask2;
    typedef SIMDVecMask<4>      SIMDMask4;
    typedef SIMDVecMask<8>      SIMDMask8;
    typedef SIMDVecMask<16>     SIMDMask16;
    typedef SIMDVecMask<32>     SIMDMask32;
    typedef SIMDVecMask<64>     SIMDMask64;
    typedef SIMDVecMask<128>    SIMDMask128;
    
    // Swizzle mask types
    typedef SIMDSwizzle<1>   SIMDSwizzle1;
    typedef SIMDSwizzle<2>   SIMDSwizzle2;
    typedef SIMDSwizzle<4>   SIMDSwizzle4;
    typedef SIMDSwizzle<8>   SIMDSwizzle8;
    typedef SIMDSwizzle<16>  SIMDSwizzle16;
    typedef SIMDSwizzle<32>  SIMDSwizzle32;
    typedef SIMDSwizzle<64>  SIMDSwizzle64;
    typedef SIMDSwizzle<128> SIMDSwizzle128;

    // 8b uint vectors
    typedef SIMDVec_u<uint8_t,  1>   SIMD1_8u;

    // 16b uint vectors
    typedef SIMDVec_u<uint8_t,  2>   SIMD2_8u;
    typedef SIMDVec_u<uint16_t, 1>   SIMD1_16u;

    // 32b uint vectors
    typedef SIMDVec_u<uint8_t,  4>   SIMD4_8u;
    typedef SIMDVec_u<uint16_t, 2>   SIMD2_16u;
    typedef SIMDVec_u<uint32_t, 1>   SIMD1_32u;

    // 64b uint vectors
    typedef SIMDVec_u<uint8_t,  8>   SIMD8_8u;
    typedef SIMDVec_u<uint16_t, 4>   SIMD4_16u;
    typedef SIMDVec_u<uint32_t, 2>   SIMD2_32u; 
    typedef SIMDVec_u<uint64_t, 1>   SIMD1_64u;

    // 128b uint vectors
    typedef SIMDVec_u<uint8_t,  16>  SIMD16_8u;
    typedef SIMDVec_u<uint16_t, 8>   SIMD8_16u;
    typedef SIMDVec_u<uint32_t, 4>   SIMD4_32u;
    typedef SIMDVec_u<uint64_t, 2>   SIMD2_64u;
    
    // 256b uint vectors
    typedef SIMDVec_u<uint8_t,  32>  SIMD32_8u;
    typedef SIMDVec_u<uint16_t, 16>  SIMD16_16u;
    typedef SIMDVec_u<uint32_t, 8>   SIMD8_32u;
    typedef SIMDVec_u<uint64_t, 4>   SIMD4_64u;
    
    // 512b uint vectors
    typedef SIMDVec_u<uint8_t,  64>  SIMD64_8u;
    typedef SIMDVec_u<uint16_t, 32>  SIMD32_16u;
    typedef SIMDVec_u<uint32_t, 16>  SIMD16_32u;
    typedef SIMDVec_u<uint64_t, 8>   SIMD8_64u;

    // 1024b uint vectors
    typedef SIMDVec_u<uint8_t, 128>  SIMD128_8u;
    typedef SIMDVec_u<uint16_t, 64>  SIMD64_16u;
    typedef SIMDVec_u<uint32_t, 32>  SIMD32_32u;
    typedef SIMDVec_u<uint64_t, 16>  SIMD16_64u;

    // 8b int vectors
    typedef SIMDVec_i<int8_t,   1>   SIMD1_8i;

    // 16b int vectors
    typedef SIMDVec_i<int8_t,   2>   SIMD2_8i;
    typedef SIMDVec_i<int16_t,  1>   SIMD1_16i;

    // 32b int vectors
    typedef SIMDVec_i<int8_t,   4>   SIMD4_8i;
    typedef SIMDVec_i<int16_t,  2>   SIMD2_16i;
    typedef SIMDVec_i<int32_t,  1>   SIMD1_32i;

    // 64b int vectors
    typedef SIMDVec_i<int8_t,   8>   SIMD8_8i; 
    typedef SIMDVec_i<int16_t,  4>   SIMD4_16i;
    typedef SIMDVec_i<int32_t,  2>   SIMD2_32i;
    typedef SIMDVec_i<int64_t,  1>   SIMD1_64i;

    // 128b int vectors
    typedef SIMDVec_i<int8_t,   16>  SIMD16_8i; 
    typedef SIMDVec_i<int16_t,  8>   SIMD8_16i;
    typedef SIMDVec_i<int32_t,  4>   SIMD4_32i;
    typedef SIMDVec_i<int64_t,  2>   SIMD2_64i;

    // 256b int vectors
    typedef SIMDVec_i<int8_t,   32>  SIMD32_8i;
    typedef SIMDVec_i<int16_t,  16>  SIMD16_16i;
    typedef SIMDVec_i<int32_t,  8>   SIMD8_32i;
    typedef SIMDVec_i<int64_t,  4>   SIMD4_64i;

    // 512b int vectors
    typedef SIMDVec_i<int8_t,   64>  SIMD64_8i;
    typedef SIMDVec_i<int16_t,  32>  SIMD32_16i;
    typedef SIMDVec_i<int32_t,  16>  SIMD16_32i;
    typedef SIMDVec_i<int64_t,  8>   SIMD8_64i;

    // 1024b int vectors
    typedef SIMDVec_i<int8_t,  128>  SIMD128_8i;
    typedef SIMDVec_i<int16_t,  64>  SIMD64_16i;
    typedef SIMDVec_i<int32_t,  32>  SIMD32_32i;
    typedef SIMDVec_i<int64_t,  16>  SIMD16_64i;

    // 32b float vectors
    typedef SIMDVec_f<float, 1>      SIMD1_32f;

    // 64b float vectors
    typedef SIMDVec_f<float, 2>      SIMD2_32f;
    typedef SIMDVec_f<double, 1>     SIMD1_64f;

    // 128b float vectors
    typedef SIMDVec_f<float,  4>     SIMD4_32f;
    typedef SIMDVec_f<double, 2>     SIMD2_64f;

    // 256b float vectors
    typedef SIMDVec_f<float,  8>     SIMD8_32f;
    typedef SIMDVec_f<double, 4>     SIMD4_64f;

    // 512b float vectors
    typedef SIMDVec_f<float,  16>    SIMD16_32f;
    typedef SIMDVec_f<double, 8>     SIMD8_64f;
    
    // 1024b float vectors
    typedef SIMDVec_f<float,  32>    SIMD32_32f;
    typedef SIMDVec_f<double, 16>    SIMD16_64f;
} // SIMD
} // UME

#if defined  (__AVX512F__)
    #include "avx512/UMESimdTraitsSpecialization.h"
#elif defined (__MIC__)
    #include "knc/UMESimdTraitsSpecialization.h"
#elif defined (__AVX2__)
    #include "avx2/UMESimdTraitsSpecialization.h"
#elif defined (__AVX__)
    #include "avx/UMESimdTraitsSpecialization.h"
#elif defined (__aarch64__) || defined (__arm__)
    #include "arm/UMESimdTraitsSpecialization.h"
#elif defined (__ALTIVEC__)
    #include "altivec/UMESimdTraitsSpecialization.h"
#else
    #include "scalar/UMESimdTraitsSpecialization.h"
#endif

#endif
