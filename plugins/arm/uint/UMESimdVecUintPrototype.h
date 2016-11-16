// The MIT License (MIT)
//
// Copyright (c) 2015 CERN
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

#ifndef UME_SIMD_VEC_UINT_PROTOTYPE_H_
#define UME_SIMD_VEC_UINT_PROTOTYPE_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#include "../UMESimdMask.h"
#include "../UMESimdSwizzle.h"

namespace UME {
namespace SIMD {
    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename VEC_TYPE, uint32_t VEC_LEN>
    struct SIMDVec_u_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 1> {
        typedef NullType<1>       HALF_LEN_VEC_TYPE;
        typedef int8_t            SCALAR_INT_TYPE;
        typedef NullType<2>       SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>    MASK_TYPE;
        typedef SIMDSwizzle<1>    SWIZZLE_MASK_TYPE;
        typedef NullType<3>       SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t          SCALAR_UINT_HIGHER_PRECISION;
    };

    // 16b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 2> {
        typedef SIMDVec_u<uint8_t, 1> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>        MASK_TYPE;
        typedef SIMDSwizzle<2>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 1> {
        typedef NullType<1>         HALF_LEN_VEC_TYPE;
        typedef int16_t             SCALAR_INT_TYPE;
        typedef NullType<2>         SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>      MASK_TYPE;
        typedef SIMDSwizzle<1>      SWIZZLE_MASK_TYPE;
        typedef uint8_t             SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 32b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 4> {
        typedef SIMDVec_u<uint8_t, 2> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>        MASK_TYPE;
        typedef SIMDSwizzle<4>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 2> {
        typedef SIMDVec_u<uint16_t, 1> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>         MASK_TYPE;
        typedef SIMDSwizzle<2>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 1> {
        typedef NullType<1>         HALF_LEN_VEC_TYPE;
        typedef int32_t             SCALAR_INT_TYPE;
        typedef float               SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>      MASK_TYPE;
        typedef SIMDSwizzle<1>      SWIZZLE_MASK_TYPE;
        typedef uint16_t            SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 64b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 8> {
        typedef SIMDVec_u<uint8_t, 4> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>        MASK_TYPE;
        typedef SIMDSwizzle<8>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 4> {
        typedef SIMDVec_u<uint16_t, 2> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>         MASK_TYPE;
        typedef SIMDSwizzle<4>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 2> {
        typedef SIMDVec_u<uint32_t, 1> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>         MASK_TYPE;
        typedef SIMDSwizzle<2>         SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 1> {
        typedef NullType<1>         HALF_LEN_VEC_TYPE;
        typedef int64_t             SCALAR_INT_TYPE;
        typedef double              SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>      MASK_TYPE;
        typedef SIMDSwizzle<1>      SWIZZLE_MASK_TYPE;
        typedef uint32_t            SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<2>         SCALAR_UINT_HIGHER_PRECISION;
    };

    // 128b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 16> {
        typedef SIMDVec_u<uint8_t, 8> HALF_LEN_VEC_TYPE;
        typedef int8_t                SCALAR_INT_TYPE;
        typedef NullType<1>           SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>       MASK_TYPE;
        typedef SIMDSwizzle<16>       SWIZZLE_MASK_TYPE;
        typedef NullType<2>           SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t              SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 8> {
        typedef SIMDVec_u<uint16_t, 4> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>         MASK_TYPE;
        typedef SIMDSwizzle<8>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 4> {
        typedef SIMDVec_u<uint32_t, 2> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>         MASK_TYPE;
        typedef SIMDSwizzle<4>         SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 2> {
        typedef SIMDVec_u<uint64_t, 1> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>         MASK_TYPE;
        typedef SIMDSwizzle<2>         SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 256b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 32> {
        typedef SIMDVec_u<uint8_t, 16> HALF_LEN_VEC_TYPE;
        typedef int8_t                 SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>        MASK_TYPE;
        typedef SIMDSwizzle<32>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>            SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 16> {
        typedef SIMDVec_u<uint16_t, 8> HALF_LEN_VEC_TYPE;
        typedef int16_t                SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>        MASK_TYPE;
        typedef SIMDSwizzle<16>        SWIZZLE_MASK_TYPE;
        typedef uint8_t                SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 8> {
        typedef SIMDVec_u<uint32_t, 4> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>         MASK_TYPE;
        typedef SIMDSwizzle<8>         SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 4> {
        typedef SIMDVec_u<uint64_t, 2> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>         MASK_TYPE;
        typedef SIMDSwizzle<4>         SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<2>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 512b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 64> {
        typedef SIMDVec_u<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef int8_t                 SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<64>        MASK_TYPE;
        typedef SIMDSwizzle<64>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>            SCALAR_UINT_LOWER_PRECISION;
        typedef uint16_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 32> {
        typedef SIMDVec_u<uint16_t, 16> HALF_LEN_VEC_TYPE;
        typedef int16_t                 SCALAR_INT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                 SCALAR_UINT_LOWER_PRECISION;
        typedef uint32_t                SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 16> {
        typedef SIMDVec_u<uint32_t, 8> HALF_LEN_VEC_TYPE;
        typedef int32_t                SCALAR_INT_TYPE;
        typedef float                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>        MASK_TYPE;
        typedef SIMDSwizzle<16>        SWIZZLE_MASK_TYPE;
        typedef uint16_t               SCALAR_UINT_LOWER_PRECISION;
        typedef uint64_t               SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 8> {
        typedef SIMDVec_u<uint64_t, 4> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>         MASK_TYPE;
        typedef SIMDSwizzle<8>         SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // 1024b vectors
    template<>
    struct SIMDVec_u_traits<uint8_t, 128> {
        typedef SIMDVec_u<uint8_t, 64> HALF_LEN_VEC_TYPE;
        typedef int8_t                 SCALAR_INT_TYPE;
        typedef NullType<1>            SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<128>       MASK_TYPE;
        typedef SIMDSwizzle<128>       SWIZZLE_MASK_TYPE;
        typedef NullType<2>            SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<3>            SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint16_t, 64> {
        typedef SIMDVec_u<uint16_t, 32> HALF_LEN_VEC_TYPE;
        typedef int16_t                 SCALAR_INT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<64>         MASK_TYPE;
        typedef SIMDSwizzle<64>         SWIZZLE_MASK_TYPE;
        typedef uint8_t                 SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint32_t, 32> {
        typedef SIMDVec_u<uint32_t, 16> HALF_LEN_VEC_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef uint16_t                SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_UINT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_u_traits<uint64_t, 16> {
        typedef SIMDVec_u<uint64_t, 8> HALF_LEN_VEC_TYPE;
        typedef int64_t                SCALAR_INT_TYPE;
        typedef double                 SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>        MASK_TYPE;
        typedef SIMDSwizzle<16>        SWIZZLE_MASK_TYPE;
        typedef uint32_t               SCALAR_UINT_LOWER_PRECISION;
        typedef NullType<1>            SCALAR_UINT_HIGHER_PRECISION;
    };

    // ***************************************************************************
    // *
    // *    Implementation of unsigned integer SIMDx_8u, SIMDx_16u, SIMDx_32u, 
    // *    and SIMDx_64u.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecUnsignedInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_UINT_TYPE, uint32_t VEC_LEN>
    class SIMDVec_u :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>,
            SCALAR_UINT_TYPE,
            VEC_LEN,
            typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
            SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>,        
            typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, VEC_LEN>   VEC_EMU_REG;

        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_INT_TYPE    SCALAR_INT_TYPE;
        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_FLOAT_TYPE  SCALAR_FLOAT_TYPE;
        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE          MASK_TYPE;

        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_UINT_LOWER_PRECISION  SCALAR_UINT_LOWER_PRECISION;
        typedef typename SIMDVec_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_UINT_HIGHER_PRECISION SCALAR_UINT_HIGHER_PRECISION;

        // Conversion operators require access to private members.
        friend class SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>;
    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i) : mVec(i) {};
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, SCALAR_UINT_TYPE>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<SCALAR_UINT_TYPE>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_u(SCALAR_UINT_TYPE const * p) { this->load(p); }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7)
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
        }

        UME_FORCE_INLINE SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15,
            SCALAR_UINT_TYPE i16, SCALAR_UINT_TYPE i17, SCALAR_UINT_TYPE i18, SCALAR_UINT_TYPE i19, SCALAR_UINT_TYPE i20, SCALAR_UINT_TYPE i21, SCALAR_UINT_TYPE i22, SCALAR_UINT_TYPE i23,
            SCALAR_UINT_TYPE i24, SCALAR_UINT_TYPE i25, SCALAR_UINT_TYPE i26, SCALAR_UINT_TYPE i27, SCALAR_UINT_TYPE i28, SCALAR_UINT_TYPE i29, SCALAR_UINT_TYPE i30, SCALAR_UINT_TYPE i31)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
        }

        UME_FORCE_INLINE SIMDVec_u(
            SCALAR_UINT_TYPE i0,  SCALAR_UINT_TYPE i1,  SCALAR_UINT_TYPE i2,  SCALAR_UINT_TYPE i3,  SCALAR_UINT_TYPE i4,  SCALAR_UINT_TYPE i5,  SCALAR_UINT_TYPE i6,  SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8,  SCALAR_UINT_TYPE i9,  SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15,
            SCALAR_UINT_TYPE i16, SCALAR_UINT_TYPE i17, SCALAR_UINT_TYPE i18, SCALAR_UINT_TYPE i19, SCALAR_UINT_TYPE i20, SCALAR_UINT_TYPE i21, SCALAR_UINT_TYPE i22, SCALAR_UINT_TYPE i23,
            SCALAR_UINT_TYPE i24, SCALAR_UINT_TYPE i25, SCALAR_UINT_TYPE i26, SCALAR_UINT_TYPE i27, SCALAR_UINT_TYPE i28, SCALAR_UINT_TYPE i29, SCALAR_UINT_TYPE i30, SCALAR_UINT_TYPE i31,
            SCALAR_UINT_TYPE i32, SCALAR_UINT_TYPE i33, SCALAR_UINT_TYPE i34, SCALAR_UINT_TYPE i35, SCALAR_UINT_TYPE i36, SCALAR_UINT_TYPE i37, SCALAR_UINT_TYPE i38, SCALAR_UINT_TYPE i39,
            SCALAR_UINT_TYPE i40, SCALAR_UINT_TYPE i41, SCALAR_UINT_TYPE i42, SCALAR_UINT_TYPE i43, SCALAR_UINT_TYPE i44, SCALAR_UINT_TYPE i45, SCALAR_UINT_TYPE i46, SCALAR_UINT_TYPE i47,
            SCALAR_UINT_TYPE i48, SCALAR_UINT_TYPE i49, SCALAR_UINT_TYPE i50, SCALAR_UINT_TYPE i51, SCALAR_UINT_TYPE i52, SCALAR_UINT_TYPE i53, SCALAR_UINT_TYPE i54, SCALAR_UINT_TYPE i55,
            SCALAR_UINT_TYPE i56, SCALAR_UINT_TYPE i57, SCALAR_UINT_TYPE i58, SCALAR_UINT_TYPE i59, SCALAR_UINT_TYPE i60, SCALAR_UINT_TYPE i61, SCALAR_UINT_TYPE i62, SCALAR_UINT_TYPE i63)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
            mVec.insert(32, i32);  mVec.insert(33, i33);  mVec.insert(34, i34);  mVec.insert(35, i35);
            mVec.insert(36, i36);  mVec.insert(37, i37);  mVec.insert(38, i38);  mVec.insert(39, i39);
            mVec.insert(40, i40);  mVec.insert(41, i41);  mVec.insert(42, i42);  mVec.insert(43, i43);
            mVec.insert(44, i44);  mVec.insert(45, i45);  mVec.insert(46, i46);  mVec.insert(47, i47);
            mVec.insert(48, i48);  mVec.insert(49, i49);  mVec.insert(50, i50);  mVec.insert(51, i51);
            mVec.insert(52, i52);  mVec.insert(53, i53);  mVec.insert(54, i54);  mVec.insert(55, i55);
            mVec.insert(56, i56);  mVec.insert(57, i57);  mVec.insert(58, i58);  mVec.insert(59, i59);
            mVec.insert(60, i60);  mVec.insert(61, i61);  mVec.insert(62, i62);  mVec.insert(63, i63);
        }
        
        UME_FORCE_INLINE SIMDVec_u(
            SCALAR_UINT_TYPE i0,   SCALAR_UINT_TYPE i1,   SCALAR_UINT_TYPE i2,   SCALAR_UINT_TYPE i3,   SCALAR_UINT_TYPE i4,   SCALAR_UINT_TYPE i5,   SCALAR_UINT_TYPE i6,   SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8,   SCALAR_UINT_TYPE i9,   SCALAR_UINT_TYPE i10,  SCALAR_UINT_TYPE i11,  SCALAR_UINT_TYPE i12,  SCALAR_UINT_TYPE i13,  SCALAR_UINT_TYPE i14,  SCALAR_UINT_TYPE i15,
            SCALAR_UINT_TYPE i16,  SCALAR_UINT_TYPE i17,  SCALAR_UINT_TYPE i18,  SCALAR_UINT_TYPE i19,  SCALAR_UINT_TYPE i20,  SCALAR_UINT_TYPE i21,  SCALAR_UINT_TYPE i22,  SCALAR_UINT_TYPE i23,
            SCALAR_UINT_TYPE i24,  SCALAR_UINT_TYPE i25,  SCALAR_UINT_TYPE i26,  SCALAR_UINT_TYPE i27,  SCALAR_UINT_TYPE i28,  SCALAR_UINT_TYPE i29,  SCALAR_UINT_TYPE i30,  SCALAR_UINT_TYPE i31,
            SCALAR_UINT_TYPE i32,  SCALAR_UINT_TYPE i33,  SCALAR_UINT_TYPE i34,  SCALAR_UINT_TYPE i35,  SCALAR_UINT_TYPE i36,  SCALAR_UINT_TYPE i37,  SCALAR_UINT_TYPE i38,  SCALAR_UINT_TYPE i39,
            SCALAR_UINT_TYPE i40,  SCALAR_UINT_TYPE i41,  SCALAR_UINT_TYPE i42,  SCALAR_UINT_TYPE i43,  SCALAR_UINT_TYPE i44,  SCALAR_UINT_TYPE i45,  SCALAR_UINT_TYPE i46,  SCALAR_UINT_TYPE i47,
            SCALAR_UINT_TYPE i48,  SCALAR_UINT_TYPE i49,  SCALAR_UINT_TYPE i50,  SCALAR_UINT_TYPE i51,  SCALAR_UINT_TYPE i52,  SCALAR_UINT_TYPE i53,  SCALAR_UINT_TYPE i54,  SCALAR_UINT_TYPE i55,
            SCALAR_UINT_TYPE i56,  SCALAR_UINT_TYPE i57,  SCALAR_UINT_TYPE i58,  SCALAR_UINT_TYPE i59,  SCALAR_UINT_TYPE i60,  SCALAR_UINT_TYPE i61,  SCALAR_UINT_TYPE i62,  SCALAR_UINT_TYPE i63,
            SCALAR_UINT_TYPE i64,  SCALAR_UINT_TYPE i65,  SCALAR_UINT_TYPE i66,  SCALAR_UINT_TYPE i67,  SCALAR_UINT_TYPE i68,  SCALAR_UINT_TYPE i69,  SCALAR_UINT_TYPE i70,  SCALAR_UINT_TYPE i71,
            SCALAR_UINT_TYPE i72,  SCALAR_UINT_TYPE i73,  SCALAR_UINT_TYPE i74,  SCALAR_UINT_TYPE i75,  SCALAR_UINT_TYPE i76,  SCALAR_UINT_TYPE i77,  SCALAR_UINT_TYPE i78,  SCALAR_UINT_TYPE i79,
            SCALAR_UINT_TYPE i80,  SCALAR_UINT_TYPE i81,  SCALAR_UINT_TYPE i82,  SCALAR_UINT_TYPE i83,  SCALAR_UINT_TYPE i84,  SCALAR_UINT_TYPE i85,  SCALAR_UINT_TYPE i86,  SCALAR_UINT_TYPE i87,
            SCALAR_UINT_TYPE i88,  SCALAR_UINT_TYPE i89,  SCALAR_UINT_TYPE i90,  SCALAR_UINT_TYPE i91,  SCALAR_UINT_TYPE i92,  SCALAR_UINT_TYPE i93,  SCALAR_UINT_TYPE i94,  SCALAR_UINT_TYPE i95,
            SCALAR_UINT_TYPE i96,  SCALAR_UINT_TYPE i97,  SCALAR_UINT_TYPE i98,  SCALAR_UINT_TYPE i99,  SCALAR_UINT_TYPE i100, SCALAR_UINT_TYPE i101, SCALAR_UINT_TYPE i102, SCALAR_UINT_TYPE i103,
            SCALAR_UINT_TYPE i104, SCALAR_UINT_TYPE i105, SCALAR_UINT_TYPE i106, SCALAR_UINT_TYPE i107, SCALAR_UINT_TYPE i108, SCALAR_UINT_TYPE i109, SCALAR_UINT_TYPE i110, SCALAR_UINT_TYPE i111,
            SCALAR_UINT_TYPE i112, SCALAR_UINT_TYPE i113, SCALAR_UINT_TYPE i114, SCALAR_UINT_TYPE i115, SCALAR_UINT_TYPE i116, SCALAR_UINT_TYPE i117, SCALAR_UINT_TYPE i118, SCALAR_UINT_TYPE i119,
            SCALAR_UINT_TYPE i120, SCALAR_UINT_TYPE i121, SCALAR_UINT_TYPE i122, SCALAR_UINT_TYPE i123, SCALAR_UINT_TYPE i124, SCALAR_UINT_TYPE i125, SCALAR_UINT_TYPE i126, SCALAR_UINT_TYPE i127)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
            mVec.insert(16, i16);  mVec.insert(17, i17);  mVec.insert(18, i18);  mVec.insert(19, i19);
            mVec.insert(20, i20);  mVec.insert(21, i21);  mVec.insert(22, i22);  mVec.insert(23, i23);
            mVec.insert(24, i24);  mVec.insert(25, i25);  mVec.insert(26, i26);  mVec.insert(27, i27);
            mVec.insert(28, i28);  mVec.insert(29, i29);  mVec.insert(30, i30);  mVec.insert(31, i31);
            mVec.insert(32, i32);  mVec.insert(33, i33);  mVec.insert(34, i34);  mVec.insert(35, i35);
            mVec.insert(36, i36);  mVec.insert(37, i37);  mVec.insert(38, i38);  mVec.insert(39, i39);
            mVec.insert(40, i40);  mVec.insert(41, i41);  mVec.insert(42, i42);  mVec.insert(43, i43);
            mVec.insert(44, i44);  mVec.insert(45, i45);  mVec.insert(46, i46);  mVec.insert(47, i47);
            mVec.insert(48, i48);  mVec.insert(49, i49);  mVec.insert(50, i50);  mVec.insert(51, i51);
            mVec.insert(52, i52);  mVec.insert(53, i53);  mVec.insert(54, i54);  mVec.insert(55, i55);
            mVec.insert(56, i56);  mVec.insert(57, i57);  mVec.insert(58, i58);  mVec.insert(59, i59);
            mVec.insert(60, i60);  mVec.insert(61, i61);  mVec.insert(62, i62);  mVec.insert(63, i63);
            mVec.insert(64,  i64);   mVec.insert(65, i65);    mVec.insert(66, i66);    mVec.insert(67, i67);
            mVec.insert(68,  i68);   mVec.insert(69, i69);    mVec.insert(70, i70);    mVec.insert(71, i71);
            mVec.insert(72,  i72);   mVec.insert(73, i73);    mVec.insert(74, i74);    mVec.insert(75, i75);
            mVec.insert(76,  i76);   mVec.insert(77, i77);    mVec.insert(78, i78);    mVec.insert(79, i79);
            mVec.insert(80,  i80);   mVec.insert(81, i81);    mVec.insert(82, i82);    mVec.insert(83, i83);
            mVec.insert(84,  i84);   mVec.insert(85, i85);    mVec.insert(86, i86);    mVec.insert(87, i87);
            mVec.insert(88,  i88);   mVec.insert(89, i89);    mVec.insert(90, i90);    mVec.insert(91, i91);
            mVec.insert(92,  i92);   mVec.insert(93, i93);    mVec.insert(94, i94);    mVec.insert(95, i95);
            mVec.insert(96,  i96);   mVec.insert(97, i97);    mVec.insert(98, i98);    mVec.insert(99, i99);
            mVec.insert(100, i100);  mVec.insert(101, i101);  mVec.insert(102, i102);  mVec.insert(103, i103);
            mVec.insert(104, i104);  mVec.insert(105, i105);  mVec.insert(106, i106);  mVec.insert(107, i107);
            mVec.insert(108, i108);  mVec.insert(109, i109);  mVec.insert(110, i110);  mVec.insert(111, i111);
            mVec.insert(112, i112);  mVec.insert(113, i113);  mVec.insert(114, i114);  mVec.insert(115, i115);
            mVec.insert(116, i116);  mVec.insert(117, i117);  mVec.insert(118, i118);  mVec.insert(119, i119);
            mVec.insert(120, i120);  mVec.insert(121, i121);  mVec.insert(122, i122);  mVec.insert(123, i123);
            mVec.insert(124, i124);  mVec.insert(125, i125);  mVec.insert(126, i126);  mVec.insert(127, i127);
        }
        
        // EXTRACT
        UME_FORCE_INLINE SCALAR_UINT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, SCALAR_UINT_TYPE> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, SCALAR_UINT_TYPE>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & operator= (SCALAR_UINT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS

        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<SCALAR_UINT_LOWER_PRECISION, VEC_LEN>() const;
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<SCALAR_UINT_HIGHER_PRECISION, VEC_LEN>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>() const;
    };

    // SIMD NullTypes. These are used whenever a terminating
    // scalar type is used as a creator function for SIMD type.
    // These types cannot be instantiated, but are necessary for 
    // typeset to be consistent.
    template<>
    class SIMDVec_u<NullType<1>, 1>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 2>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 4>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 8>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 16>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 32>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 64>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<1>, 128>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 1>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 2>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 4>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 8>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 16>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 32>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 64>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<2>, 128>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 1>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 2>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 4>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 8>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 16>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 32>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 64>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };

    template<>
    class SIMDVec_u<NullType<3>, 128>
    {
    private:
        SIMDVec_u() {}
        ~SIMDVec_u() {}
    };
}
}

#endif
