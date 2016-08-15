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

#include "../UMESimdMaskKNC.h"
#include "../UMESimdSwizzleKNC.h"

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
        inline SIMDVec_u() : mVec() {};

        // SET-CONSTR
        inline SIMDVec_u(SCALAR_UINT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_u(SCALAR_UINT_TYPE const * p) { this->load(p); }

        inline SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        inline SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7)
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
        }

        inline SIMDVec_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
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

        // EXTRACT
        inline SCALAR_UINT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_u, SCALAR_UINT_TYPE> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, SCALAR_UINT_TYPE>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_u, SCALAR_UINT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_u & operator= (SIMDVec_u const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        // ASSIGNS
        inline SIMDVec_u & operator= (SCALAR_UINT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS

        // DEGRADE
        inline operator SIMDVec_u<SCALAR_UINT_LOWER_PRECISION, VEC_LEN>() const;
        // PROMOTE
        inline operator SIMDVec_u<SCALAR_UINT_HIGHER_PRECISION, VEC_LEN>() const;

        // UTOI
        inline operator SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>() const;
        // UTOF
        inline operator SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>() const;
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
