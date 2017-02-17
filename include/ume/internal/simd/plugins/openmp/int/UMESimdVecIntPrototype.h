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

#ifndef UME_SIMD_VEC_INT_PROTOTYPE_H_
#define UME_SIMD_VEC_INT_PROTOTYPE_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"

#include "../UMESimdMask.h"
#include "../UMESimdSwizzle.h"
#include "../UMESimdVecUint.h"

namespace UME {
namespace SIMD {

    // ********************************************************************************************
    // SIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    struct SIMDVec_i_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 1> {
        typedef NullType<1>             HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 1>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDSwizzle<1>          SWIZZLE_MASK_TYPE;
        typedef NullType<3>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    // 16b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 2> {
        typedef SIMDVec_i<int8_t, 1>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 2>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDSwizzle<2>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 1> {
        typedef NullType<1>             HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 1>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDSwizzle<1>          SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef int32_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    // 32b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 4> {
        typedef SIMDVec_i<int8_t, 2>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 4>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDSwizzle<4>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 2> {
        typedef SIMDVec_i<int16_t, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 2>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDSwizzle<2>          SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef int32_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 1> {
        typedef NullType<1>             HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 1>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDSwizzle<1>          SWIZZLE_MASK_TYPE;
        typedef int16_t                 SCALAR_INT_LOWER_PRECISION;
        typedef int64_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    // 64b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 8> {
        typedef SIMDVec_i<int8_t, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 8>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDSwizzle<8>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 4> {
        typedef SIMDVec_i<int16_t, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 4>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDSwizzle<4>          SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef int32_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 2> {
        typedef SIMDVec_i<int32_t, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 2>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDSwizzle<2>          SWIZZLE_MASK_TYPE;
        typedef int16_t                 SCALAR_INT_LOWER_PRECISION;
        typedef int64_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 1> {
        typedef NullType<1>             HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 1>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDSwizzle<1>          SWIZZLE_MASK_TYPE;
        typedef int32_t                 SCALAR_INT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_INT_HIGHER_PRECISION;
    };

    // 128b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 16> {
        typedef SIMDVec_i<int8_t, 8>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 16>  VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDSwizzle<16>         SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 8> {
        typedef SIMDVec_i<int16_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 8>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDSwizzle<8>          SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef int32_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 4> {
        typedef SIMDVec_i<int32_t, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 4>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDSwizzle<4>          SWIZZLE_MASK_TYPE;
        typedef int16_t                 SCALAR_INT_LOWER_PRECISION;
        typedef int64_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 2> {
        typedef SIMDVec_i<int64_t, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 2>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDSwizzle<2>          SWIZZLE_MASK_TYPE;
        typedef int32_t                 SCALAR_INT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_INT_HIGHER_PRECISION;
    };

    // 256b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 32> {
        typedef SIMDVec_i<int8_t, 16>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 32>  VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 16> {
        typedef SIMDVec_i<int16_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDSwizzle<16>         SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef int32_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 8> {
        typedef SIMDVec_i<int32_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 8>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDSwizzle<8>          SWIZZLE_MASK_TYPE;
        typedef int16_t                 SCALAR_INT_LOWER_PRECISION;
        typedef int64_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 4> {
        typedef SIMDVec_i<int64_t, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 4>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDSwizzle<4>          SWIZZLE_MASK_TYPE;
        typedef int32_t                 SCALAR_INT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_INT_HIGHER_PRECISION;
    };

    // 512b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 64> {
        typedef SIMDVec_i<int8_t, 32>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 64>  VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<64>         MASK_TYPE;
        typedef SIMDSwizzle<64>         SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef int16_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 32> {
        typedef SIMDVec_i<int16_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef int32_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 16> {
        typedef SIMDVec_i<int32_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDSwizzle<16>         SWIZZLE_MASK_TYPE;
        typedef int16_t                 SCALAR_INT_LOWER_PRECISION;
        typedef int64_t                 SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 8> {
        typedef SIMDVec_i<int64_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 8>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDSwizzle<8>          SWIZZLE_MASK_TYPE;
        typedef int32_t                 SCALAR_INT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_INT_HIGHER_PRECISION;
    };

    // 1024b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 128> {
        typedef SIMDVec_i<int8_t, 64>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<128>        MASK_TYPE;
        typedef SIMDSwizzle<128>        SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_INT_LOWER_PRECISION;
        typedef NullType<3>             SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 64> {
        typedef SIMDVec_i<int16_t, 32>  HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef NullType<1>             SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<64>         MASK_TYPE;
        typedef SIMDSwizzle<64>         SWIZZLE_MASK_TYPE;
        typedef int8_t                  SCALAR_INT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 32> {
        typedef SIMDVec_i<int32_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float                   SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef int16_t                 SCALAR_INT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_INT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 16> {
        typedef SIMDVec_i<int64_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 16> VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double                  SCALAR_FLOAT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDSwizzle<16>         SWIZZLE_MASK_TYPE;
        typedef int32_t                 SCALAR_INT_LOWER_PRECISION;
        typedef NullType<1>             SCALAR_INT_HIGHER_PRECISION;
    };

    // ***************************************************************************
    // *
    // *    Implementation of signed integer SIMDx_8i, SIMDx_16i, SIMDx_32i, 
    // *    and SIMDx_64i.
    // *
    // *    This implementation uses scalar emulation available through to 
    // *    SIMDVecSignedInterface.
    // *
    // ***************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    class SIMDVec_i :
        public SIMDVecSignedInterface<
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>,
            typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT,
            SCALAR_INT_TYPE,
            VEC_LEN,
            typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>,
            typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, VEC_LEN> VEC_EMU_REG;

        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE   SCALAR_UINT_TYPE;
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_FLOAT_TYPE  SCALAR_FLOAT_TYPE;
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT           VEC_UINT;
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE          MASK_TYPE;

        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_INT_LOWER_PRECISION  SCALAR_INT_LOWER_PRECISION;
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_INT_HIGHER_PRECISION SCALAR_INT_HIGHER_PRECISION;

        friend class SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>;
        friend class SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>;
        
    public:
        constexpr static uint32_t alignment() { return VEC_LEN*sizeof(SCALAR_INT_TYPE); }

    private:
        alignas(alignment()) SCALAR_INT_TYPE mVec[VEC_LEN];

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(SCALAR_INT_TYPE x) {
            SCALAR_INT_TYPE *local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for (unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = x;
            }
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, SCALAR_INT_TYPE>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<SCALAR_INT_TYPE>(i)) {}

        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(SCALAR_INT_TYPE const * p) { this->load(p); }

        UME_FORCE_INLINE SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1) {
            mVec[0] = i0;  mVec[1] = i1;
        }

        UME_FORCE_INLINE SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1,
            SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            insert(0, i0);  insert(1, i1);
            insert(2, i2);  insert(3, i3);
        }

        UME_FORCE_INLINE SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1,
            SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3,
            SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5,
            SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7)
        {
            insert(0, i0);  insert(1, i1);
            insert(2, i2);  insert(3, i3);
            insert(4, i4);  insert(5, i5);
            insert(6, i6);  insert(7, i7);
        }

        UME_FORCE_INLINE SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1,
            SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3,
            SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5,
            SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9,
            SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11,
            SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13,
            SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            insert(0, i0);    insert(1, i1);
            insert(2, i2);    insert(3, i3);
            insert(4, i4);    insert(5, i5);
            insert(6, i6);    insert(7, i7);
            insert(8, i8);    insert(9, i9);
            insert(10, i10);  insert(11, i11);
            insert(12, i12);  insert(13, i13);
            insert(14, i14);  insert(15, i15);
        }

        UME_FORCE_INLINE SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1,
            SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3,
            SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5,
            SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9,
            SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11,
            SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13,
            SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15,
            SCALAR_INT_TYPE i16, SCALAR_INT_TYPE i17,
            SCALAR_INT_TYPE i18, SCALAR_INT_TYPE i19,
            SCALAR_INT_TYPE i20, SCALAR_INT_TYPE i21,
            SCALAR_INT_TYPE i22, SCALAR_INT_TYPE i23,
            SCALAR_INT_TYPE i24, SCALAR_INT_TYPE i25,
            SCALAR_INT_TYPE i26, SCALAR_INT_TYPE i27,
            SCALAR_INT_TYPE i28, SCALAR_INT_TYPE i29,
            SCALAR_INT_TYPE i30, SCALAR_INT_TYPE i31)
        {
            insert(0, i0);    insert(1, i1);
            insert(2, i2);    insert(3, i3);
            insert(4, i4);    insert(5, i5);
            insert(6, i6);    insert(7, i7);
            insert(8, i8);    insert(9, i9);
            insert(10, i10);  insert(11, i11);
            insert(12, i12);  insert(13, i13);
            insert(14, i14);  insert(15, i15);
            insert(16, i16);  insert(17, i17);
            insert(18, i18);  insert(19, i19);
            insert(20, i20);  insert(21, i21);
            insert(22, i22);  insert(23, i23);
            insert(24, i24);  insert(25, i25);
            insert(26, i26);  insert(27, i27);
            insert(28, i28);  insert(29, i29);
            insert(30, i30);  insert(31, i31);
        }

        UME_FORCE_INLINE SIMDVec_i(
            SCALAR_INT_TYPE i0,  SCALAR_INT_TYPE i1,  SCALAR_INT_TYPE i2,  SCALAR_INT_TYPE i3,  SCALAR_INT_TYPE i4,  SCALAR_INT_TYPE i5,  SCALAR_INT_TYPE i6,  SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8,  SCALAR_INT_TYPE i9,  SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15,
            SCALAR_INT_TYPE i16, SCALAR_INT_TYPE i17, SCALAR_INT_TYPE i18, SCALAR_INT_TYPE i19, SCALAR_INT_TYPE i20, SCALAR_INT_TYPE i21, SCALAR_INT_TYPE i22, SCALAR_INT_TYPE i23,
            SCALAR_INT_TYPE i24, SCALAR_INT_TYPE i25, SCALAR_INT_TYPE i26, SCALAR_INT_TYPE i27, SCALAR_INT_TYPE i28, SCALAR_INT_TYPE i29, SCALAR_INT_TYPE i30, SCALAR_INT_TYPE i31,
            SCALAR_INT_TYPE i32, SCALAR_INT_TYPE i33, SCALAR_INT_TYPE i34, SCALAR_INT_TYPE i35, SCALAR_INT_TYPE i36, SCALAR_INT_TYPE i37, SCALAR_INT_TYPE i38, SCALAR_INT_TYPE i39,
            SCALAR_INT_TYPE i40, SCALAR_INT_TYPE i41, SCALAR_INT_TYPE i42, SCALAR_INT_TYPE i43, SCALAR_INT_TYPE i44, SCALAR_INT_TYPE i45, SCALAR_INT_TYPE i46, SCALAR_INT_TYPE i47,
            SCALAR_INT_TYPE i48, SCALAR_INT_TYPE i49, SCALAR_INT_TYPE i50, SCALAR_INT_TYPE i51, SCALAR_INT_TYPE i52, SCALAR_INT_TYPE i53, SCALAR_INT_TYPE i54, SCALAR_INT_TYPE i55,
            SCALAR_INT_TYPE i56, SCALAR_INT_TYPE i57, SCALAR_INT_TYPE i58, SCALAR_INT_TYPE i59, SCALAR_INT_TYPE i60, SCALAR_INT_TYPE i61, SCALAR_INT_TYPE i62, SCALAR_INT_TYPE i63)
        {
            insert(0, i0);    insert(1, i1);    insert(2, i2);    insert(3, i3);
            insert(4, i4);    insert(5, i5);    insert(6, i6);    insert(7, i7);
            insert(8, i8);    insert(9, i9);    insert(10, i10);  insert(11, i11);
            insert(12, i12);  insert(13, i13);  insert(14, i14);  insert(15, i15);
            insert(16, i16);  insert(17, i17);  insert(18, i18);  insert(19, i19);
            insert(20, i20);  insert(21, i21);  insert(22, i22);  insert(23, i23);
            insert(24, i24);  insert(25, i25);  insert(26, i26);  insert(27, i27);
            insert(28, i28);  insert(29, i29);  insert(30, i30);  insert(31, i31);
            insert(32, i32);  insert(33, i33);  insert(34, i34);  insert(35, i35);
            insert(36, i36);  insert(37, i37);  insert(38, i38);  insert(39, i39);
            insert(40, i40);  insert(41, i41);  insert(42, i42);  insert(43, i43);
            insert(44, i44);  insert(45, i45);  insert(46, i46);  insert(47, i47);
            insert(48, i48);  insert(49, i49);  insert(50, i50);  insert(51, i51);
            insert(52, i52);  insert(53, i53);  insert(54, i54);  insert(55, i55);
            insert(56, i56);  insert(57, i57);  insert(58, i58);  insert(59, i59);
            insert(60, i60);  insert(61, i61);  insert(62, i62);  insert(63, i63);
        }

        UME_FORCE_INLINE SIMDVec_i(
            SCALAR_INT_TYPE i0,   SCALAR_INT_TYPE i1,   SCALAR_INT_TYPE i2,   SCALAR_INT_TYPE i3,   SCALAR_INT_TYPE i4,   SCALAR_INT_TYPE i5,   SCALAR_INT_TYPE i6,   SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8,   SCALAR_INT_TYPE i9,   SCALAR_INT_TYPE i10,  SCALAR_INT_TYPE i11,  SCALAR_INT_TYPE i12,  SCALAR_INT_TYPE i13,  SCALAR_INT_TYPE i14,  SCALAR_INT_TYPE i15,
            SCALAR_INT_TYPE i16,  SCALAR_INT_TYPE i17,  SCALAR_INT_TYPE i18,  SCALAR_INT_TYPE i19,  SCALAR_INT_TYPE i20,  SCALAR_INT_TYPE i21,  SCALAR_INT_TYPE i22,  SCALAR_INT_TYPE i23,
            SCALAR_INT_TYPE i24,  SCALAR_INT_TYPE i25,  SCALAR_INT_TYPE i26,  SCALAR_INT_TYPE i27,  SCALAR_INT_TYPE i28,  SCALAR_INT_TYPE i29,  SCALAR_INT_TYPE i30,  SCALAR_INT_TYPE i31,
            SCALAR_INT_TYPE i32,  SCALAR_INT_TYPE i33,  SCALAR_INT_TYPE i34,  SCALAR_INT_TYPE i35,  SCALAR_INT_TYPE i36,  SCALAR_INT_TYPE i37,  SCALAR_INT_TYPE i38,  SCALAR_INT_TYPE i39,
            SCALAR_INT_TYPE i40,  SCALAR_INT_TYPE i41,  SCALAR_INT_TYPE i42,  SCALAR_INT_TYPE i43,  SCALAR_INT_TYPE i44,  SCALAR_INT_TYPE i45,  SCALAR_INT_TYPE i46,  SCALAR_INT_TYPE i47,
            SCALAR_INT_TYPE i48,  SCALAR_INT_TYPE i49,  SCALAR_INT_TYPE i50,  SCALAR_INT_TYPE i51,  SCALAR_INT_TYPE i52,  SCALAR_INT_TYPE i53,  SCALAR_INT_TYPE i54,  SCALAR_INT_TYPE i55,
            SCALAR_INT_TYPE i56,  SCALAR_INT_TYPE i57,  SCALAR_INT_TYPE i58,  SCALAR_INT_TYPE i59,  SCALAR_INT_TYPE i60,  SCALAR_INT_TYPE i61,  SCALAR_INT_TYPE i62,  SCALAR_INT_TYPE i63,
            SCALAR_INT_TYPE i64,  SCALAR_INT_TYPE i65,  SCALAR_INT_TYPE i66,  SCALAR_INT_TYPE i67,  SCALAR_INT_TYPE i68,  SCALAR_INT_TYPE i69,  SCALAR_INT_TYPE i70,  SCALAR_INT_TYPE i71,
            SCALAR_INT_TYPE i72,  SCALAR_INT_TYPE i73,  SCALAR_INT_TYPE i74,  SCALAR_INT_TYPE i75,  SCALAR_INT_TYPE i76,  SCALAR_INT_TYPE i77,  SCALAR_INT_TYPE i78,  SCALAR_INT_TYPE i79,
            SCALAR_INT_TYPE i80,  SCALAR_INT_TYPE i81,  SCALAR_INT_TYPE i82,  SCALAR_INT_TYPE i83,  SCALAR_INT_TYPE i84,  SCALAR_INT_TYPE i85,  SCALAR_INT_TYPE i86,  SCALAR_INT_TYPE i87,
            SCALAR_INT_TYPE i88,  SCALAR_INT_TYPE i89,  SCALAR_INT_TYPE i90,  SCALAR_INT_TYPE i91,  SCALAR_INT_TYPE i92,  SCALAR_INT_TYPE i93,  SCALAR_INT_TYPE i94,  SCALAR_INT_TYPE i95,
            SCALAR_INT_TYPE i96,  SCALAR_INT_TYPE i97,  SCALAR_INT_TYPE i98,  SCALAR_INT_TYPE i99,  SCALAR_INT_TYPE i100, SCALAR_INT_TYPE i101, SCALAR_INT_TYPE i102, SCALAR_INT_TYPE i103,
            SCALAR_INT_TYPE i104, SCALAR_INT_TYPE i105, SCALAR_INT_TYPE i106, SCALAR_INT_TYPE i107, SCALAR_INT_TYPE i108, SCALAR_INT_TYPE i109, SCALAR_INT_TYPE i110, SCALAR_INT_TYPE i111,
            SCALAR_INT_TYPE i112, SCALAR_INT_TYPE i113, SCALAR_INT_TYPE i114, SCALAR_INT_TYPE i115, SCALAR_INT_TYPE i116, SCALAR_INT_TYPE i117, SCALAR_INT_TYPE i118, SCALAR_INT_TYPE i119,
            SCALAR_INT_TYPE i120, SCALAR_INT_TYPE i121, SCALAR_INT_TYPE i122, SCALAR_INT_TYPE i123, SCALAR_INT_TYPE i124, SCALAR_INT_TYPE i125, SCALAR_INT_TYPE i126, SCALAR_INT_TYPE i127)
        {
            insert(0, i0);    insert(1, i1);    insert(2, i2);    insert(3, i3);
            insert(4, i4);    insert(5, i5);    insert(6, i6);    insert(7, i7);
            insert(8, i8);    insert(9, i9);    insert(10, i10);  insert(11, i11);
            insert(12, i12);  insert(13, i13);  insert(14, i14);  insert(15, i15);
            insert(16, i16);  insert(17, i17);  insert(18, i18);  insert(19, i19);
            insert(20, i20);  insert(21, i21);  insert(22, i22);  insert(23, i23);
            insert(24, i24);  insert(25, i25);  insert(26, i26);  insert(27, i27);
            insert(28, i28);  insert(29, i29);  insert(30, i30);  insert(31, i31);
            insert(32, i32);  insert(33, i33);  insert(34, i34);  insert(35, i35);
            insert(36, i36);  insert(37, i37);  insert(38, i38);  insert(39, i39);
            insert(40, i40);  insert(41, i41);  insert(42, i42);  insert(43, i43);
            insert(44, i44);  insert(45, i45);  insert(46, i46);  insert(47, i47);
            insert(48, i48);  insert(49, i49);  insert(50, i50);  insert(51, i51);
            insert(52, i52);  insert(53, i53);  insert(54, i54);  insert(55, i55);
            insert(56, i56);  insert(57, i57);  insert(58, i58);  insert(59, i59);
            insert(60, i60);  insert(61, i61);  insert(62, i62);  insert(63, i63);
            insert(64,  i64);   insert(65, i65);    insert(66, i66);    insert(67, i67);
            insert(68,  i68);   insert(69, i69);    insert(70, i70);    insert(71, i71);
            insert(72,  i72);   insert(73, i73);    insert(74, i74);    insert(75, i75);
            insert(76,  i76);   insert(77, i77);    insert(78, i78);    insert(79, i79);
            insert(80,  i80);   insert(81, i81);    insert(82, i82);    insert(83, i83);
            insert(84,  i84);   insert(85, i85);    insert(86, i86);    insert(87, i87);
            insert(88,  i88);   insert(89, i89);    insert(90, i90);    insert(91, i91);
            insert(92,  i92);   insert(93, i93);    insert(94, i94);    insert(95, i95);
            insert(96,  i96);   insert(97, i97);    insert(98, i98);    insert(99, i99);
            insert(100, i100);  insert(101, i101);  insert(102, i102);  insert(103, i103);
            insert(104, i104);  insert(105, i105);  insert(106, i106);  insert(107, i107);
            insert(108, i108);  insert(109, i109);  insert(110, i110);  insert(111, i111);
            insert(112, i112);  insert(113, i113);  insert(114, i114);  insert(115, i115);
            insert(116, i116);  insert(117, i117);  insert(118, i118);  insert(119, i119);
            insert(120, i120);  insert(121, i121);  insert(122, i122);  insert(123, i123);
            insert(124, i124);  insert(125, i125);  insert(126, i126);  insert(127, i127);
        }

        // EXTRACT
        UME_FORCE_INLINE SCALAR_INT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, SCALAR_INT_TYPE> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, SCALAR_INT_TYPE>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, SCALAR_INT_TYPE, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_i, SCALAR_INT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, SCALAR_INT_TYPE, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_i, SCALAR_INT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & src) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_src_ptr = &src.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & src) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_src_ptr = &src.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SCALAR_INT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = b;
            }
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(SCALAR_INT_TYPE const *p) {
            SCALAR_INT_TYPE *local_ptr = &mVec[0];
            SCALAR_INT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE const *p) {
            SCALAR_INT_TYPE *local_ptr = &mVec[0];
            SCALAR_INT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SCALAR_INT_TYPE const *p) {
            SCALAR_INT_TYPE *local_ptr = &mVec[0];
            SCALAR_INT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE const *p) {
            SCALAR_INT_TYPE *local_ptr = &mVec[0];
            SCALAR_INT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // STORE
        UME_FORCE_INLINE SCALAR_INT_TYPE* store(SCALAR_INT_TYPE* p) const {
            SCALAR_INT_TYPE const *local_ptr = &mVec[0];
            SCALAR_INT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE SCALAR_INT_TYPE* store(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE* p) const {
            SCALAR_INT_TYPE const *local_ptr = &mVec[0];
            SCALAR_INT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // STOREA
        UME_FORCE_INLINE SCALAR_INT_TYPE* storea(SCALAR_INT_TYPE* p) const {
            SCALAR_INT_TYPE const *local_ptr = &mVec[0];
            SCALAR_INT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE SCALAR_INT_TYPE* storea(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE* p) const {
            SCALAR_INT_TYPE const *local_ptr = &mVec[0];
            SCALAR_INT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE *retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const *local_ptr = &mVec[0];
            SCALAR_INT_TYPE const *local_b_ptr = &b.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = local_b_ptr[i];
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE const *local_ptr = &mVec[0];
            SCALAR_INT_TYPE *retval_ptr = &retval.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = b;
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SCALAR_INT_TYPE b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] += b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SCALAR_INT_TYPE b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] += b;
            }
            return *this;
        }

        // SADDV
        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA

        // POSTINC
        UME_FORCE_INLINE SIMDVec_i postinc() {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]++;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<VEC_LEN> const & mask) {
             SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]++;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                ++local_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) ++local_ptr[i];
            }
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SCALAR_INT_TYPE b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] -= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SCALAR_INT_TYPE b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] -= b;
            }
            return *this;
        }

        // SSUBV
        // MSSUBV
        // SSUBS
        // MSSUBS
        // SSUBVA
        // MSSUBVA
        // SSUBSA
        // MSSUBSA

        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b - local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b - local_ptr[i];
                else local_ptr[i] = b;
            }
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]--;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<VEC_LEN> const & mask) {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]--;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                --local_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) --local_ptr[i];
            }
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SCALAR_INT_TYPE b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] *= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SCALAR_INT_TYPE b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= b;
            }
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_i div(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SCALAR_INT_TYPE b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] /= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SCALAR_INT_TYPE b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= b;
            }
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_i rcp() const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = SCALAR_INT_TYPE(1.0f) / local_ptr[i];
            }
            return retval;
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_i rcp(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = SCALAR_INT_TYPE(1.0f) / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_i rcp(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b / local_ptr[i];
            }
            return retval;
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_i rcp(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_i & rcpa() {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = SCALAR_INT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = SCALAR_INT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpeq(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpeq(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator== (SCALAR_INT_TYPE b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpne(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpne(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator!= (SCALAR_INT_TYPE b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpgt(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpgt(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator> (SCALAR_INT_TYPE b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmplt(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmplt(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator< (SCALAR_INT_TYPE b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpge(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpge(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator>= (SCALAR_INT_TYPE b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmple(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmple(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator<= (SCALAR_INT_TYPE b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_i const & b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool local_mask_ptr[VEC_LEN];
            bool retval = true;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_mask_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            #pragma omp simd reduction(&&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval && local_mask_ptr[i];
            }
            return retval;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool local_mask_ptr[VEC_LEN];
            bool retval = true;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_mask_ptr[i] = local_ptr[i] == b;
            }
            #pragma omp simd reduction(&&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval && local_mask_ptr[i];
            }
            return retval;
        }
        // UNIQUE
        // TODO

        // HADD
        UME_FORCE_INLINE SCALAR_INT_TYPE hadd() const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0.0f);
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE SCALAR_INT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0.0f);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE SCALAR_INT_TYPE hadd(SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADDS
        UME_FORCE_INLINE SCALAR_INT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HMUL
        UME_FORCE_INLINE SCALAR_INT_TYPE hmul() const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(1.0f);
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;        }
        // MHMUL
        UME_FORCE_INLINE SCALAR_INT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(1.0f);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE SCALAR_INT_TYPE hmul(SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;
        }
        // MHMULS
        UME_FORCE_INLINE SCALAR_INT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
            }
            return retval;
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
            }
            return retval;
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_INT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = b;
            }
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = b;
            }
            return *this;
        }

        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] & local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] & local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] & b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SCALAR_INT_TYPE b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] & b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] &= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] &= local_b_ptr[i];
            }
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] &= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] &= b;
            }
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] | local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] | local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] | b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SCALAR_INT_TYPE b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] | b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] |= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] |= local_b_ptr[i];
            }
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] |= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SCALAR_INT_TYPE b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] |= b;
            }
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] ^ local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] ^ local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] ^ b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SCALAR_INT_TYPE b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] ^ b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] ^= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] ^= local_b_ptr[i];
            }
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] ^= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SCALAR_INT_TYPE b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] ^= b;
            }
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = ~local_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = ~local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = ~local_ptr[i];
            }
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = ~local_ptr[i];
            }
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE SCALAR_INT_TYPE hband() const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0xFFFFFFFFFFFFFFFF);
            #pragma omp simd reduction(&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval & local_ptr[i];
            }
            return retval;
        }
        // MHBAND
        UME_FORCE_INLINE SCALAR_INT_TYPE hband(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0xFFFFFFFFFFFFFFFF);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0xFFFFFFFFFFFFFFFF);
            }
            #pragma omp simd reduction(&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval & masked_copy[i];
            }
            return retval;
        }
        // HBANDS
        UME_FORCE_INLINE SCALAR_INT_TYPE hband(SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd reduction(&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval & local_ptr[i];
            }
            return retval;
        }
        // MHBANDS
        UME_FORCE_INLINE SCALAR_INT_TYPE hband(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0xFFFFFFFFFFFFFFFF);
            }
            #pragma omp simd reduction(&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval & masked_copy[i];
            }
            return retval;
        }
        // HBOR
        UME_FORCE_INLINE SCALAR_INT_TYPE hbor() const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0);
            #pragma omp simd reduction(|:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval | local_ptr[i];
            }
            return retval;
        }
        // MHBOR
        UME_FORCE_INLINE SCALAR_INT_TYPE hbor(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0);
            }
            #pragma omp simd reduction(&:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval | masked_copy[i];
            }
            return retval;
        }
        // HBORS
        UME_FORCE_INLINE SCALAR_INT_TYPE hbor(SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd reduction(|:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval | local_ptr[i];
            }
            return retval;
        }
        // MHBORS
        UME_FORCE_INLINE SCALAR_INT_TYPE hbor(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0);
            }
            #pragma omp simd reduction(|:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval | masked_copy[i];
            }
            return retval;
        }
        // HBXOR
        UME_FORCE_INLINE SCALAR_INT_TYPE hbxor() const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0);
            #pragma omp simd reduction(^:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ local_ptr[i];
            }
            return retval;
        }
        // MHBXOR
        UME_FORCE_INLINE SCALAR_INT_TYPE hbxor(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = SCALAR_INT_TYPE(0);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0);
            }
            #pragma omp simd reduction(^:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ masked_copy[i];
            }
            return retval;
        }
        // HBXORS
        UME_FORCE_INLINE SCALAR_INT_TYPE hbxor(SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd reduction(^:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ local_ptr[i];
            }
            return retval;
        }
        // MHBXORS
        UME_FORCE_INLINE SCALAR_INT_TYPE hbxor(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_INT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_INT_TYPE(0);
            }
            #pragma omp simd reduction(^:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval ^ masked_copy[i];
            }
            return retval;
        }
        // REMV
        UME_FORCE_INLINE SIMDVec_i rem(SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] % local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator% (SIMDVec_i const & b) const {
            return rem(b);
        }
        // MREMV
        UME_FORCE_INLINE SIMDVec_i rem(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] % local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // REMS
        UME_FORCE_INLINE SIMDVec_i rem(SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] % b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator% (SCALAR_INT_TYPE b) const {
            return rem(b);
        }
        // MREMS
        UME_FORCE_INLINE SIMDVec_i rem(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] % b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // REMVA
        UME_FORCE_INLINE SIMDVec_i & rema(SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] %= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator%= (SIMDVec_i const & b) {
            return rema(b);
        }
        // MREMVA
        UME_FORCE_INLINE SIMDVec_i & rema(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_i const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] %= local_b_ptr[i];
            }
            return *this;
        }
        // REMSA
        UME_FORCE_INLINE SIMDVec_i & rema(SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] %= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator%= (SCALAR_INT_TYPE b) {
            return rema(b);
        }
        // MREMSA
        UME_FORCE_INLINE SIMDVec_i & rema(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] %= b;
            }
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> land(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] && local_b_ptr[i];
            }
            return retval;
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> land(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] && b;
            }
            return retval;
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> lor(SIMDVec_i const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_INT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] || local_b_ptr[i];
            }
            return retval;
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> lor(SCALAR_INT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            bool * local_retval_ptr = &retval.mMask[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] || b;
            }
            return retval;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SCALAR_INT_TYPE const * baseAddr, SCALAR_UINT_TYPE const * indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE const * baseAddr, SCALAR_UINT_TYPE const * indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SCALAR_INT_TYPE const * baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE const * baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE SCALAR_INT_TYPE* scatter(SCALAR_INT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE SCALAR_INT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE SCALAR_INT_TYPE* scatter(SCALAR_INT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE SCALAR_INT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_INT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // LSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] << local_b_ptr[i];
            }
            return retval;
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] << local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_i lsh(SCALAR_UINT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] << b;
            }
            return retval;
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] << b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] <<= local_b_ptr[i];
            }
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] <<= local_b_ptr[i];
            }
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(SCALAR_UINT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] <<= b;
            }
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] <<= b;
            }
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >> local_b_ptr[i];
            }
            return retval;
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] >> local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_i rsh(SCALAR_UINT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >> b;
            }
            return retval;
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] >> b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] >>= local_b_ptr[i];
            }
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            SCALAR_UINT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] >>= local_b_ptr[i];
            }
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(SCALAR_UINT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] >>= b;
            }
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<VEC_LEN> const & mask, SCALAR_UINT_TYPE b) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] >>= b;
            }
            return *this;
        }
        // ROLV
        // MROLV
        // ROLS
        // MROLS
        // ROLVA
        // MROLVA
        // ROLSA
        // MROLSA
        // RORV
        // MRORV
        // RORS
        // MRORS
        // RORVA
        // MRORVA
        // RORSA
        // MRORSA
        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = -local_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = -local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] >= 0 ) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = -local_ptr[i];
            }
            return retval;
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_i retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_INT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < 0;
                bool cond = local_mask_ptr[i] && predicate;

                if(cond) local_retval_ptr[i] = -local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < 0 ) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_INT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < 0;
                bool cond = local_mask_ptr[i] && predicate;

                if(cond) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }

        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<SCALAR_INT_LOWER_PRECISION, VEC_LEN>() const;
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_i<SCALAR_INT_HIGHER_PRECISION, VEC_LEN>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>() const;
    };

    // SIMD NullTypes. These are used whenever a terminating
    // scalar type is used as a creator function for SIMD type.
    // These types cannot be instantiated, but are necessary for 
    // typeset to be consistent.
    template<>
    class SIMDVec_i<NullType<1>, 1>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 2>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 4>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 8>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 16>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 32>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 64>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<1>, 128>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 1>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 2>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 4>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 8>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 16>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 32>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 64>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<2>, 128>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 1>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 2>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 4>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 8>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 16>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 32>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 64>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };

    template<>
    class SIMDVec_i<NullType<3>, 128>
    {
    private:
        SIMDVec_i() {}
        ~SIMDVec_i() {}
    };
}
}

#endif
