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

#ifndef UME_SIMD_VEC_INT_PROTOTYPE_H_
#define UME_SIMD_VEC_INT_PROTOTYPE_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

#include "../UMESimdMaskKNC.h"
#include "../UMESimdSwizzleKNC.h"
#include "../UMESimdVecUintKNC.h"

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
        typedef SIMDVec_u<uint8_t, 1>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDVecSwizzle<1>       SWIZZLE_MASK_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 2> {
        typedef SIMDVec_i<int8_t, 1>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 2>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDVecSwizzle<2>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 1> {
        typedef SIMDVec_u<uint16_t, 1>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDVecSwizzle<1>       SWIZZLE_MASK_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 4> {
        typedef SIMDVec_i<int8_t, 2>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 4>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDVecSwizzle<4>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 2> {
        typedef SIMDVec_i<int16_t, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 2>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDVecSwizzle<2>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 1> {
        typedef SIMDVec_u<uint32_t, 1>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDVecSwizzle<1>       SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 8> {
        typedef SIMDVec_i<int8_t, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 8>   VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDVecSwizzle<8>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 4> {
        typedef SIMDVec_i<int16_t, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 4>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDVecSwizzle<4>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 2> {
        typedef SIMDVec_i<int32_t, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 2>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDVecSwizzle<2>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 1> {
        typedef SIMDVec_u<uint64_t, 1>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDVecSwizzle<1>       SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 16> {
        typedef SIMDVec_i<int8_t, 8>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 16>  VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDVecSwizzle<16>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 8> {
        typedef SIMDVec_i<int16_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 8>  VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDVecSwizzle<8>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 4> {
        typedef SIMDVec_i<int32_t, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 4>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDVecSwizzle<4>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 2> {
        typedef SIMDVec_i<int64_t, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 2>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDVecSwizzle<2>       SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 32> {
        typedef SIMDVec_i<int8_t, 16>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 32>  VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDVecSwizzle<32>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 16> {
        typedef SIMDVec_i<int16_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDVecSwizzle<16>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 8> {
        typedef SIMDVec_i<int32_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 8>  VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDVecSwizzle<8>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 4> {
        typedef SIMDVec_i<int64_t, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 4>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDVecSwizzle<4>       SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 64> {
        typedef SIMDVec_i<int8_t, 32>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 64>  VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<64>         MASK_TYPE;
        typedef SIMDVecSwizzle<64>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 32> {
        typedef SIMDVec_i<int16_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDVecSwizzle<32>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 16> {
        typedef SIMDVec_i<int32_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDVecSwizzle<16>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 8> {
        typedef SIMDVec_i<int64_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 8>  VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDVecSwizzle<8>       SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVec_i_traits<int8_t, 128> {
        typedef SIMDVec_i<int8_t, 64>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t                 SCALAR_UINT_TYPE;
        typedef SIMDVecMask<128>        MASK_TYPE;
        typedef SIMDVecSwizzle<128>     SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int16_t, 64> {
        typedef SIMDVec_i<int16_t, 32>  HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<64>         MASK_TYPE;
        typedef SIMDVecSwizzle<64>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int32_t, 32> {
        typedef SIMDVec_i<int32_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDVecSwizzle<32>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_i_traits<int64_t, 16> {
        typedef SIMDVec_i<int64_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 16> VEC_UINT;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDVecSwizzle<16>      SWIZZLE_MASK_TYPE;
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
    class SIMDVec_i final :
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
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT           VEC_UINT;
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE          MASK_TYPE;

    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVec_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_i(SCALAR_INT_TYPE const * p) { this->load(p); }

        inline SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        inline SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7)
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
        }

        inline SIMDVec_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15,
            SCALAR_INT_TYPE i16, SCALAR_INT_TYPE i17, SCALAR_INT_TYPE i18, SCALAR_INT_TYPE i19, SCALAR_INT_TYPE i20, SCALAR_INT_TYPE i21, SCALAR_INT_TYPE i22, SCALAR_INT_TYPE i23,
            SCALAR_INT_TYPE i24, SCALAR_INT_TYPE i25, SCALAR_INT_TYPE i26, SCALAR_INT_TYPE i27, SCALAR_INT_TYPE i28, SCALAR_INT_TYPE i29, SCALAR_INT_TYPE i30, SCALAR_INT_TYPE i31)
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

        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_i, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_i, MASK_TYPE>(mask, static_cast<SIMDVec_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
            for (uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ***************************************************************************
    // *
    // *    Partial specialization of signed integer SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_INT_TYPE>
    class SIMDVec_i<SCALAR_INT_TYPE, 1> final :
        public SIMDVecSignedInterface<
        SIMDVec_i<SCALAR_INT_TYPE, 1>,
        typename SIMDVec_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT,
        SCALAR_INT_TYPE,
        1,
        typename SIMDVec_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE,
        typename SIMDVec_i_traits<SCALAR_INT_TYPE, 1>::MASK_TYPE,
        typename SIMDVec_i_traits<SCALAR_INT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, 1> VEC_EMU_REG;

        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVec_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT             VEC_UINT;

    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVec_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR
        inline explicit SIMDVec_i(SCALAR_INT_TYPE const *p) { this->load(p); }

        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<1>>(mask, static_cast<SIMDVec_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVec_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVec_u<SCALAR_UINT_TYPE, 1> retval(mVec[0]);
            return retval;
        }

        // UNIQUE
        inline bool unique() const {
            return true;
        }
    };

}
}

#endif
