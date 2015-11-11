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

#ifndef UME_SIMD_VEC_INT_KNC_H_
#define UME_SIMD_VEC_INT_KNC_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

#include "UMESimdMaskKNC.h"
#include "UMESimdSwizzleKNC.h"
#include "UMESimdVecUintKNC.h"

namespace UME {
namespace SIMD {
    // ********************************************************************************************
    // SIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNC_i_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 1> {
        typedef SIMDVecKNC_u<uint8_t, 1> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                MASK_TYPE;
        typedef SIMDSwizzle1             SWIZZLE_MASK_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 2> {
        typedef SIMDVecKNC_i<int8_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 2> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                MASK_TYPE;
        typedef SIMDSwizzle2             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 1> {
        typedef SIMDVecKNC_u<uint16_t, 1> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 4> {
        typedef SIMDVecKNC_i<int8_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 4> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                MASK_TYPE;
        typedef SIMDSwizzle4             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 2> {
        typedef SIMDVecKNC_i<int16_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint16_t, 2> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 1> {
        typedef SIMDVecKNC_u<uint32_t, 1> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 8> {
        typedef SIMDVecKNC_i<int8_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 8> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                MASK_TYPE;
        typedef SIMDSwizzle8             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 4> {
        typedef SIMDVecKNC_i<int16_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint16_t, 4> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 2> {
        typedef SIMDVecKNC_i<int32_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 2> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 1> {
        typedef SIMDVecKNC_u<uint64_t, 1> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 16> {
        typedef SIMDVecKNC_i<int8_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 16> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 8> {
        typedef SIMDVecKNC_i<int8_t, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint16_t, 8> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 4> {
        typedef SIMDVecKNC_i<int32_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 4> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 2> {
        typedef SIMDVecKNC_i<int64_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 2> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 32> {
        typedef SIMDVecKNC_i<int8_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 32> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                MASK_TYPE;
        typedef SIMDSwizzle32             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 16> {
        typedef SIMDVecKNC_i<int16_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 8> {
        typedef SIMDVecKNC_i<int32_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 8> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 4> {
        typedef SIMDVecKNC_i<int64_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 4> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 64> {
        typedef SIMDVecKNC_i<int8_t, 32>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 64> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask64                MASK_TYPE;
        typedef SIMDSwizzle64             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 32> {
        typedef SIMDVecKNC_i<int16_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 16> {
        typedef SIMDVecKNC_i<int32_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 8> {
        typedef SIMDVecKNC_i<int64_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 8> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecKNC_i_traits<int8_t, 128> {
        typedef SIMDVecKNC_i<int8_t, 64>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t                    SCALAR_UINT_TYPE;
        typedef SIMDMask128                MASK_TYPE;
        typedef SIMDSwizzle128             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int16_t, 64> {
        typedef SIMDVecKNC_i<int16_t, 32>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask64                 MASK_TYPE;
        typedef SIMDSwizzle64              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int32_t, 32> {
        typedef SIMDVecKNC_i<int32_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_i_traits<int64_t, 16> {
        typedef SIMDVecKNC_i<int32_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 16> VEC_UINT;
        typedef uint64_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
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
    class SIMDVecKNC_i final :
        public SIMDVecSignedInterface<
        SIMDVecKNC_i<SCALAR_INT_TYPE, VEC_LEN>,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT,
        SCALAR_INT_TYPE,
        VEC_LEN,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
        SIMDVecKNC_i<SCALAR_INT_TYPE, VEC_LEN>,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, VEC_LEN>                            VEC_EMU_REG;

        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE    SCALAR_UINT_TYPE;
        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT            VEC_UINT;
        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE           MASK_TYPE;

        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecKNC_i() : mVec() {};

        inline explicit SIMDVecKNC_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_i(SCALAR_INT_TYPE const * p) { this->load(p); }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7)
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
        }

        inline SIMDVecKNC_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
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
        inline IntermediateMask<SIMDVecKNC_i, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVecKNC_i, MASK_TYPE>(mask, static_cast<SIMDVecKNC_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecKNC_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecKNC_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVecKNC_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
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
    class SIMDVecKNC_i<SCALAR_INT_TYPE, 1> final :
        public SIMDVecSignedInterface<
        SIMDVecKNC_i<SCALAR_INT_TYPE, 1>,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT,
        SCALAR_INT_TYPE,
        1,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, 1>::MASK_TYPE,
        typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, 1>                            VEC_EMU_REG;

        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVecKNC_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT             VEC_UINT;

        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecKNC_i() : mVec() {};

        inline explicit SIMDVecKNC_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_i(SCALAR_INT_TYPE const * p) { this->load(p); }

        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_i, SIMDMask1> operator[] (SIMDMask1 & mask) {
            return IntermediateMask<SIMDVecKNC_i, SIMDMask1>(mask, static_cast<SIMDVecKNC_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecKNC_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecKNC_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVecKNC_u<SCALAR_UINT_TYPE, 1> retval(mVec[0]);
            return retval;
        }
    };

    // ********************************************************************************************
    // SIGNED INTEGER VECTOR specializations
    // ********************************************************************************************

    template<>
    class SIMDVecKNC_i<int32_t, 16> :
        public SIMDVecSignedInterface<
        SIMDVecKNC_i<int32_t, 16>,
        SIMDVecKNC_u<uint32_t, 16>,
        int32_t,
        16,
        uint32_t,
        SIMDMask16,
        SIMDSwizzle16>,
        public SIMDVecPackableInterface<
        SIMDVecKNC_i<int32_t, 16>,
        typename SIMDVecKNC_i_traits<int32_t, 16>::HALF_LEN_VEC_TYPE>
    {
        friend class SIMDVecKNC_u<uint32_t, 16>;
        friend class SIMDVecKNC_f<float, 16>;
        friend class SIMDVecKNC_f<double, 16>;

    private:
        __m512i mVec;

        inline explicit SIMDVecKNC_i(__m512i & x) {
            this->mVec = x;
        }
    public:
        inline SIMDVecKNC_i() {};

        inline explicit SIMDVecKNC_i(int32_t i) {
            mVec = _mm512_set1_epi32(i);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_i(int32_t const * p) { this->load(p); }


        inline SIMDVecKNC_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7,
            int32_t i8, int32_t i9, int32_t i10, int32_t i11,
            int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVec = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7,
                i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline int32_t extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) int32_t raw[16];
            _mm512_store_si512(raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_i, SIMDMask16> operator[] (SIMDMask16 & mask) {
            return IntermediateMask<SIMDVecKNC_i, SIMDMask16>(mask, static_cast<SIMDVecKNC_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecKNC_i & insert(uint32_t index, int32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(64) int32_t raw[16];
            _mm512_store_si512(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512(raw);
            return *this;
        }

        // 1. Base vector
        // ASSIGNV
        inline SIMDVecKNC_i & assign(SIMDVecKNC_i const & src) {
            mVec = src.mVec;
            return *this;
        }
        // MASSIGNV
        inline SIMDVecKNC_i & assign(SIMDMask16 const & mask, SIMDVecKNC_i const & src) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, src.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVecKNC_i & assign(int32_t value) {
            mVec = _mm512_set1_epi32(value);
            return *this;
        }
        // MASSIGNS
        inline SIMDVecKNC_i & assign(SIMDMask16 const & mask, int32_t value) {
            mVec = _mm512_mask_mov_epi32(mVec, mask.mMask, _mm512_set1_epi32(value));
            return *this;
        }

        // PREFETCH0
        static inline void prefetch0(int32_t const *p) {
            _mm_prefetch((char *)p, _MM_HINT_T0);
        }
        // PREFETCH1
        static inline void prefetch1(int32_t const *p) {
            _mm_prefetch((char *)p, _MM_HINT_T1);
        }
        // PREFETCH2
        static inline void prefetch2(int32_t const *p) {
            _mm_prefetch((char *)p, _MM_HINT_T2);
        }
        // LOAD
        inline SIMDVecKNC_i & load(int32_t const *p) {
            if ((uint64_t(p) % 64) == 0) {
                mVec = _mm512_load_epi32(p);
            }
            else {
                alignas(64) int32_t raw[16];
                memcpy(raw, p, 16 * sizeof(int32_t));
                mVec = _mm512_load_epi32(raw);
            }
            return *this;
        }
        // MLOAD
        inline SIMDVecKNC_i & load(SIMDMask16 const & mask, int32_t const * p) {
            if ((uint64_t(p) % 64) == 0) {
                mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
            }
            else {
                alignas(64) int32_t raw[16];
                memcpy(raw, p, 16 * sizeof(int32_t));
                mVec = _mm512_mask_load_epi32(mVec, mask.mMask, raw);
            }
            return *this;
        }
        // LOADA
        inline SIMDVecKNC_i & loada(int32_t const * p) {
            mVec = _mm512_load_epi32(p);
        }
        // MLOADA
        inline SIMDVecKNC_i & loada(SIMDMask16 const & mask, int32_t const *p) {
            mVec = _mm512_mask_load_epi32(mVec, mask.mMask, p);
        }
        // STORE
        inline int32_t* store(int32_t* p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_epi32(p, mVec);
            }
            else {
                alignas(64) int32_t raw[16];
                _mm512_store_epi32(raw, mVec);
                memcpy(p, raw, 16 * sizeof(int32_t));
                return p;
            }
        }
        // MSTORE
        inline int32_t* store(SIMDMask16 const & mask, int32_t* p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_epi32(p, mVec);
            }
            else {
                alignas(64) int32_t raw[16];
                _mm512_store_epi32(raw, mVec);
                memcpy(p, raw, 16 * sizeof(int32_t));
                return p;
            }
        }
        // STOREA
        inline int32_t* storea(int32_t* p) {
            _mm512_store_epi32(p, mVec);
            return p;
        }
        // MSTOREA
        inline int32_t* storea(SIMDMask16 const & mask, int32_t* p) {
            _mm512_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVecKNC_i add(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }

        inline SIMDVecKNC_i operator+ (SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MADDV
        inline SIMDVecKNC_i add(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // ADDS
        inline SIMDVecKNC_i add(int32_t b) const {
            __m512i t0 = _mm512_add_epi32(mVec, _mm512_set1_epi32(b));
            return SIMDVecKNC_i(t0);
        }
        // MADDS
        inline SIMDVecKNC_i add(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // ADDVA
        inline SIMDVecKNC_i & adda(SIMDVecKNC_i const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }

        inline SIMDVecKNC_i & operator+= (SIMDVecKNC_i const & b) {
            mVec = _mm512_add_epi32(mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVecKNC_i & adda(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        inline SIMDVecKNC_i & adda(int32_t b) {
            mVec = _mm512_add_epi32(mVec, _mm512_set1_epi32(b));
            return *this;
        }
        // MADDSA
        inline SIMDVecKNC_i & adda(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
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
        inline SIMDVecKNC_i postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        inline SIMDVecKNC_i operator++ (int) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MPOSTINC
        inline SIMDVecKNC_i postinc(SIMDMask16 const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // PREFINC
        inline SIMDVecKNC_i & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVecKNC_i & operator++ () {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_add_epi32(mVec, t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVecKNC_i & prefinc(SIMDMask16 const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        inline SIMDVecKNC_i sub(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MSUBV
        inline SIMDVecKNC_i sub(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // SUBS
        inline SIMDVecKNC_i sub(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sub_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MSUBS
        inline SIMDVecKNC_i sub(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // SUBVA
        inline SIMDVecKNC_i & suba(SIMDVecKNC_i const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }

        inline SIMDVecKNC_i & operator-= (SIMDVecKNC_i const & b) {
            mVec = _mm512_sub_epi32(mVec, b.mVec);
            return *this;
        }
        // MSUBVA
        inline SIMDVecKNC_i & suba(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVecKNC_i & suba(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        // MSUBSA
        inline SIMDVecKNC_i & suba(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
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
        inline SIMDVecKNC_i subfrom(SIMDVecKNC_i const & a) const {
            __m512i t0 = _mm512_sub_epi32(a.mVec, mVec);
            return SIMDVecKNC_i(t0);
        }
        // MSUBFROMV
        inline SIMDVecKNC_i subfrom(SIMDMask16 const & mask, SIMDVecKNC_i const & a) const {
            __m512i t0 = _mm512_mask_sub_epi32(a.mVec, mask.mMask, a.mVec, mVec);
            return SIMDVecKNC_i(t0);
        }
        // SUBFROMS
        inline SIMDVecKNC_i subfrom(int32_t a) const {
            __m512i t0 = _mm512_set1_epi32(a);
            __m512i t1 = _mm512_sub_epi32(t0, mVec);
            return SIMDVecKNC_i(t1);
        }
        // MSUBFROMS
        inline SIMDVecKNC_i subfrom(SIMDMask16 const & mask, int32_t a) const {
            __m512i t0 = _mm512_set1_epi32(a);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVecKNC_i(t1);
        }
        // SUBFROMVA
        inline SIMDVecKNC_i & subfroma(SIMDVecKNC_i const & a) {
            mVec = _mm512_sub_epi32(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVecKNC_i & subfroma(SIMDMask16 const & mask, SIMDVecKNC_i const & a) {
            mVec = _mm512_mask_sub_epi32(a.mVec, mask.mMask, a.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVecKNC_i & subfroma(int32_t a) {
            __m512i t0 = _mm512_set1_epi32(a);
            mVec = _mm512_sub_epi32(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVecKNC_i & subfroma(SIMDMask16 const & mask, int32_t a) {
            __m512i t0 = _mm512_set1_epi32(a);
            mVec = _mm512_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }

        // POSTDEC
        inline SIMDVecKNC_i postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        inline SIMDVecKNC_i operator-- (int) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MPOSTDEC
        inline SIMDVecKNC_i postdec(SIMDMask16 const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec;
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // PREFDEC
        inline SIMDVecKNC_i & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        inline SIMDVecKNC_i & operator-- () {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_sub_epi32(mVec, t0);
            return *this;
        }
        // MPREFDEC
        inline SIMDVecKNC_i & prefdec(SIMDMask16 const & mask) {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        inline SIMDVecKNC_i mul(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        inline SIMDVecKNC_i operator* (SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MMULV
        inline SIMDVecKNC_i mul(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MULS
        inline SIMDVecKNC_i mul(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mullo_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MMULS
        inline SIMDVecKNC_i mul(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MULVA
        inline SIMDVecKNC_i & mula(SIMDVecKNC_i const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_i & operator*= (SIMDVecKNC_i const & b) {
            mVec = _mm512_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        // MMULVA
        inline SIMDVecKNC_i & mula(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVecKNC_i & mula(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mullo_epi32(mVec, t0);
            return *this;
        }
        // MMULSA
        inline SIMDVecKNC_i mula(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        // MDIVV
        // DIVS
        // MDIVS
        // DIVVA
        // MDIVVA
        // DIVSA
        // MDIVSA
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        inline SIMDMask16 cmpeq(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        inline SIMDMask16 operator== (SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPEQS
        inline SIMDMask16 cmpeq(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, t0);
            return SIMDMask16(m0);
        }
        // CMPNEV
        inline SIMDMask16 cmpne(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        inline SIMDMask16 operator!= (SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPNES
        inline SIMDMask16 cmpne(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec, t0);
            return SIMDMask16(m0);
        }
        // CMPGTV
        inline SIMDMask16 cmpgt(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        inline SIMDMask16 operator> (SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPGTS
        inline SIMDMask16 cmpgt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec, t0);
            return SIMDMask16(m0);
        }
        // CMPLTV
        inline SIMDMask16 cmplt(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        inline SIMDMask16 operator< (SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPLTS
        inline SIMDMask16 cmplt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec, t0);
            return SIMDMask16(m0);
        }
        // CMPGEV
        inline SIMDMask16 cmpge(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        inline SIMDMask16 operator>= (SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPGES
        inline SIMDMask16 cmpge(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec, t0);
            return SIMDMask16(m0);
        }
        // CMPLEV
        inline SIMDMask16 cmple(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        inline SIMDMask16 operator<= (SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPLES
        inline SIMDMask16 cmple(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec, t0);
            return SIMDMask16(m0);
        }
        // CMPEV
        inline bool cmpe(SIMDVecKNC_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, b.mVec);
            return m0 == 0xFFFF;
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec, t0);
            return m0 == 0xFFFF;
        }
        // BLENDV
        // BLENDS
        // HADD
        inline int32_t hadd() const {
            return _mm512_reduce_add_epi32(mVec);
        }
        // MHADD
        inline int32_t hadd(SIMDMask16 const & mask) const {
            return _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            int32_t t0 = _mm512_reduce_add_epi32(mVec);
            return t0 + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDMask16 const & mask, int32_t b) const {
            int32_t t0 = _mm512_mask_reduce_add_epi32(mask.mMask, mVec);
            return t0 + b;
        }
        // HMUL
        inline int32_t hmul() const {
            return _mm512_reduce_mul_epi32(mVec);
        }
        // MHMUL
        inline int32_t hmul(SIMDMask16 const & mask) const {
            return _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
        }
        // HMULS
        inline int32_t hmul(int32_t a) const {
            int32_t t0 = _mm512_reduce_mul_epi32(mVec);
            return a + t0;
        }
        // MHMULS
        inline int32_t hmul(SIMDMask16 const & mask, int32_t a) const {
            int32_t t0 = _mm512_mask_reduce_mul_epi32(mask.mMask, mVec);
            return a + t0;
        }

        // FMULADDV
        inline SIMDVecKNC_i fmuladd(SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_fmadd_epi32(mVec, b.mVec, c.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MFMULADDV
        inline SIMDVecKNC_i fmuladd(SIMDMask16 const & mask, SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_mask_fmadd_epi32(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVecKNC_i(t0);
        }
        // FMULSUBV
        inline SIMDVecKNC_i fmulsub(SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_sub_epi32(t0, c.mVec);
            return SIMDVecKNC_i(t1);
        }
        // MFMULSUBV
        inline SIMDVecKNC_i fmulsub(SIMDMask16 const & mask, SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVecKNC_i(t1);
        }
        // FADDMULV
        inline SIMDVecKNC_i faddmul(SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVecKNC_i(t1);
        }
        // MFADDMULV
        inline SIMDVecKNC_i faddmul(SIMDMask16 const & mask, SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVecKNC_i(t1);
        }
        // FSUBMULV
        inline SIMDVecKNC_i fsubmul(SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec, b.mVec);
            __m512i t1 = _mm512_mullo_epi32(t0, c.mVec);
            return SIMDVecKNC_i(t1);
        }
        // MFSUBMULV
        inline SIMDVecKNC_i fsubmul(SIMDMask16 const & mask, SIMDVecKNC_i const & b, SIMDVecKNC_i const & c) const {
            __m512i t0 = _mm512_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m512i t1 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVecKNC_i(t1);
        }

        // MAXV
        inline SIMDVecKNC_i max(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_max_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MMAXV
        inline SIMDVecKNC_i max(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MAXS
        inline SIMDVecKNC_i max(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MMAXS
        inline SIMDVecKNC_i max(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MAXVA
        inline SIMDVecKNC_i & maxa(SIMDVecKNC_i const & b) {
            mVec = _mm512_max_epi32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVecKNC_i & maxa(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVecKNC_i & maxa(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_max_epi32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVecKNC_i & maxa(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVecKNC_i min(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_min_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MMINV
        inline SIMDVecKNC_i min(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MINS
        inline SIMDVecKNC_i min(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MMINS
        inline SIMDVecKNC_i min(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MINVA
        inline SIMDVecKNC_i & mina(SIMDVecKNC_i const & b) {
            mVec = _mm512_min_epi32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVecKNC_i & mina(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVecKNC_i & mina(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_min_epi32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVecKNC_i & mina(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        inline int32_t hmax() const {
            return _mm512_reduce_max_epi32(mVec);
        }
        // MHMAX
        inline int32_t hmax(SIMDMask16 const & mask) const {
            return _mm512_mask_reduce_max_epi32(mask.mMask, mVec);
        }
        // IMAX
        // MIMAX
        // HMIN
        inline int32_t hmin() const {
            return _mm512_reduce_min_epi32(mVec);
        }
        // MHMIN
        inline int32_t hmin(SIMDMask16 const & mask) const {
            return _mm512_mask_reduce_min_epi32(mask.mMask, mVec);
        }
        // IMIN
        // MIMIN

        // 2. Bitwise operations
        // BANDV
        inline SIMDVecKNC_i band(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        inline SIMDVecKNC_i operator& (SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MBANDV
        inline SIMDVecKNC_i band(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // BANDS
        inline SIMDVecKNC_i band(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MBANDS
        inline SIMDVecKNC_i band(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // BANDVA
        inline SIMDVecKNC_i & banda(SIMDVecKNC_i const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_i & operator&= (SIMDVecKNC_i const & b) {
            mVec = _mm512_and_epi32(mVec, b.mVec);
            return *this;
        }
        // MBANDVA
        inline SIMDVecKNC_i & banda(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        inline SIMDVecKNC_i & banda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_and_epi32(mVec, t0);
            return *this;
        }
        // MBANDSA
        inline SIMDVecKNC_i & banda(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        inline SIMDVecKNC_i bor(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        inline SIMDVecKNC_i operator| (SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MBORV
        inline SIMDVecKNC_i bor(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // BORS
        inline SIMDVecKNC_i bor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MBORS
        inline SIMDVecKNC_i bor(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // BORVA
        inline SIMDVecKNC_i & bora(SIMDVecKNC_i const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_i & operator|= (SIMDVecKNC_i const & b) {
            mVec = _mm512_or_epi32(mVec, b.mVec);
            return *this;
        }
        // MBORVA
        inline SIMDVecKNC_i & bora(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        inline SIMDVecKNC_i & bora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_or_epi32(mVec, t0);
            return *this;
        }
        // MBORSA
        inline SIMDVecKNC_i & bora(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        inline SIMDVecKNC_i bxor(SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        inline SIMDVecKNC_i operator^ (SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // MBXORV
        inline SIMDVecKNC_i bxor(SIMDMask16 const & mask, SIMDVecKNC_i const & b) const {
            __m512i t0 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_i(t0);
        }
        // BXORS
        inline SIMDVecKNC_i bxor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MBXORS
        inline SIMDVecKNC_i bxor(SIMDMask16 const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // BXORVA
        inline SIMDVecKNC_i & bxora(SIMDVecKNC_i const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_i & operator^= (SIMDVecKNC_i const & b) {
            mVec = _mm512_xor_epi32(mVec, b.mVec);
            return *this;
        }
        // MBXORVA
        inline SIMDVecKNC_i & bxora(SIMDMask16 const & mask, SIMDVecKNC_i const & b) {
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        inline SIMDVecKNC_i & bxora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        // MBXORSA
        inline SIMDVecKNC_i & bxora(SIMDMask16 const & mask, int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        inline SIMDVecKNC_i bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        inline SIMDVecKNC_i operator~ () const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_xor_epi32(mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // MBNOT
        inline SIMDVecKNC_i bnot(SIMDMask16 const & mask) const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_i(t1);
        }
        // BNOTA
        inline SIMDVecKNC_i & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_xor_epi32(mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVecKNC_i & bnota(SIMDMask16 const & mask) {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec = _mm512_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            return _mm512_reduce_and_epi32(mVec);
        }
        // MHBAND
        inline int32_t hband(SIMDMask16 const & mask) const {
            return _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
        }
        // HBANDS
        inline int32_t hband(int32_t a) const {
            int32_t t0 = _mm512_reduce_and_epi32(mVec);
            return a & t0;
        }
        // MHBANDS
        inline int32_t hband(SIMDMask16 const & mask, int32_t a) const {
            int32_t t0 = _mm512_mask_reduce_and_epi32(mask.mMask, mVec);
            return a & t0;
        }
        // HBOR
        inline int32_t hbor() const {
            return _mm512_reduce_or_epi32(mVec);
        }
        // MHBOR
        inline int32_t hbor(SIMDMask16 const & mask) const {
            return _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
        }
        // HBORS
        inline int32_t hbor(int32_t a) const {
            int32_t t0 = _mm512_reduce_or_epi32(mVec);
            return a | t0;
        }
        // MHBORS
        inline int32_t hbor(SIMDMask16 const & mask, int32_t a) const {
            int32_t t0 = _mm512_mask_reduce_or_epi32(mask.mMask, mVec);
            return a | t0;
        }
        // Note: reduce_xor not available in IMCI
        // HBXOR
        // MHBXOR
        // HBXORS
        // MHBXORS

        // 3. gather/scatter
        // GATHER
        // MGATHER
        // GATHERV
        // MGATHERV
        // SCATTER
        // MSCATTER
        // SCATTERV
        // MSCATTERV

        // 4. shift/rotate
        // LSHV
        // MLSHV
        // LSHS
        // MLSHS
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA   
        // RSHV 
        // MRSHV
        // RSHS
        // MRSHS
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
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

        // 4. sign
        // NEG
        // MNEG
        // NEGA
        // MNEGA
        // ABS
        // MABS
        // ABSA
        // MABSA

        // 5. pack 
        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        inline  operator SIMDVecKNC_u<uint32_t, 16> const ();

    };

    inline SIMDVecKNC_i<int32_t, 16>::operator const SIMDVecKNC_u<uint32_t, 16>() {
        return SIMDVecKNC_u<uint32_t, 16>(this->mVec);
    }

    inline SIMDVecKNC_u<uint32_t, 16>::operator const SIMDVecKNC_i<int32_t, 16>() {
        return SIMDVecKNC_i<int32_t, 16>(this->mVec);
    }
}
}

#endif
