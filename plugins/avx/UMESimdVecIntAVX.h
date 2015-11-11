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

#ifndef UME_SIMD_VEC_INT_AVX_H_
#define UME_SIMD_VEC_INT_AVX_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

#include "UMESimdMaskAVX.h"
#include "UMESimdSwizzleAVX.h"
#include "UMESimdVecUintAVX.h"

namespace UME
{
namespace SIMD
{
    // ********************************************************************************************
    // SIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename SCALAR_INT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecAVX_i_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 1> {
        typedef SIMDVecAVX_u<uint8_t, 1> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                MASK_TYPE;
        typedef SIMDSwizzle1             SWIZZLE_MASK_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 2> {
        typedef SIMDVecAVX_i<int8_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 2> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                MASK_TYPE;
        typedef SIMDSwizzle2             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 1> {
        typedef SIMDVecAVX_u<uint16_t, 1> VEC_UINT;
        typedef uint16_t                 SCALAR_UINT_TYPE;
        typedef SIMDMask1                MASK_TYPE;
        typedef SIMDSwizzle1             SWIZZLE_MASK_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 4> {
        typedef SIMDVecAVX_i<int8_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 4> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                MASK_TYPE;
        typedef SIMDSwizzle4             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 2> {
        typedef SIMDVecAVX_i<int16_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 2> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 1> {
        typedef SIMDVecAVX_u<uint32_t, 1> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 8> {
        typedef SIMDVecAVX_i<int8_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 8> VEC_UINT;
        typedef uint8_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                MASK_TYPE;
        typedef SIMDSwizzle8             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 4> {
        typedef SIMDVecAVX_i<int16_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 4> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 2> {
        typedef SIMDVecAVX_i<int32_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 2> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 1> {
        typedef SIMDVecAVX_u<uint64_t, 1> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 16> {
        typedef SIMDVecAVX_i<int8_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 16> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 8> {
        typedef SIMDVecAVX_i<int16_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 8> VEC_UINT;
        typedef uint16_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 4> {
        typedef SIMDVecAVX_i<int32_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 4> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 2> {
        typedef SIMDVecAVX_i<int64_t, 1>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 2> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    //256b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 32> {
        typedef SIMDVecAVX_i<int8_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 32> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                MASK_TYPE;
        typedef SIMDSwizzle32             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 16> {
        typedef SIMDVecAVX_i<int16_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 16> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 8> {
        typedef SIMDVecAVX_i<int32_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 8> VEC_UINT;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 4> {
        typedef SIMDVecAVX_i<int64_t, 2>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 4> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 64> {
        typedef SIMDVecAVX_i<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 64> VEC_UINT;
        typedef uint8_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask64                MASK_TYPE;
        typedef SIMDSwizzle64             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 32> {
        typedef SIMDVecAVX_i<int16_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 32> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 16> {
        typedef SIMDVecAVX_i<int32_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 16> VEC_UINT;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 8> {
        typedef SIMDVecAVX_i<int64_t, 4>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 8> VEC_UINT;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecAVX_i_traits<int8_t, 128> {
        typedef SIMDVecAVX_i<int8_t, 64>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint8_t, 128> VEC_UINT;
        typedef uint8_t                    SCALAR_UINT_TYPE;
        typedef SIMDMask128                MASK_TYPE;
        typedef SIMDSwizzle128             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int16_t, 64> {
        typedef SIMDVecAVX_i<int16_t, 32>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint16_t, 64> VEC_UINT;
        typedef uint16_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask64                 MASK_TYPE;
        typedef SIMDSwizzle64              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int32_t, 32> {
        typedef SIMDVecAVX_i<int32_t, 16>  HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint32_t, 32> VEC_UINT;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX_i_traits<int64_t, 16> {
        typedef SIMDVecAVX_i<int64_t, 8>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecAVX_u<uint64_t, 16> VEC_UINT;
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
    class SIMDVecAVX_i final :
        public SIMDVecSignedInterface<
        SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT,
        SCALAR_INT_TYPE,
        VEC_LEN,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
        SIMDVecAVX_i<SCALAR_INT_TYPE, VEC_LEN>,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, VEC_LEN>                            VEC_EMU_REG;

        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE     SCALAR_UINT_TYPE;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::VEC_UINT             VEC_UINT;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, VEC_LEN>::MASK_TYPE            MASK_TYPE;

        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, VEC_LEN>;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE const * p) { this->load(p); };

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1) {
            mVec.insert(0, i0);  mVec.insert(1, i1);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7)
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
            SCALAR_INT_TYPE i8, SCALAR_INT_TYPE i9, SCALAR_INT_TYPE i10, SCALAR_INT_TYPE i11, SCALAR_INT_TYPE i12, SCALAR_INT_TYPE i13, SCALAR_INT_TYPE i14, SCALAR_INT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
        }

        inline SIMDVecAVX_i(SCALAR_INT_TYPE i0, SCALAR_INT_TYPE i1, SCALAR_INT_TYPE i2, SCALAR_INT_TYPE i3, SCALAR_INT_TYPE i4, SCALAR_INT_TYPE i5, SCALAR_INT_TYPE i6, SCALAR_INT_TYPE i7,
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
        inline IntermediateMask<SIMDVecAVX_i, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_i, MASK_TYPE>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVecAVX_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
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
    class SIMDVecAVX_i<SCALAR_INT_TYPE, 1> :
        public SIMDVecSignedInterface<
        SIMDVecAVX_i<SCALAR_INT_TYPE, 1>,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT,
        SCALAR_INT_TYPE,
        1,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::MASK_TYPE,
        typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_INT_TYPE, 1>                                 VEC_EMU_REG;

        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::SCALAR_UINT_TYPE      SCALAR_UINT_TYPE;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::VEC_UINT              VEC_UINT;
        typedef typename SIMDVecAVX_i_traits<SCALAR_INT_TYPE, 1>::MASK_TYPE             MASK_TYPE;

        friend class SIMDVecScalarEmu_u<SCALAR_UINT_TYPE, 1>;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR
        inline explicit SIMDVecAVX_i(SCALAR_INT_TYPE const * p) { this->load(p); };

        // Override Access operators
        inline SCALAR_INT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_i, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX_i, MASK_TYPE>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, SCALAR_INT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVecAVX_u<SCALAR_UINT_TYPE, 1> retval(mVec[0]);
            return retval;
        }
    };

    // ********************************************************************************************
    // SIGNED INTEGER VECTOR specializations
    // ********************************************************************************************

    template<>
    class SIMDVecAVX_i<int32_t, 8> :
        public SIMDVecSignedInterface<
        SIMDVecAVX_i<int32_t, 8>,
        SIMDVecAVX_u<uint32_t, 8>,
        int32_t,
        8,
        uint32_t,
        SIMDMask8,
        SIMDSwizzle8>,
        public SIMDVecPackableInterface<
        SIMDVecAVX_i<int32_t, 8>,
        SIMDVecAVX_i<int32_t, 4 >>
    {
        friend class SIMDVecAVX_u<uint32_t, 8>;
        friend class SIMDVecAVX_f<float, 8>;
        friend class SIMDVecAVX_f<double, 8>;
    private:
        __m256i mVec;

        inline explicit SIMDVecAVX_i(__m256i & x) {
            this->mVec = x;
        }
    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(int32_t i) {
            mVec = _mm256_set1_epi32(i);
        }

        // LOAD-CONSTR
        inline explicit SIMDVecAVX_i(int32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i *)p);
        }

        inline SIMDVecAVX_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        inline int32_t extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_i, SIMDMask8> operator[] (SIMDMask8 const & mask) {
            return IntermediateMask<SIMDVecAVX_i, SIMDMask8>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i *)raw);
            return *this;
        }

        inline  operator SIMDVecAVX_u<uint32_t, 8> const ();
        inline  operator SIMDVecAVX_f<float, 8> const ();

        // ABS
        SIMDVecAVX_i abs() {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, _mm_abs_epi32(a_low), 0);
            ret = _mm256_insertf128_si256(ret, _mm_abs_epi32(a_high), 1);
            return SIMDVecAVX_i(ret);
        }
        // MABS
        SIMDVecAVX_i abs(SIMDMask8 const & mask) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);

            __m128i r_low = _mm_blendv_epi8(a_low, _mm_abs_epi32(a_low), m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, _mm_abs_epi32(a_high), m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX_i(ret);
        }
    };

    inline SIMDVecAVX_i<int32_t, 8>::operator const SIMDVecAVX_u<uint32_t, 8>() {
        return SIMDVecAVX_u<uint32_t, 8>(this->mVec);
    }

    inline SIMDVecAVX_u<uint32_t, 8>::operator const SIMDVecAVX_i<int32_t, 8>() {
        return SIMDVecAVX_i<int32_t, 8>(this->mVec);
    }

    template<>
    class SIMDVecAVX_i<int32_t, 16> :
        public SIMDVecSignedInterface<
        SIMDVecAVX_i<int32_t, 16>,
        SIMDVecAVX_u<uint32_t, 16>,
        int32_t,
        16,
        uint32_t,
        SIMDMask16,
        SIMDSwizzle16>,
        public SIMDVecPackableInterface<
        SIMDVecAVX_i<int32_t, 16>,
        SIMDVecAVX_i<int32_t, 8 >>
    {
        friend class SIMDVecAVX_u<uint32_t, 16>;
        friend class SIMDVecAVX_f<float, 16>;
        friend class SIMDVecAVX_f<double, 16>;
    private:
        __m256i mVecLo;
        __m256i mVecHi;

        inline explicit SIMDVecAVX_i(__m256i & x_lo, __m256i & x_hi) {
            this->mVecLo = x_lo;
            this->mVecHi = x_hi;
        }
    public:
        // ZERO-CONSTR
        inline SIMDVecAVX_i() {};

        // SET-CONSTR
        inline explicit SIMDVecAVX_i(int32_t i) {
            mVecLo = _mm256_set1_epi32(i);
            mVecHi = mVecLo;
        }

        // LOAD-CONSTR
        inline explicit SIMDVecAVX_i(int32_t const * p) {
            mVecLo = _mm256_loadu_si256((__m256i *)p);
            mVecHi = _mm256_loadu_si256((__m256i *)(p + 8));
        }

        inline SIMDVecAVX_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7,
            int32_t i8, int32_t i9, int32_t i10, int32_t i11,
            int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVecLo = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVecHi = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
        }

        inline int32_t extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            int32_t value;
            if (index < 8) {
                _mm256_store_si256((__m256i *)raw, mVecLo);
                value = raw[index];
            }
            else {
                _mm256_store_si256((__m256i *)raw, mVecHi);
                value = raw[index - 8];
            }
            return value;
        }

        // Override Access operators
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX_i, SIMDMask16> operator[] (SIMDMask16 const & mask) {
            return IntermediateMask<SIMDVecAVX_i, SIMDMask16>(mask, static_cast<SIMDVecAVX_i &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i *) raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm256_load_si256((__m256i *)raw);
            }
            else {
                _mm256_store_si256((__m256i *) raw, mVecHi);
                raw[index - 8] = value;
                mVecHi = _mm256_load_si256((__m256i *) raw);
            }
            return *this;
        }

        inline  operator SIMDVecAVX_u<uint32_t, 16> const ();
        inline  operator SIMDVecAVX_f<float, 16> const ();

        // ABS
        SIMDVecAVX_i abs() {
            __m128i a_low = _mm256_extractf128_si256(mVecLo, 0);
            __m128i a_high = _mm256_extractf128_si256(mVecLo, 1);
            __m256i ret_lo = _mm256_setzero_si256();
            __m256i ret_hi = _mm256_setzero_si256();
            ret_lo = _mm256_insertf128_si256(ret_lo, _mm_abs_epi32(a_low), 0);
            ret_lo = _mm256_insertf128_si256(ret_lo, _mm_abs_epi32(a_high), 1);

            a_low = _mm256_extractf128_si256(mVecHi, 0);
            a_high = _mm256_extractf128_si256(mVecHi, 1);
            ret_hi = _mm256_insertf128_si256(ret_hi, _mm_abs_epi32(a_low), 0);
            ret_hi = _mm256_insertf128_si256(ret_hi, _mm_abs_epi32(a_high), 1);

            return SIMDVecAVX_i(ret_lo, ret_hi);
        }
        // MABS
        SIMDVecAVX_i abs(SIMDMask16 const & mask) {
            __m128i a_lo = _mm256_extractf128_si256(mVecLo, 0);
            __m128i a_hi = _mm256_extractf128_si256(mVecLo, 1);
            __m128i m_lo = _mm256_extractf128_si256(mask.mMaskLo, 0);
            __m128i m_hi = _mm256_extractf128_si256(mask.mMaskLo, 1);

            __m128i r_lo = _mm_blendv_epi8(a_lo, _mm_abs_epi32(a_lo), m_lo);
            __m128i r_hi = _mm_blendv_epi8(a_hi, _mm_abs_epi32(a_hi), m_hi);
            __m256i ret_lo = _mm256_setzero_si256();
            __m256i ret_hi = _mm256_setzero_si256();
            ret_lo = _mm256_insertf128_si256(ret_lo, r_lo, 0);
            ret_lo = _mm256_insertf128_si256(ret_lo, r_hi, 1);

            a_lo = _mm256_extractf128_si256(mVecHi, 0);
            a_hi = _mm256_extractf128_si256(mVecHi, 1);
            m_lo = _mm256_extractf128_si256(mask.mMaskHi, 0);
            m_hi = _mm256_extractf128_si256(mask.mMaskHi, 1);

            r_lo = _mm_blendv_epi8(a_lo, _mm_abs_epi32(a_lo), m_lo);
            r_hi = _mm_blendv_epi8(a_hi, _mm_abs_epi32(a_hi), m_hi);

            ret_hi = _mm256_insertf128_si256(ret_hi, r_lo, 0);
            ret_hi = _mm256_insertf128_si256(ret_hi, r_hi, 1);

            return SIMDVecAVX_i(ret_lo, ret_hi);
        }
    };

    inline SIMDVecAVX_i<int32_t, 16>::operator const SIMDVecAVX_u<uint32_t, 16>() {
        return SIMDVecAVX_i<int32_t, 16>(this->mVecLo, this->mVecHi);
    }

    inline SIMDVecAVX_u<uint32_t, 16>::operator const SIMDVecAVX_i<int32_t, 16>() {
        return SIMDVecAVX_u<uint32_t, 16>(this->mVecLo, this->mVecHi);
    }
}
}

#endif
