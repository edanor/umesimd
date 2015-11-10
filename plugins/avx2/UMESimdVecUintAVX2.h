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

#ifndef UME_SIMD_VEC_UINT_AVX2_H_
#define UME_SIMD_VEC_UINT_AVX2_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

#include "UMESimdMaskAVX2.h"
#include "UMESimdSwizzleAVX2.h"

namespace UME {
namespace SIMD {

    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS
    // ********************************************************************************************
    template<typename VEC_TYPE, uint32_t VEC_LEN>
    struct SIMDVecAVX2_u_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 8b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 1> {
        typedef int8_t       SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 16b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 2> {
        typedef SIMDVecAVX2_u<uint8_t, 1> HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 1> {
        typedef int16_t      SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 32b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 4> {
        typedef SIMDVecAVX2_u<uint8_t, 2> HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 2> {
        typedef SIMDVecAVX2_u<uint16_t, 1> HALF_LEN_VEC_TYPE;
        typedef int16_t                    SCALAR_INT_TYPE;
        typedef SIMDMask2                  MASK_TYPE;
        typedef SIMDSwizzle2               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint32_t, 1> {
        typedef int32_t      SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 8> {
        typedef SIMDVecAVX2_u<uint8_t, 4> HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 4> {
        typedef SIMDVecAVX2_u<uint16_t, 2> HALF_LEN_VEC_TYPE;
        typedef int16_t                    SCALAR_INT_TYPE;
        typedef SIMDMask4                  MASK_TYPE;
        typedef SIMDSwizzle4               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint32_t, 2> {
        typedef SIMDVecAVX2_u<uint32_t, 1> HALF_LEN_VEC_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef SIMDMask2                  MASK_TYPE;
        typedef SIMDSwizzle2               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint64_t, 1> {
        typedef int64_t      SCALAR_INT_TYPE;
        typedef SIMDMask1    MASK_TYPE;
        typedef SIMDSwizzle1 SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 16> {
        typedef SIMDVecAVX2_u<uint8_t, 8> HALF_LEN_VEC_TYPE;
        typedef int8_t                    SCALAR_INT_TYPE;
        typedef SIMDMask16                MASK_TYPE;
        typedef SIMDSwizzle16             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 8> {
        typedef SIMDVecAVX2_u<uint16_t, 4> HALF_LEN_VEC_TYPE;
        typedef int16_t                    SCALAR_INT_TYPE;
        typedef SIMDMask8                  MASK_TYPE;
        typedef SIMDSwizzle8               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint32_t, 4> {
        typedef SIMDVecAVX2_u<uint32_t, 2> HALF_LEN_VEC_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef SIMDMask4                  MASK_TYPE;
        typedef SIMDSwizzle4               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint64_t, 2> {
        typedef SIMDVecAVX2_u<uint64_t, 1> HALF_LEN_VEC_TYPE;
        typedef int64_t                    SCALAR_INT_TYPE;
        typedef SIMDMask2                  MASK_TYPE;
        typedef SIMDSwizzle2               SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 32> {
        typedef SIMDVecAVX2_u<uint8_t, 16> HALF_LEN_VEC_TYPE;
        typedef int8_t                     SCALAR_INT_TYPE;
        typedef SIMDMask32                 MASK_TYPE;
        typedef SIMDSwizzle32              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 16> {
        typedef SIMDVecAVX2_u<uint16_t, 8> HALF_LEN_VEC_TYPE;
        typedef int16_t                    SCALAR_INT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint32_t, 8> {
        typedef SIMDVecAVX2_u<uint32_t, 4> HALF_LEN_VEC_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef SIMDMask8                  MASK_TYPE;
        typedef SIMDSwizzle8               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint64_t, 4> {
        typedef SIMDVecAVX2_u<uint64_t, 2> HALF_LEN_VEC_TYPE;
        typedef int64_t                    SCALAR_INT_TYPE;
        typedef SIMDMask4                  MASK_TYPE;
        typedef SIMDSwizzle4               SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 64> {
        typedef SIMDVecAVX2_u<uint8_t, 32> HALF_LEN_VEC_TYPE;
        typedef int8_t                     SCALAR_INT_TYPE;
        typedef SIMDMask64                 MASK_TYPE;
        typedef SIMDSwizzle64              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 32> {
        typedef SIMDVecAVX2_u<uint16_t, 16> HALF_LEN_VEC_TYPE;
        typedef int16_t                     SCALAR_INT_TYPE;
        typedef SIMDMask32                  MASK_TYPE;
        typedef SIMDSwizzle32               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint32_t, 16> {
        typedef SIMDVecAVX2_u<uint32_t, 8> HALF_LEN_VEC_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint64_t, 8> {
        typedef SIMDVecAVX2_u<uint64_t, 4> HALF_LEN_VEC_TYPE;
        typedef int64_t                    SCALAR_INT_TYPE;
        typedef SIMDMask8                  MASK_TYPE;
        typedef SIMDSwizzle8               SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecAVX2_u_traits<uint8_t, 128> {
        typedef SIMDVecAVX2_u<uint8_t, 64> HALF_LEN_VEC_TYPE;
        typedef int8_t                     SCALAR_INT_TYPE;
        typedef SIMDMask128                MASK_TYPE;
        typedef SIMDSwizzle128             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint16_t, 64> {
        typedef SIMDVecAVX2_u<uint16_t, 32> HALF_LEN_VEC_TYPE;
        typedef int16_t                     SCALAR_INT_TYPE;
        typedef SIMDMask64                  MASK_TYPE;
        typedef SIMDSwizzle64               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint32_t, 32> {
        typedef SIMDVecAVX2_u<uint32_t, 16> HALF_LEN_VEC_TYPE;
        typedef int32_t                     SCALAR_INT_TYPE;
        typedef SIMDMask32                  MASK_TYPE;
        typedef SIMDSwizzle32               SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecAVX2_u_traits<uint64_t, 16> {
        typedef SIMDVecAVX2_u<uint64_t, 16> HALF_LEN_VEC_TYPE;
        typedef int64_t                     SCALAR_INT_TYPE;
        typedef SIMDMask16                  MASK_TYPE;
        typedef SIMDSwizzle16               SWIZZLE_MASK_TYPE;
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
    class SIMDVecAVX2_u final :
        public SIMDVecUnsignedInterface<
        SIMDVecAVX2_u<SCALAR_UINT_TYPE, VEC_LEN>, // DERIVED_VEC_UINT_TYPE
        SCALAR_UINT_TYPE,                        // SCALAR_UINT_TYPE
        VEC_LEN,
        typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE,
        typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
        SIMDVecAVX2_u<SCALAR_UINT_TYPE, VEC_LEN>,
        typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, VEC_LEN>                               VEC_EMU_REG;

        typedef typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::SCALAR_INT_TYPE   SCALAR_INT_TYPE;
        typedef typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, VEC_LEN>::MASK_TYPE         MASK_TYPE;

        // Conversion operators require access to private members.
        friend class SIMDVecAVX2_i<SCALAR_INT_TYPE, VEC_LEN>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecAVX2_u() : mVec() {};

        inline explicit SIMDVecAVX2_u(SCALAR_UINT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2_u(SCALAR_UINT_TYPE const *p) { this->load(p); };

        inline SIMDVecAVX2_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3) {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
        }

        inline SIMDVecAVX2_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7)
        {
            mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
        }

        inline SIMDVecAVX2_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
            SCALAR_UINT_TYPE i8, SCALAR_UINT_TYPE i9, SCALAR_UINT_TYPE i10, SCALAR_UINT_TYPE i11, SCALAR_UINT_TYPE i12, SCALAR_UINT_TYPE i13, SCALAR_UINT_TYPE i14, SCALAR_UINT_TYPE i15)
        {
            mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
            mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
            mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
            mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
        }

        inline SIMDVecAVX2_u(SCALAR_UINT_TYPE i0, SCALAR_UINT_TYPE i1, SCALAR_UINT_TYPE i2, SCALAR_UINT_TYPE i3, SCALAR_UINT_TYPE i4, SCALAR_UINT_TYPE i5, SCALAR_UINT_TYPE i6, SCALAR_UINT_TYPE i7,
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

        // Override Access operators
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX2_u, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVecAVX2_u, MASK_TYPE>(mask, static_cast<SIMDVecAVX2_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX2_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        // UTOI
        inline  operator SIMDVecAVX2_i<SCALAR_INT_TYPE, VEC_LEN>() const {
            SIMDVecAVX2_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            for (uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ***************************************************************************
    // *
    // *    Partial specialization of unsigned integer SIMD for VEC_LEN == 1.
    // *    This specialization is necessary to eliminate PACK operations from
    // *    being used on SIMD1 types.
    // *
    // ***************************************************************************
    template<typename SCALAR_UINT_TYPE>
    class SIMDVecAVX2_u<SCALAR_UINT_TYPE, 1> final :
        public SIMDVecUnsignedInterface<
        SIMDVecAVX2_u<SCALAR_UINT_TYPE, 1>, // DERIVED_VEC_UINT_TYPE
        SCALAR_UINT_TYPE,                        // SCALAR_UINT_TYPE
        1,
        typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, 1>::MASK_TYPE,
        typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_UINT_TYPE, 1>                                   VEC_EMU_REG;

        typedef typename SIMDVecAVX2_u_traits<SCALAR_UINT_TYPE, 1>::SCALAR_INT_TYPE  SCALAR_INT_TYPE;

        // Conversion operators require access to private members.
        friend class SIMDVecAVX2_i<SCALAR_INT_TYPE, 1>;

    private:
        // This is the only data member and it is a low level representation of vector register.
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecAVX2_u() : mVec() {};

        inline explicit SIMDVecAVX2_u(SCALAR_UINT_TYPE i) : mVec(i) {};

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2_u(SCALAR_UINT_TYPE const *p) { this->load(p); };

        // Override Access operators
        inline SCALAR_UINT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX2_u, SIMDMask1> operator[] (SIMDMask1 & mask) {
            return IntermediateMask<SIMDVecAVX2_u, SIMDMask1>(mask, static_cast<SIMDVecAVX2_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX2_u & insert(uint32_t index, SCALAR_UINT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline  operator SIMDVecAVX2_i<SCALAR_INT_TYPE, 1>() const {
            SIMDVecAVX2_i<SCALAR_INT_TYPE, 1> retval(mVec[0]);
            return retval;
        }

        // UNIQUE
        inline bool unique() const {
            return true;
        }
    };

    // ********************************************************************************************
    // UNSIGNED INTEGER VECTORS specialization
    // ********************************************************************************************
    template<>
    class SIMDVecAVX2_u<uint32_t, 2> :
        public SIMDVecUnsignedInterface<
        SIMDVecAVX2_u<uint32_t, 2>,
        uint32_t,
        2,
        SIMDMask2,
        SIMDSwizzle2>,
        public SIMDVecPackableInterface<
        SIMDVecAVX2_u<uint32_t, 2>,
        SIMDVecAVX2_u<uint32_t, 1 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVecAVX2_i<int32_t, 2>;

    private:
        uint32_t mVec[2];

    public:
        inline SIMDVecAVX2_u() {}

        inline explicit SIMDVecAVX2_u(uint32_t i) {
            mVec[0] = i;
            mVec[1] = i;
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2_u(uint32_t const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        };

        inline SIMDVecAVX2_u(uint32_t i0, uint32_t i1)
        {
            mVec[0] = i0;
            mVec[1] = i1;
        }

        // EXTRACT
        inline uint32_t extract(uint32_t index) const {
            return mVec[index & 1];
        }

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return mVec[index & 1];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX2_u, SIMDMask2> operator[] (SIMDMask2 const & mask) {
            return IntermediateMask<SIMDVecAVX2_u, SIMDMask2>(mask, static_cast<SIMDVecAVX2_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX2_u & insert(uint32_t index, uint32_t value) {
            mVec[index & 1] = value;
            return *this;
        }

        // PREFINC
        inline SIMDVecAVX2_u & prefinc() {
            mVec[0]++;
            mVec[1]++;
            return *this;
        }

        // MPREFINC
        inline SIMDVecAVX2_u & prefinc(SIMDMask2 const & mask) {
            if (mask[0] == true) mVec[0]++;
            if (mask[1] == true) mVec[1]++;
            return *this;
        }

        // UNIQUE
        inline bool unique() const {
            return mVec[0] != mVec[1];
        }

        // GATHERS
        inline SIMDVecAVX2_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        inline SIMDVecAVX2_u & gather(SIMDMask2 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            if (mask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        inline SIMDVecAVX2_u & gather(uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERV
        inline SIMDVecAVX2_u & gather(SIMDMask2 const & mask, uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            if (mask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDMask2 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            if (mask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDMask2 const & mask, uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            if (mask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        inline void unpack(SIMDVecAVX2_u<uint32_t, 1> & a, SIMDVecAVX2_u<uint32_t, 1> & b) const {
            a.insert(0, mVec[0]);
            b.insert(0, mVec[1]);
        }
        // UNPACKLO
        // UNPACKHI

        // UTOI
        inline  operator SIMDVecAVX2_i<int32_t, 2> const ();
    };

    template<>
    class SIMDVecAVX2_u<uint32_t, 4> :
        public SIMDVecUnsignedInterface<
        SIMDVecAVX2_u<uint32_t, 4>,
        uint32_t,
        4,
        SIMDMask4,
        SIMDSwizzle4>,
        public SIMDVecPackableInterface<
        SIMDVecAVX2_u<uint32_t, 4>,
        SIMDVecAVX2_u<uint32_t, 2 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVecAVX2_i<int32_t, 4>;

    private:
        __m128i mVec;

        inline SIMDVecAVX2_u(__m128i & x) { this->mVec = x; }
    public:
        inline SIMDVecAVX2_u() {}

        inline explicit SIMDVecAVX2_u(uint32_t i) {
            mVec = _mm_set1_epi32(i);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2_u(uint32_t const *p) { this->load(p); };

        inline SIMDVecAVX2_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
        {
            mVec = _mm_set_epi32(i3, i2, i1, i0);
        }

        inline uint32_t extract(uint32_t index) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX2_u, SIMDMask4> operator[] (SIMDMask4 const & mask) {
            return IntermediateMask<SIMDVecAVX2_u, SIMDMask4>(mask, static_cast<SIMDVecAVX2_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX2_u & insert(uint32_t index, uint32_t value) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }

        // ASSIGNV
        inline SIMDVecAVX2_u & assign(SIMDVecAVX2_u const & b) {
            mVec = b.mVec;
            return *this;
        }

        // PREFINC
        inline SIMDVecAVX2_u & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_add_epi32(mVec, t0);
            return *this;
        }

        // MPREFINC
        inline SIMDVecAVX2_u & prefinc(SIMDMask4 const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = _mm_add_epi32(mVec, t0);
            mVec = _mm_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }

        // UNIQUE
        inline bool unique() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            for (unsigned int i = 0; i < 3; i++) {
                for (unsigned int j = i + 1; j < 4; j++) {
                    if (raw[i] == raw[j]) return false;
                }
            }
            return true;
        }

        // GATHERS
        inline SIMDVecAVX2_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVecAVX2_u & gather(SIMDMask4 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            __m128i t0 = _mm_load_si128((__m128i*)raw);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // GATHERV
        inline SIMDVecAVX2_u & gather(uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(16) uint32_t rawInd[4];
            alignas(16) uint32_t raw[4];

            _mm_store_si128((__m128i*) rawInd, indices.mVec);
            for (int i = 0; i < 4; i++) { raw[i] = baseAddr[rawInd[i]]; }
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERV
        inline SIMDVecAVX2_u & gather(SIMDMask4 const & mask, uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(16) uint32_t rawInd[4];
            alignas(16) uint32_t raw[4];

            _mm_store_si128((__m128i*) rawInd, indices.mVec);
            for (int i = 0; i < 4; i++) { raw[i] = baseAddr[rawInd[i]]; }
            __m128i t0 = _mm_load_si128((__m128i*)&raw[0]);
            mVec = _mm_blendv_epi8(mVec, t0, mask.mMask);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            for (int i = 0; i < 4; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDMask4 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4];
            alignas(16) uint32_t rawMask[4];
            _mm_store_si128((__m128i*) raw, mVec);
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
            for (int i = 0; i < 4; i++) { if (rawMask[i] == SIMDMask4::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(16) uint32_t raw[4];
            alignas(16) uint32_t rawIndices[4];
            _mm_store_si128((__m128i*) raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            for (int i = 0; i < 4; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDMask4 const & mask, uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(16) uint32_t raw[4];
            alignas(16) uint32_t rawIndices[4];
            alignas(16) uint32_t rawMask[4];
            _mm_store_si128((__m128i*) raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
            for (int i = 0; i < 4; i++) {
                if (rawMask[i] == SIMDMask4::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
            return baseAddr;
        }

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        inline void unpack(SIMDVecAVX2_u<uint32_t, 2> & a, SIMDVecAVX2_u<uint32_t, 2> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            a.insert(0, raw[0]);
            a.insert(1, raw[1]);
            b.insert(0, raw[2]);
            b.insert(1, raw[3]);
        }
        // UNPACKLO
        // UNPACKHI

        // UTOI
        inline  operator SIMDVecAVX2_i<int32_t, 4> const ();
    };

    template<>
    class SIMDVecAVX2_u<uint32_t, 8> :
        public SIMDVecUnsignedInterface<
        SIMDVecAVX2_u<uint32_t, 8>,
        uint32_t,
        8,
        SIMDMask8,
        SIMDSwizzle8>,
        public SIMDVecPackableInterface<
        SIMDVecAVX2_u<uint32_t, 8>,
        SIMDVecAVX2_u<uint32_t, 4 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVecAVX2_i<int32_t, 8>;

    private:
        __m256i mVec;

        inline SIMDVecAVX2_u(__m256i & x) { this->mVec = x; }
    public:
        inline SIMDVecAVX2_u() {
            mVec = _mm256_setzero_si256();
        }

        inline explicit SIMDVecAVX2_u(uint32_t i) {
            mVec = _mm256_set1_epi32(i);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecAVX2_u(uint32_t const *p) { this->load(p); };

        inline SIMDVecAVX2_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
            uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        inline uint32_t extract(uint32_t index) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecAVX2_u, SIMDMask8> operator[] (SIMDMask8 const & mask) {
            return IntermediateMask<SIMDVecAVX2_u, SIMDMask8>(mask, static_cast<SIMDVecAVX2_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecAVX2_u & insert(uint32_t index, uint32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }

        // ASSIGNV
        inline SIMDVecAVX2_u & assign(SIMDVecAVX2_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV
        inline SIMDVecAVX2_u & assign(SIMDMask8 const & mask, SIMDVecAVX2_u const & b) {
            mVec = _mm256_blendv_epi8(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        // MASSIGNS

        // LOAD
        // MLOAD
        // LOADA
        inline SIMDVecAVX2_u & loada(uint32_t const * p) {
            _mm256_load_si256((__m256i *)p);
        }
        // MLOADA
        inline SIMDVecAVX2_u & loada(SIMDMask8 const & mask, uint32_t const * p) {
            _mm256_maskload_epi32((int *)p, mask.mMask);
            _mm256_load_si256((__m256i *)p);
        }

        // STOREA
        inline uint32_t * storea(uint32_t * addrAligned) {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }

        // ADDV
        inline SIMDVecAVX2_u add(SIMDVecAVX2_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVecAVX2_u(t0);
        }

        inline SIMDVecAVX2_u operator+ (SIMDVecAVX2_u const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVecAVX2_u add(SIMDMask8 const & mask, SIMDVecAVX2_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_blendv_epi8(mVec, t0, mask.mMask);
            return SIMDVecAVX2_u(t1);
        }
        // ADDS
        inline SIMDVecAVX2_u add(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            return SIMDVecAVX2_u(t1);
        }
        // MADDS
        inline SIMDVecAVX2_u add(SIMDMask8 const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            __m256i t2 = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return SIMDVecAVX2_u(t2);
        }
        // ADDVA
        inline SIMDVecAVX2_u adda(SIMDVecAVX2_u const & b) {
            mVec = _mm256_add_epi32(mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVecAVX2_u & adda(SIMDMask8 const & mask, SIMDVecAVX2_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);
            __m128i r_low = _mm_add_epi32(a_low, b_low);
            __m128i r_high = _mm_add_epi32(a_high, b_high);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // ADDSA 
        inline SIMDVecAVX2_u & adda(uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_add_epi32(a_low, b_vec);
            __m128i r_high = _mm_add_epi32(a_high, b_vec);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // MADDSA
        inline SIMDVecAVX2_u & adda(SIMDMask8 const & mask, uint32_t b) {
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i r_low = _mm_add_epi32(a_low, b_vec);
            __m128i r_high = _mm_add_epi32(a_high, b_vec);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }

        // POSTINC
        inline SIMDVecAVX2_u postinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_add_epi32(mVec, t0);
            return SIMDVecAVX2_u(t1);
        }

        // MPOSTINC
        inline SIMDVecAVX2_u postinc(SIMDMask8 const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            __m256i t2 = _mm256_add_epi32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t2, mask.mMask);
            return SIMDVecAVX2_u(t1);
        }

        // PREFINC
        inline SIMDVecAVX2_u & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }

        // MPREFINC
        inline SIMDVecAVX2_u & prefinc(SIMDMask8 const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            mVec = _mm256_blendv_epi8(mVec, t1, mask.mMask);
            return *this;
        }

        // MULV
        inline SIMDVecAVX2_u mul(SIMDVecAVX2_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);
            __m128i r_low = _mm_mullo_epi32(a_low, b_low);
            __m128i r_high = _mm_mullo_epi32(a_high, b_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX2_u(ret);
        }
        // MMULV
        inline SIMDVecAVX2_u mul(SIMDMask8 const & mask, SIMDVecAVX2_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);
            __m128i r_low = _mm_mullo_epi32(a_low, b_low);
            __m128i r_high = _mm_mullo_epi32(a_high, b_high);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX2_u(ret);
        }
        // MULS
        inline SIMDVecAVX2_u mul(uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_mullo_epi32(a_low, b_vec);
            __m128i r_high = _mm_mullo_epi32(a_high, b_vec);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX2_u(ret);
        }
        // MMULS
        inline SIMDVecAVX2_u mul(SIMDMask8 const & mask, uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_mullo_epi32(a_low, b_vec);
            __m128i r_high = _mm_mullo_epi32(a_high, b_vec);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecAVX2_u(ret);
        }
        // CMPEQV
        inline SIMDMask8 cmpeq(SIMDVecAVX2_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);

            __m128i r_low = _mm_cmpeq_epi32(a_low, b_low);
            __m128i r_high = _mm_cmpeq_epi32(a_high, b_high);

            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDMask8(ret);
        }
        // MCMPEQ
        inline SIMDMask8 cmpeq(uint32_t b) {
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);

            __m128i r_low = _mm_cmpeq_epi32(a_low, b_vec);
            __m128i r_high = _mm_cmpeq_epi32(a_high, b_vec);

            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDMask8(ret);
        }

        // UNIQUE
        inline bool unique() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            for (unsigned int i = 0; i < 7; i++) {
                for (unsigned int j = i + 1; j < 8; j++) {
                    if (raw[i] == raw[j]) {
                        return false;
                    }
                }
            }
            return true;
        }

        // GATHERS
        inline SIMDVecAVX2_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]] };
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVecAVX2_u & gather(SIMDMask8 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]] };
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // GATHERV
        inline SIMDVecAVX2_u & gather(uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(32) uint32_t rawInd[8];
            alignas(32) uint32_t raw[8];

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for (int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERV
        inline SIMDVecAVX2_u & gather(SIMDMask8 const & mask, uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(32) uint32_t rawInd[8];
            alignas(32) uint32_t raw[8];

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for (int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            for (int i = 0; i < 8; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDMask8 const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) { if (rawMask[i] == SIMDMask8::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            for (int i = 0; i < 8; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDMask8 const & mask, uint32_t* baseAddr, SIMDVecAVX2_u const & indices) {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) {
                if (rawMask[i] == SIMDMask8::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
            return baseAddr;
        }

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        inline void unpack(SIMDVecAVX2_u<uint32_t, 4> & a, SIMDVecAVX2_u<uint32_t, 4> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            a.loada(raw);
            b.loada(raw + 4);
        }
        // UNPACKLO
        // UNPACKHI

        inline  operator SIMDVecAVX2_i<int32_t, 8> const ();
    };
}
}
#endif
