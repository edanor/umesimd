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

#ifndef UME_SIMD_VEC_FLOAT_AVX2_H_
#define UME_SIMD_VEC_FLOAT_AVX2_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

#include "UMESimdMaskAVX2.h"
#include "UMESimdSwizzleAVX2.h"
#include "UMESimdVecUintAVX2.h"
#include "UMESimdVecIntAVX2.h"

namespace UME {
    namespace SIMD {

        // ********************************************************************************************
        // FLOATING POINT VECTORS
        // ********************************************************************************************

        template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
        struct SIMDVecAVX2_f_traits {
            // Generic trait class not containing type definition so that only correct explicit
            // type definitions are compiled correctly
        };

        // 32b vectors
        template<>
        struct SIMDVecAVX2_f_traits<float, 1> {
            typedef SIMDVecAVX2_u<uint32_t, 1> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 1>  VEC_INT_TYPE;
            typedef int32_t                    SCALAR_INT_TYPE;
            typedef uint32_t                   SCALAR_UINT_TYPE;
            typedef float*                     SCALAR_TYPE_PTR;
            typedef SIMDMask1                  MASK_TYPE;
            typedef SIMDSwizzle1               SWIZZLE_MASK_TYPE;
        };

        // 64b vectors
        template<>
        struct SIMDVecAVX2_f_traits<float, 2> {
            typedef SIMDVecAVX2_f<float, 1>    HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint32_t, 2> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 2>  VEC_INT_TYPE;
            typedef int32_t                    SCALAR_INT_TYPE;
            typedef uint32_t                   SCALAR_UINT_TYPE;
            typedef float*                     SCALAR_TYPE_PTR;
            typedef SIMDMask2                  MASK_TYPE;
            typedef SIMDSwizzle2               SWIZZLE_MASK_TYPE;
        };

        template<>
        struct SIMDVecAVX2_f_traits<double, 1> {
            typedef SIMDVecAVX2_u<uint64_t, 1> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int64_t, 1>  VEC_INT_TYPE;
            typedef int64_t                    SCALAR_INT_TYPE;
            typedef uint64_t                   SCALAR_UINT_TYPE;
            typedef float*                     SCALAR_TYPE_PTR;
            typedef SIMDMask1                  MASK_TYPE;
            typedef SIMDSwizzle1               SWIZZLE_MASK_TYPE;
        };

        // 128b vectors
        template<>
        struct SIMDVecAVX2_f_traits<float, 4> {
            typedef SIMDVecAVX2_f<float, 2>    HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint32_t, 4> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 4>  VEC_INT_TYPE;
            typedef int32_t                    SCALAR_INT_TYPE;
            typedef uint32_t                   SCALAR_UINT_TYPE;
            typedef float*                     SCALAR_TYPE_PTR;
            typedef SIMDMask4                  MASK_TYPE;
            typedef SIMDSwizzle4               SWIZZLE_MASK_TYPE;
        };

        template<>
        struct SIMDVecAVX2_f_traits<double, 2> {
            typedef SIMDVecAVX2_f<double, 1>   HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint64_t, 2> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int64_t, 2>  VEC_INT_TYPE;
            typedef int64_t                    SCALAR_INT_TYPE;
            typedef uint64_t                   SCALAR_UINT_TYPE;
            typedef double*                    SCALAR_TYPE_PTR;
            typedef SIMDMask2                  MASK_TYPE;
            typedef SIMDSwizzle2               SWIZZLE_MASK_TYPE;
        };

        // 256b vectors
        template<>
        struct SIMDVecAVX2_f_traits<float, 8> {
            typedef SIMDVecAVX2_f<float, 4>    HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint64_t, 8> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 8>  VEC_INT_TYPE;
            typedef int32_t                    SCALAR_INT_TYPE;
            typedef uint32_t                   SCALAR_UINT_TYPE;
            typedef float*                     SCALAR_TYPE_PTR;
            typedef SIMDMask8                  MASK_TYPE;
            typedef SIMDSwizzle8               SWIZZLE_MASK_TYPE;
        };

        template<>
        struct SIMDVecAVX2_f_traits<double, 4> {
            typedef SIMDVecAVX2_f<double, 2>   HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint64_t, 4> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int64_t, 4>  VEC_INT_TYPE;
            typedef int64_t                    SCALAR_INT_TYPE;
            typedef uint64_t                   SCALAR_UINT_TYPE;
            typedef double*                    SCALAR_TYPE_PTR;
            typedef SIMDMask4                  MASK_TYPE;
            typedef SIMDSwizzle4               SWIZZLE_MASK_TYPE;
        };

        // 512b vectors
        template<>
        struct SIMDVecAVX2_f_traits<float, 16> {
            typedef SIMDVecAVX2_f<float, 8>     HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint32_t, 16> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 16> VEC_INT_TYPE;
            typedef int32_t                     SCALAR_INT_TYPE;
            typedef uint32_t                    SCALAR_UINT_TYPE;
            typedef float*                      SCALAR_TYPE_PTR;
            typedef SIMDMask16                  MASK_TYPE;
            typedef SIMDSwizzle16               SWIZZLE_MASK_TYPE;
        };

        template<>
        struct SIMDVecAVX2_f_traits<double, 8> {
            typedef SIMDVecAVX2_f<double, 4>   HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint64_t, 8> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int64_t, 8>  VEC_INT_TYPE;
            typedef int64_t                    SCALAR_INT_TYPE;
            typedef uint64_t                   SCALAR_UINT_TYPE;
            typedef double*                    SCALAR_TYPE_PTR;
            typedef SIMDMask8                  MASK_TYPE;
            typedef SIMDSwizzle8               SWIZZLE_MASK_TYPE;
        };

        // 1024b vectors
        template<>
        struct SIMDVecAVX2_f_traits<float, 32> {
            typedef SIMDVecAVX2_f<float, 16>    HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint32_t, 32> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 32> VEC_INT_TYPE;
            typedef int32_t                     SCALAR_INT_TYPE;
            typedef uint32_t                    SCALAR_UINT_TYPE;
            typedef float*                      SCALAR_TYPE_PTR;
            typedef SIMDMask32                  MASK_TYPE;
            typedef SIMDSwizzle32               SWIZZLE_MASK_TYPE;
        };

        template<>
        struct SIMDVecAVX2_f_traits<double, 16> {
            typedef SIMDVecAVX2_f<double, 8>    HALF_LEN_VEC_TYPE;
            typedef SIMDVecAVX2_u<uint64_t, 16> VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int64_t, 16>  VEC_INT_TYPE;
            typedef int64_t                     SCALAR_INT_TYPE;
            typedef uint64_t                    SCALAR_UINT_TYPE;
            typedef double*                     SCALAR_TYPE_PTR;
            typedef SIMDMask16                  MASK_TYPE;
            typedef SIMDSwizzle16               SWIZZLE_MASK_TYPE;
        };

        // ***************************************************************************
        // *
        // *    Implementation of floating point types SIMDx_32f and SIMDx_64f.
        // *
        // *    This implementation uses scalar emulation available through to 
        // *    SIMDVecFloatInterface.
        // *
        // ***************************************************************************
        template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
        class SIMDVecAVX2_f final :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE,
            VEC_LEN,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
            public SIMDVecPackableInterface<
            SIMDVecAVX2_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
        {
        public:
            typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, VEC_LEN>                            VEC_EMU_REG;

            typedef SIMDVecAVX2_f VEC_TYPE;
            typedef typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE    VEC_UINT_TYPE;
            typedef typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE     VEC_INT_TYPE;
            typedef typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE        MASK_TYPE;
        private:
            VEC_EMU_REG mVec;

        public:
            inline SIMDVecAVX2_f() : mVec() {};

            inline explicit SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i) : mVec(i) {};

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {

            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {

            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(SCALAR_FLOAT_TYPE const *p) { this->load(p); };

            inline SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1) {
                mVec.insert(0, i0);  mVec.insert(1, i1);
            }

            inline SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3) {
                mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
            }

            inline SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7)
            {
                mVec.insert(0, i0);  mVec.insert(1, i1);  mVec.insert(2, i2);  mVec.insert(3, i3);
                mVec.insert(4, i4);  mVec.insert(5, i5);  mVec.insert(6, i6);  mVec.insert(7, i7);
            }

            inline SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7,
                SCALAR_FLOAT_TYPE i8, SCALAR_FLOAT_TYPE i9, SCALAR_FLOAT_TYPE i10, SCALAR_FLOAT_TYPE i11, SCALAR_FLOAT_TYPE i12, SCALAR_FLOAT_TYPE i13, SCALAR_FLOAT_TYPE i14, SCALAR_FLOAT_TYPE i15)
            {
                mVec.insert(0, i0);    mVec.insert(1, i1);    mVec.insert(2, i2);    mVec.insert(3, i3);
                mVec.insert(4, i4);    mVec.insert(5, i5);    mVec.insert(6, i6);    mVec.insert(7, i7);
                mVec.insert(8, i8);    mVec.insert(9, i9);    mVec.insert(10, i10);  mVec.insert(11, i11);
                mVec.insert(12, i12);  mVec.insert(13, i13);  mVec.insert(14, i14);  mVec.insert(15, i15);
            }

            inline SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i0, SCALAR_FLOAT_TYPE i1, SCALAR_FLOAT_TYPE i2, SCALAR_FLOAT_TYPE i3, SCALAR_FLOAT_TYPE i4, SCALAR_FLOAT_TYPE i5, SCALAR_FLOAT_TYPE i6, SCALAR_FLOAT_TYPE i7,
                SCALAR_FLOAT_TYPE i8, SCALAR_FLOAT_TYPE i9, SCALAR_FLOAT_TYPE i10, SCALAR_FLOAT_TYPE i11, SCALAR_FLOAT_TYPE i12, SCALAR_FLOAT_TYPE i13, SCALAR_FLOAT_TYPE i14, SCALAR_FLOAT_TYPE i15,
                SCALAR_FLOAT_TYPE i16, SCALAR_FLOAT_TYPE i17, SCALAR_FLOAT_TYPE i18, SCALAR_FLOAT_TYPE i19, SCALAR_FLOAT_TYPE i20, SCALAR_FLOAT_TYPE i21, SCALAR_FLOAT_TYPE i22, SCALAR_FLOAT_TYPE i23,
                SCALAR_FLOAT_TYPE i24, SCALAR_FLOAT_TYPE i25, SCALAR_FLOAT_TYPE i26, SCALAR_FLOAT_TYPE i27, SCALAR_FLOAT_TYPE i28, SCALAR_FLOAT_TYPE i29, SCALAR_FLOAT_TYPE i30, SCALAR_FLOAT_TYPE i31)
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
            inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
                return mVec[index];
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, MASK_TYPE>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // insert[] (scalar)
            inline SIMDVecAVX2_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
                mVec.insert(index, value);
                return *this;
            }
        };

        // ***************************************************************************
        // *
        // *    Partial specialization of floating point SIMD for VEC_LEN == 1.
        // *    This specialization is necessary to eliminate PACK operations from
        // *    being used on SIMD1 types.
        // *
        // ***************************************************************************
        template<typename SCALAR_FLOAT_TYPE>
        class SIMDVecAVX2_f<SCALAR_FLOAT_TYPE, 1> final :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<SCALAR_FLOAT_TYPE, 1>,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE,
            1,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE,
            typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::SWIZZLE_MASK_TYPE>
        {
        public:
            typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, 1>                       VEC_EMU_REG;
            typedef typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE MASK_TYPE;

            typedef SIMDVecAVX2_f VEC_TYPE;
            typedef typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE    VEC_UINT_TYPE;
            typedef typename SIMDVecAVX2_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE     VEC_INT_TYPE;
        private:
            VEC_EMU_REG mVec;

        public:
            inline SIMDVecAVX2_f() : mVec() {};

            inline explicit SIMDVecAVX2_f(SCALAR_FLOAT_TYPE i) : mVec(i) {};

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {

            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {

            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(SCALAR_FLOAT_TYPE const *p) { this->load(p); }

            // Override Access operators
            inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
                return mVec[index];
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, SIMDMask1> operator[] (SIMDMask1 const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, SIMDMask1>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // insert[] (scalar)
            inline SIMDVecAVX2_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
                mVec.insert(index, value);
                return *this;
            }
        };

        // ********************************************************************************************
        // FLOATING POINT VECTOR specializations
        // ********************************************************************************************
        template<>
        class SIMDVecAVX2_f<float, 2> :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<float, 2>,
            SIMDVecAVX2_u<uint32_t, 2>,
            SIMDVecAVX2_i<int32_t, 2>,
            float,
            2,
            uint32_t,
            SIMDMask2,
            SIMDSwizzle2>,
            public SIMDVecPackableInterface<
            SIMDVecAVX2_f<float, 2>,
            SIMDVecAVX2_f<float, 1 >>
        {
        private:
            float mVec[2];

            typedef SIMDVecAVX2_u<uint32_t, 2>    VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 2>     VEC_INT_TYPE;
        public:

            constexpr static float alignment() {
                return 4;
            }

            // ZERO-CONSTR - Zero element constructor 
            inline SIMDVecAVX2_f() {}

            // SET-CONSTR  - One element constructor
            inline explicit SIMDVecAVX2_f(float f) {
                mVec[0] = f;
                mVec[1] = f;
            }

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {
                // TODO
            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {
                // TODO
            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(float const *p) {
                mVec[0] = p[0];
                mVec[1] = p[1];
            }

            // FULL-CONSTR - constructor with VEC_LEN scalar element 
            inline SIMDVecAVX2_f(float x_lo, float x_hi) {
                mVec[0] = x_lo;
                mVec[1] = x_hi;
            }

            // EXTRACT
            inline float extract(uint32_t index) const {
                return mVec[index & 1];
            }

            // EXTRACT
            inline float operator[] (uint32_t index) const {
                return mVec[index & 1];
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, SIMDMask2> operator[] (SIMDMask2 const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, SIMDMask2>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // INSERT
            inline SIMDVecAVX2_f & insert(uint32_t index, float value) {
                mVec[index & 1] = value;
            }
            // ****************************************************************************************
            // Overloading Interface functions starts here!
            // ****************************************************************************************

            //(Initialization)
            // ASSIGNV     - Assignment with another vector
            // MASSIGNV    - Masked assignment with another vector
            // ASSIGNS     - Assignment with scalar
            // MASSIGNS    - Masked assign with scalar

            //(Memory access)
            // LOAD    - Load from memory (either aligned or unaligned) to vector 
            // MLOAD   - Masked load from memory (either aligned or unaligned) to
            //        vector
            // LOADA   - Load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(float const * p) {
                mVec[0] = p[0];
                mVec[1] = p[1];
            }

            // MLOADA  - Masked load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(SIMDMask2 const & mask, float const * p) {
                if (mask.mMask[0] == true) mVec[0] = p[0];
                if (mask.mMask[1] == true) mVec[1] = p[1];
                return *this;
            }

            // STORE   - Store vector content into memory (either aligned or unaligned)
            // MSTORE  - Masked store vector content into memory (either aligned or
            //        unaligned)
            // STOREA  - Store vector content into aligned memory
            // MSTOREA - Masked store vector content into aligned memory

            //(Addition operations)
            // ADDV     - Add with vector 
            inline SIMDVecAVX2_f add(SIMDVecAVX2_f const & b) const {
                float t0 = mVec[0] + b.mVec[0];
                float t1 = mVec[1] + b.mVec[1];
                return SIMDVecAVX2_f(t0, t1);
            }

            inline SIMDVecAVX2_f operator+ (SIMDVecAVX2_f const & b) const {
                return add(b);
            }
            // MADDV    - Masked add with vector
            inline SIMDVecAVX2_f add(SIMDMask2 const & mask, SIMDVecAVX2_f const & b) const {
                float t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
                float t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
                return SIMDVecAVX2_f(t0, t1);
            }
            // ADDS     - Add with scalar
            inline SIMDVecAVX2_f add(float a) const {
                float t0 = mVec[0] + a;
                float t1 = mVec[1] + a;
                return SIMDVecAVX2_f(t0, t1);
            }
            // MADDS    - Masked add with scalar
            inline SIMDVecAVX2_f add(SIMDMask2 const & mask, float b) const {
                float t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
                float t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
                return SIMDVecAVX2_f(t0, t1);
            }
            // ADDVA    - Add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDVecAVX2_f const & b) {
                mVec[0] += b.mVec[0];
                mVec[1] += b.mVec[1];
                return *this;
            }
            // MADDVA   - Masked add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDMask2 const & mask, SIMDVecAVX2_f const & b) {
                mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
                mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
                return *this;
            }
            // ADDSA    - Add with scalar and assign
            inline SIMDVecAVX2_f & adda(float a) {
                mVec[0] += a;
                mVec[1] += a;
                return *this;
            }
            // MADDSA   - Masked add with scalar and assign
            inline SIMDVecAVX2_f & adda(SIMDMask2 const & mask, float b) {
                mVec[0] = mask.mMask[0] ? mVec[0] + b : mVec[0];
                mVec[1] = mask.mMask[1] ? mVec[1] + b : mVec[1];
                return *this;
            }
            // SADDV    - Saturated add with vector
            // MSADDV   - Masked saturated add with vector
            // SADDS    - Saturated add with scalar
            // MSADDS   - Masked saturated add with scalar
            // SADDVA   - Saturated add with vector and assign
            // MSADDVA  - Masked saturated add with vector and assign
            // SADDSA   - Satureated add with scalar and assign
            // MSADDSA  - Masked staturated add with vector and assign
            // POSTINC  - Postfix increment
            // MPOSTINC - Masked postfix increment
            // PREFINC  - Prefix increment
            inline SIMDVecAVX2_f & prefinc() {
                ++mVec[0];
                ++mVec[1];
                return *this;
            }
            // MPREFINC - Masked prefix increment
            inline SIMDVecAVX2_f & prefinc(SIMDMask2 const & mask) {
                if (mask.mMask[0] == true) ++mVec[0];
                if (mask.mMask[1] == true) ++mVec[1];
                return *this;
            }

            //(Subtraction operations)
            // SUBV       - Sub with vector
            // MSUBV      - Masked sub with vector
            // SUBS       - Sub with scalar
            // MSUBS      - Masked subtraction with scalar
            // SUBVA      - Sub with vector and assign
            // MSUBVA     - Masked sub with vector and assign
            // SUBSA      - Sub with scalar and assign
            // MSUBSA     - Masked sub with scalar and assign
            // SSUBV      - Saturated sub with vector
            // MSSUBV     - Masked saturated sub with vector
            // SSUBS      - Saturated sub with scalar
            // MSSUBS     - Masked saturated sub with scalar
            // SSUBVA     - Saturated sub with vector and assign
            // MSSUBVA    - Masked saturated sub with vector and assign
            // SSUBSA     - Saturated sub with scalar and assign
            // MSSUBSA    - Masked saturated sub with scalar and assign
            // SUBFROMV   - Sub from vector
            // MSUBFROMV  - Masked sub from vector
            // SUBFROMS   - Sub from scalar (promoted to vector)
            // MSUBFROMS  - Masked sub from scalar (promoted to vector)
            // SUBFROMVA  - Sub from vector and assign
            // MSUBFROMVA - Masked sub from vector and assign
            // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
            // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
            // POSTDEC    - Postfix decrement
            // MPOSTDEC   - Masked postfix decrement
            // PREFDEC    - Prefix decrement
            // MPREFDEC   - Masked prefix decrement

            //(Multiplication operations)
            // MULV   - Multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDVecAVX2_f const & b) const {
                float t0 = mVec[0] * b.mVec[0];
                float t1 = mVec[1] * b.mVec[1];
                return SIMDVecAVX2_f(t0, t1);
            }
            // MMULV  - Masked multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDMask2 const & mask, SIMDVecAVX2_f const & b) const {
                float t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
                float t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
                return SIMDVecAVX2_f(t0, t1);
            }
            // MULS   - Multiplication with scalar
            inline SIMDVecAVX2_f mul(float b) const {
                float t0 = mVec[0] * b;
                float t1 = mVec[1] * b;
                return SIMDVecAVX2_f(t0, t1);
            }
            // MMULS  - Masked multiplication with scalar
            inline SIMDVecAVX2_f mul(SIMDMask2 const & mask, float b) const {
                float t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
                float t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
                return SIMDVecAVX2_f(t0, t1);
            }
            // MULVA  - Multiplication with vector and assign
            // MMULVA - Masked multiplication with vector and assign
            // MULSA  - Multiplication with scalar and assign
            // MMULSA - Masked multiplication with scalar and assign

            //(Division operations)
            // DIVV   - Division with vector
            // MDIVV  - Masked division with vector
            // DIVS   - Division with scalar
            // MDIVS  - Masked division with scalar
            // DIVVA  - Division with vector and assign
            // MDIVVA - Masked division with vector and assign
            // DIVSA  - Division with scalar and assign
            // MDIVSA - Masked division with scalar and assign
            // RCP    - Reciprocal
            // MRCP   - Masked reciprocal
            // RCPS   - Reciprocal with scalar numerator
            // MRCPS  - Masked reciprocal with scalar
            // RCPA   - Reciprocal and assign
            // MRCPA  - Masked reciprocal and assign
            // RCPSA  - Reciprocal with scalar and assign
            // MRCPSA - Masked reciprocal with scalar and assign

            //(Comparison operations)
            // CMPEQV - Element-wise 'equal' with vector
            // CMPEQS - Element-wise 'equal' with scalar
            // CMPNEV - Element-wise 'not equal' with vector
            // CMPNES - Element-wise 'not equal' with scalar
            // CMPGTV - Element-wise 'greater than' with vector
            // CMPGTS - Element-wise 'greater than' with scalar
            // CMPLTV - Element-wise 'less than' with vector
            // CMPLTS - Element-wise 'less than' with scalar
            // CMPGEV - Element-wise 'greater than or equal' with vector
            // CMPGES - Element-wise 'greater than or equal' with scalar
            // CMPLEV - Element-wise 'less than or equal' with vector
            // CMPLES - Element-wise 'less than or equal' with scalar
            // CMPEX  - Check if vectors are exact (returns scalar 'bool')
            // (Pack/Unpack operations - not available for SIMD1)
            // PACK     - assign vector with two half-length vectors
            // PACKLO   - assign lower half of a vector with a half-length vector
            // PACKHI   - assign upper half of a vector with a half-length vector
            // UNPACK   - Unpack lower and upper halfs to half-length vectors.
            // UNPACKLO - Unpack lower half and return as a half-length vector.
            // UNPACKHI - Unpack upper half and return as a half-length vector.

            //(Blend/Swizzle operations)
            // BLENDV   - Blend (mix) two vectors
            // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
            //         assign
            // SWIZZLE  - Swizzle (reorder/permute) vector elements
            // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

            //(Reduction to scalar operations)
            // HADD  - Add elements of a vector (horizontal add)
            // MHADD - Masked add elements of a vector (horizontal add)
            // HMUL  - Multiply elements of a vector (horizontal mul)
            // MHMUL - Masked multiply elements of a vector (horizontal mul)

            //(Fused arithmetics)
            // FMULADDV  - Fused multiply and add (A*B + C) with vectors
            // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
            // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
            // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
            // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
            // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
            // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
            // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

            // (Mathematical operations)
            // MAXV   - Max with vector
            // MMAXV  - Masked max with vector
            // MAXS   - Max with scalar
            // MMAXS  - Masked max with scalar
            // MAXVA  - Max with vector and assign
            // MMAXVA - Masked max with vector and assign
            // MAXSA  - Max with scalar (promoted to vector) and assign
            // MMAXSA - Masked max with scalar (promoted to vector) and assign
            // MINV   - Min with vector
            // MMINV  - Masked min with vector
            // MINS   - Min with scalar (promoted to vector)
            // MMINS  - Masked min with scalar (promoted to vector)
            // MINVA  - Min with vector and assign
            // MMINVA - Masked min with vector and assign
            // MINSA  - Min with scalar (promoted to vector) and assign
            // MMINSA - Masked min with scalar (promoted to vector) and assign
            // HMAX   - Max of elements of a vector (horizontal max)
            // MHMAX  - Masked max of elements of a vector (horizontal max)
            // IMAX   - Index of max element of a vector
            // HMIN   - Min of elements of a vector (horizontal min)
            // MHMIN  - Masked min of elements of a vector (horizontal min)
            // IMIN   - Index of min element of a vector
            // MIMIN  - Masked index of min element of a vector

            // (Gather/Scatter operations)
            // GATHERS   - Gather from memory using indices from array
            // MGATHERS  - Masked gather from memory using indices from array
            // GATHERV   - Gather from memory using indices from vector
            // MGATHERV  - Masked gather from memory using indices from vector
            // SCATTERS  - Scatter to memory using indices from array
            // MSCATTERS - Masked scatter to memory using indices from array
            // SCATTERV  - Scatter to memory using indices from vector
            // MSCATTERV - Masked scatter to memory using indices from vector

            // 3) Operations available for Signed integer and Unsigned integer 
            // data types:

            //(Signed/Unsigned cast)
            // UTOI - Cast unsigned vector to signed vector
            // ITOU - Cast signed vector to unsigned vector

            // 4) Operations available for Signed integer and floating point SIMD types:

            // (Sign modification)
            // NEG   - Negate signed values
            // MNEG  - Masked negate signed values
            // NEGA  - Negate signed values and assign
            // MNEGA - Masked negate signed values and assign

            // (Mathematical functions)
            // ABS   - Absolute value
            // MABS  - Masked absolute value
            // ABSA  - Absolute value and assign
            // MABSA - Masked absolute value and assign

            // 5) Operations available for floating point SIMD types:

            // (Comparison operations)
            // CMPEQRV - Compare 'Equal within range' with margins from vector
            // CMPEQRS - Compare 'Equal within range' with scalar margin

            // (Mathematical functions)
            // SQR       - Square of vector values
            // MSQR      - Masked square of vector values
            // SQRA      - Square of vector values and assign
            // MSQRA     - Masked square of vector values and assign
            // SQRT      - Square root of vector values
            // MSQRT     - Masked square root of vector values 
            // SQRTA     - Square root of vector values and assign
            // MSQRTA    - Masked square root of vector values and assign
            // POWV      - Power (exponents in vector)
            // MPOWV     - Masked power (exponents in vector)
            // POWS      - Power (exponent in scalar)
            // MPOWS     - Masked power (exponent in scalar) 
            // ROUND     - Round to nearest integer
            // MROUND    - Masked round to nearest integer
            // TRUNC     - Truncate to integer (returns Signed integer vector)
            inline SIMDVecAVX2_i<int32_t, 2> trunc() {
                int32_t t0 = (int32_t)mVec[0];
                int32_t t1 = (int32_t)mVec[1];
                return SIMDVecAVX2_i<int32_t, 2>(t0, t1);
            }
            // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
            inline SIMDVecAVX2_i<int32_t, 2> trunc(SIMDMask2 const & mask) {
                int32_t t0 = mask.mMask[0] ? (int32_t)mVec[0] : 0;
                int32_t t1 = mask.mMask[1] ? (int32_t)mVec[1] : 0;
                return SIMDVecAVX2_i<int32_t, 2>(t0, t1);
            }
            // FLOOR     - Floor
            // MFLOOR    - Masked floor
            // CEIL      - Ceil
            // MCEIL     - Masked ceil
            // ISFIN     - Is finite
            // ISINF     - Is infinite (INF)
            // ISAN      - Is a number
            // ISNAN     - Is 'Not a Number (NaN)'
            // ISSUB     - Is subnormal
            // ISZERO    - Is zero
            // ISZEROSUB - Is zero or subnormal
            // SIN       - Sine
            // MSIN      - Masked sine
            // COS       - Cosine
            // MCOS      - Masked cosine
            // TAN       - Tangent
            // MTAN      - Masked tangent
            // CTAN      - Cotangent
            // MCTAN     - Masked cotangent
        };

        template<>
        class SIMDVecAVX2_f<float, 4> :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<float, 4>,
            SIMDVecAVX2_u<uint32_t, 4>,
            SIMDVecAVX2_i<int32_t, 4>,
            float,
            4,
            uint32_t,
            SIMDMask4,
            SIMDSwizzle4>,
            public SIMDVecPackableInterface<
            SIMDVecAVX2_f<float, 4>,
            SIMDVecAVX2_f<float, 2 >>
        {
        private:
            __m128 mVec;

            inline SIMDVecAVX2_f(__m128 const & x) {
                this->mVec = x;
            }

            typedef SIMDVecAVX2_u<uint32_t, 4>    VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 4>     VEC_INT_TYPE;
        public:
            // ZERO-CONSTR - Zero element constructor 
            inline SIMDVecAVX2_f() {}

            // SET-CONSTR  - One element constructor
            inline explicit SIMDVecAVX2_f(float f) {
                mVec = _mm_set1_ps(f);
            }

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {

            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {

            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(float const *p) { this->load(p); }

            // FULL-CONSTR - constructor with VEC_LEN scalar element 
            inline SIMDVecAVX2_f(float f0, float f1, float f2, float f3) {
                mVec = _mm_setr_ps(f0, f1, f2, f3);
            }

            // EXTRACT
            inline float extract(uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(16) float raw[4];
                _mm_store_ps(raw, mVec);
                return raw[index];
            }

            // EXTRACT
            inline float operator[] (uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                return extract(index);
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, SIMDMask4> operator[] (SIMDMask4 const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, SIMDMask4>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // INSERT
            inline SIMDVecAVX2_f & insert(uint32_t index, float value) {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(16) float raw[4];
                _mm_store_ps(raw, mVec);
                raw[index] = value;
                mVec = _mm_load_ps(raw);
                return *this;
            }
            // ****************************************************************************************
            // Overloading Interface functions starts here!
            // ****************************************************************************************

            //(Initialization)
            // ASSIGNV     - Assignment with another vector
            // MASSIGNV    - Masked assignment with another vector
            // ASSIGNS     - Assignment with scalar
            // MASSIGNS    - Masked assign with scalar

            //(Memory access)
            // LOAD    - Load from memory (either aligned or unaligned) to vector 
            // MLOAD   - Masked load from memory (either aligned or unaligned) to
            //        vector
            // LOADA   - Load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(float const * p) {
                mVec = _mm_load_ps(p);
                return *this;
            }
            // MLOADA  - Masked load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(SIMDMask4 const & mask, float const * p) {
                __m128 t0 = _mm_load_ps(p);
                mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
                return *this;
            }

            // STORE   - Store vector content into memory (either aligned or unaligned)
            // MSTORE  - Masked store vector content into memory (either aligned or
            //        unaligned)
            // STOREA  - Store vector content into aligned memory
            // MSTOREA - Masked store vector content into aligned memory

            //(Addition operations)
            // ADDV     - Add with vector 
            inline SIMDVecAVX2_f add(SIMDVecAVX2_f const & b) const {
                __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
                return SIMDVecAVX2_f(t0);
            }

            inline SIMDVecAVX2_f operator+ (SIMDVecAVX2_f const & b) const {
                return add(b);
            }
            // MADDV    - Masked add with vector
            inline SIMDVecAVX2_f add(SIMDMask4 const & mask, SIMDVecAVX2_f const & b) const {
                __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
                return SIMDVecAVX2_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
            }
            // ADDS     - Add with scalar
            inline SIMDVecAVX2_f add(float a) const {
                return SIMDVecAVX2_f(_mm_add_ps(this->mVec, _mm_set1_ps(a)));
            }
            // MADDS    - Masked add with scalar
            inline SIMDVecAVX2_f add(SIMDMask4 const & mask, float a) const {
                __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(a));
                return SIMDVecAVX2_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
            }
            // ADDVA    - Add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDVecAVX2_f const & b) {
                mVec = _mm_add_ps(this->mVec, b.mVec);
                return *this;
            }
            // MADDVA   - Masked add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDMask4 const & mask, SIMDVecAVX2_f const & b) {
                __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
                mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
                return *this;
            }
            // ADDSA    - Add with scalar and assign
            inline SIMDVecAVX2_f & adda(float a) {
                mVec = _mm_add_ps(this->mVec, _mm_set1_ps(a));
                return *this;
            }
            // MADDSA   - Masked add with scalar and assign
            inline SIMDVecAVX2_f & adda(SIMDMask4 const & mask, float a) {
                __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(a));
                mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
                return *this;
            }
            // SADDV    - Saturated add with vector
            // MSADDV   - Masked saturated add with vector
            // SADDS    - Saturated add with scalar
            // MSADDS   - Masked saturated add with scalar
            // SADDVA   - Saturated add with vector and assign
            // MSADDVA  - Masked saturated add with vector and assign
            // SADDSA   - Satureated add with scalar and assign
            // MSADDSA  - Masked staturated add with vector and assign
            // POSTINC  - Postfix increment
            // MPOSTINC - Masked postfix increment
            // PREFINC  - Prefix increment
            // MPREFINC - Masked prefix increment

            //(Subtraction operations)
            // SUBV       - Sub with vector
            // MSUBV      - Masked sub with vector
            // SUBS       - Sub with scalar
            // MSUBS      - Masked subtraction with scalar
            // SUBVA      - Sub with vector and assign
            // MSUBVA     - Masked sub with vector and assign
            // SUBSA      - Sub with scalar and assign
            // MSUBSA     - Masked sub with scalar and assign
            // SSUBV      - Saturated sub with vector
            // MSSUBV     - Masked saturated sub with vector
            // SSUBS      - Saturated sub with scalar
            // MSSUBS     - Masked saturated sub with scalar
            // SSUBVA     - Saturated sub with vector and assign
            // MSSUBVA    - Masked saturated sub with vector and assign
            // SSUBSA     - Saturated sub with scalar and assign
            // MSSUBSA    - Masked saturated sub with scalar and assign
            // SUBFROMV   - Sub from vector
            // MSUBFROMV  - Masked sub from vector
            // SUBFROMS   - Sub from scalar (promoted to vector)
            // MSUBFROMS  - Masked sub from scalar (promoted to vector)
            // SUBFROMVA  - Sub from vector and assign
            // MSUBFROMVA - Masked sub from vector and assign
            // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
            // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
            // POSTDEC    - Postfix decrement
            // MPOSTDEC   - Masked postfix decrement
            // PREFDEC    - Prefix decrement
            // MPREFDEC   - Masked prefix decrement

            //(Multiplication operations)
            // MULV   - Multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDVecAVX2_f const & b) {
                __m128 t0 = _mm_mul_ps(mVec, b.mVec);
                return SIMDVecAVX2_f(t0);
            }
            // MMULV  - Masked multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDMask4 const & mask, SIMDVecAVX2_f const & b) {
                __m128 t0 = _mm_mul_ps(mVec, b.mVec);
                __m128 t1 = _mm_castsi128_ps(mask.mMask);
                __m128 t2 = _mm_blendv_ps(mVec, t0, t1);
                return SIMDVecAVX2_f(t2);
            }
            // MULS   - Multiplication with scalar
            inline SIMDVecAVX2_f mul(float b) {
                __m128 t0 = _mm_set1_ps(b);
                __m128 t1 = _mm_mul_ps(mVec, t0);
                return SIMDVecAVX2_f(t1);
            }
            // MMULS  - Masked multiplication with scalar
            inline SIMDVecAVX2_f mul(SIMDMask4 const & mask, float b) {
                __m128 t0 = _mm_set1_ps(b);
                __m128 t1 = _mm_mul_ps(mVec, t0);
                __m128 t2 = _mm_castsi128_ps(mask.mMask);
                __m128 t3 = _mm_blendv_ps(mVec, t1, t2);
                return SIMDVecAVX2_f(t3);
            }
            // MULVA  - Multiplication with vector and assign
            // MMULVA - Masked multiplication with vector and assign
            // MULSA  - Multiplication with scalar and assign
            // MMULSA - Masked multiplication with scalar and assign

            //(Division operations)
            // DIVV   - Division with vector
            // MDIVV  - Masked division with vector
            // DIVS   - Division with scalar
            // MDIVS  - Masked division with scalar
            // DIVVA  - Division with vector and assign
            // MDIVVA - Masked division with vector and assign
            // DIVSA  - Division with scalar and assign
            // MDIVSA - Masked division with scalar and assign
            // RCP    - Reciprocal
            // MRCP   - Masked reciprocal
            // RCPS   - Reciprocal with scalar numerator
            // MRCPS  - Masked reciprocal with scalar
            // RCPA   - Reciprocal and assign
            // MRCPA  - Masked reciprocal and assign
            // RCPSA  - Reciprocal with scalar and assign
            // MRCPSA - Masked reciprocal with scalar and assign

            //(Comparison operations)
            // CMPEQV - Element-wise 'equal' with vector
            // CMPEQS - Element-wise 'equal' with scalar
            // CMPNEV - Element-wise 'not equal' with vector
            // CMPNES - Element-wise 'not equal' with scalar
            // CMPGTV - Element-wise 'greater than' with vector
            // CMPGTS - Element-wise 'greater than' with scalar
            // CMPLTV - Element-wise 'less than' with vector
            // CMPLTS - Element-wise 'less than' with scalar
            // CMPGEV - Element-wise 'greater than or equal' with vector
            // CMPGES - Element-wise 'greater than or equal' with scalar
            // CMPLEV - Element-wise 'less than or equal' with vector
            // CMPLES - Element-wise 'less than or equal' with scalar
            // CMPEX  - Check if vectors are exact (returns scalar 'bool')
            // (Pack/Unpack operations - not available for SIMD1)
            // PACK     - assign vector with two half-length vectors
            // PACKLO   - assign lower half of a vector with a half-length vector
            // PACKHI   - assign upper half of a vector with a half-length vector
            // UNPACK   - Unpack lower and upper halfs to half-length vectors.
            // UNPACKLO - Unpack lower half and return as a half-length vector.
            // UNPACKHI - Unpack upper half and return as a half-length vector.

            //(Blend/Swizzle operations)
            // BLENDV   - Blend (mix) two vectors
            // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
            //         assign
            // SWIZZLE  - Swizzle (reorder/permute) vector elements
            // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

            //(Reduction to scalar operations)
            // HADD  - Add elements of a vector (horizontal add)
            // MHADD - Masked add elements of a vector (horizontal add)
            // HMUL  - Multiply elements of a vector (horizontal mul)
            // MHMUL - Masked multiply elements of a vector (horizontal mul)

            //(Fused arithmetics)
            // FMULADDV  - Fused multiply and add (A*B + C) with vectors
            // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
            // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
            // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
            // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
            // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
            // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
            // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

            // (Mathematical operations)
            // MAXV   - Max with vector
            // MMAXV  - Masked max with vector
            // MAXS   - Max with scalar
            // MMAXS  - Masked max with scalar
            // MAXVA  - Max with vector and assign
            // MMAXVA - Masked max with vector and assign
            // MAXSA  - Max with scalar (promoted to vector) and assign
            // MMAXSA - Masked max with scalar (promoted to vector) and assign
            // MINV   - Min with vector
            // MMINV  - Masked min with vector
            // MINS   - Min with scalar (promoted to vector)
            // MMINS  - Masked min with scalar (promoted to vector)
            // MINVA  - Min with vector and assign
            // MMINVA - Masked min with vector and assign
            // MINSA  - Min with scalar (promoted to vector) and assign
            // MMINSA - Masked min with scalar (promoted to vector) and assign
            // HMAX   - Max of elements of a vector (horizontal max)
            // MHMAX  - Masked max of elements of a vector (horizontal max)
            // IMAX   - Index of max element of a vector
            // HMIN   - Min of elements of a vector (horizontal min)
            // MHMIN  - Masked min of elements of a vector (horizontal min)
            // IMIN   - Index of min element of a vector
            // MIMIN  - Masked index of min element of a vector

            // (Gather/Scatter operations)
            // GATHERS   - Gather from memory using indices from array
            // MGATHERS  - Masked gather from memory using indices from array
            // GATHERV   - Gather from memory using indices from vector
            // MGATHERV  - Masked gather from memory using indices from vector
            // SCATTERS  - Scatter to memory using indices from array
            // MSCATTERS - Masked scatter to memory using indices from array
            // SCATTERV  - Scatter to memory using indices from vector
            // MSCATTERV - Masked scatter to memory using indices from vector

            // 3) Operations available for Signed integer and Unsigned integer 
            // data types:

            //(Signed/Unsigned cast)
            // UTOI - Cast unsigned vector to signed vector
            // ITOU - Cast signed vector to unsigned vector

            // 4) Operations available for Signed integer and floating point SIMD types:

            // (Sign modification)
            // NEG   - Negate signed values
            // MNEG  - Masked negate signed values
            // NEGA  - Negate signed values and assign
            // MNEGA - Masked negate signed values and assign

            // (Mathematical functions)
            // ABS   - Absolute value
            // MABS  - Masked absolute value
            // ABSA  - Absolute value and assign
            // MABSA - Masked absolute value and assign

            // 5) Operations available for floating point SIMD types:

            // (Comparison operations)
            // CMPEQRV - Compare 'Equal within range' with margins from vector
            // CMPEQRS - Compare 'Equal within range' with scalar margin

            // (Mathematical functions)
            // SQR       - Square of vector values
            // MSQR      - Masked square of vector values
            // SQRA      - Square of vector values and assign
            // MSQRA     - Masked square of vector values and assign
            // SQRT      - Square root of vector values
            // MSQRT     - Masked square root of vector values 
            // SQRTA     - Square root of vector values and assign
            // MSQRTA    - Masked square root of vector values and assign
            // POWV      - Power (exponents in vector)
            // MPOWV     - Masked power (exponents in vector)
            // POWS      - Power (exponent in scalar)
            // MPOWS     - Masked power (exponent in scalar) 
            // ROUND     - Round to nearest integer
            // MROUND    - Masked round to nearest integer
            // TRUNC     - Truncate to integer (returns Signed integer vector)
            SIMDVecAVX2_i<int32_t, 4> trunc() {
                __m128i t0 = _mm_cvttps_epi32(mVec);
                return SIMDVecAVX2_i<int32_t, 4>(t0);
            }
            // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
            SIMDVecAVX2_i<int32_t, 4> trunc(SIMDMask4 const & mask) {
                __m128 t0 = _mm_castsi128_ps(mask.mMask);
                __m128 t1 = _mm_setzero_ps();
                __m128i t2 = _mm_cvttps_epi32(_mm_blendv_ps(t1, mVec, t0));
                return SIMDVecAVX2_i<int32_t, 4>(t2);
            }
            // FLOOR     - Floor
            // MFLOOR    - Masked floor
            // CEIL      - Ceil
            // MCEIL     - Masked ceil
            // ISFIN     - Is finite
            // ISINF     - Is infinite (INF)
            // ISAN      - Is a number
            // ISNAN     - Is 'Not a Number (NaN)'
            // ISSUB     - Is subnormal
            // ISZERO    - Is zero
            // ISZEROSUB - Is zero or subnormal
            // SIN       - Sine
            // MSIN      - Masked sine
            // COS       - Cosine
            // MCOS      - Masked cosine
            // TAN       - Tangent
            // MTAN      - Masked tangent
            // CTAN      - Cotangent
            // MCTAN     - Masked cotangent
        };

        template<>
        class SIMDVecAVX2_f<float, 8> :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<float, 8>,
            SIMDVecAVX2_u<uint32_t, 8>,
            SIMDVecAVX2_i<int32_t, 8>,
            float,
            8,
            uint32_t,
            SIMDMask8,
            SIMDSwizzle8>,
            public SIMDVecPackableInterface<
            SIMDVecAVX2_f<float, 8>,
            SIMDVecAVX2_f<float, 4 >>
        {
        private:
            __m256 mVec;

            inline SIMDVecAVX2_f(__m256 const & x) {
                this->mVec = x;
            }

            typedef SIMDVecAVX2_u<uint32_t, 8>    VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 8>     VEC_INT_TYPE;

        public:
            // ZERO-CONSTR - Zero element constructor 
            inline SIMDVecAVX2_f() {}

            // SET-CONSTR  - One element constructor
            inline explicit SIMDVecAVX2_f(float f) {
                mVec = _mm256_set1_ps(f);
            }

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {

            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {

            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(float const *p) { this->load(p); }

            // FULL-CONSTR - constructor with VEC_LEN scalar element 
            inline SIMDVecAVX2_f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
                mVec = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
            }

            // EXTRACT
            inline float extract(uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(32) float raw[8];
                _mm256_store_ps(raw, mVec);
                return raw[index];
            }

            // EXTRACT
            inline float operator[] (uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                return extract(index);
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, SIMDMask8> operator[] (SIMDMask8 const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, SIMDMask8>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // INSERT
            inline SIMDVecAVX2_f & insert(uint32_t index, float value) {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(32) float raw[8];
                _mm256_store_ps(raw, mVec);
                raw[index] = value;
                mVec = _mm256_load_ps(raw);
                return *this;
            }

            // ****************************************************************************************
            // Overloading Interface functions starts here!
            // ****************************************************************************************



            // ****************************************************************************************
            // Overloading Interface functions starts here!
            // ****************************************************************************************

            //(Initialization)
            // ASSIGNV     - Assignment with another vector
            // MASSIGNV    - Masked assignment with another vector
            // ASSIGNS     - Assignment with scalar
            // MASSIGNS    - Masked assign with scalar

            //(Memory access)
            // LOAD    - Load from memory (either aligned or unaligned) to vector
            // MLOAD   - Masked load from memory (either aligned or unaligned) to
            //           vector
            // LOADA   - Load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(float const * p) {
                mVec = _mm256_load_ps(p);
                return *this;
            }
            // MLOADA  - Masked load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(SIMDMask8 const & mask, float const * p) {
                __m256 t0 = _mm256_load_ps(p);
                mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
                return *this;
            }
            // STORE   - Store vector content into memory (either aligned or unaligned)
            // MSTORE  - Masked store vector content into memory (either aligned or
            //           unaligned)
            // STOREA  - Store vector content into aligned memory
            inline float* storea(float* p) {
                _mm256_store_ps(p, mVec);
                return p;
            }
            // MSTOREA - Masked store vector content into aligned memory
            inline float* storea(SIMDMask8 const & mask, float* p) {
                _mm256_maskstore_ps(p, mask.mMask, mVec);
                return p;
            }

            //(Addition operations)
            // ADDV     - Add with vector
            inline SIMDVecAVX2_f add(SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(this->mVec, b.mVec);
                return SIMDVecAVX2_f(t0);
            }
            // MADDV    - Masked add with vector
            inline SIMDVecAVX2_f add(SIMDMask8 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(this->mVec, b.mVec);
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
            }
            // ADDS     - Add with scalar
            inline SIMDVecAVX2_f add(float b) {
                return SIMDVecAVX2_f(_mm256_add_ps(this->mVec, _mm256_set1_ps(b)));
            }
            // MADDS    - Masked add with scalar
            inline SIMDVecAVX2_f add(SIMDMask8 const & mask, float b) {
                __m256 t0 = _mm256_add_ps(this->mVec, _mm256_set1_ps(b));
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
            }
            // ADDVA    - Add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDVecAVX2_f const & b) {
                mVec = _mm256_add_ps(this->mVec, b.mVec);
                return *this;
            }
            // MADDVA   - Masked add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDMask8 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(this->mVec, b.mVec);
                mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
                return *this;
            }
            // ADDSA    - Add with scalar and assign
            inline SIMDVecAVX2_f & adda(float b) {
                mVec = _mm256_add_ps(this->mVec, _mm256_set1_ps(b));
                return *this;
            }
            // MADDSA   - Masked add with scalar and assign
            inline SIMDVecAVX2_f & adda(SIMDMask8 const & mask, float b) {
                __m256 t0 = _mm256_add_ps(this->mVec, _mm256_set1_ps(b));
                mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
                return *this;
            }
            // SADDV    - Saturated add with vector
            // MSADDV   - Masked saturated add with vector
            // SADDS    - Saturated add with scalar
            // MSADDS   - Masked saturated add with scalar
            // SADDVA   - Saturated add with vector and assign
            // MSADDVA  - Masked saturated add with vector and assign
            // SADDSA   - Satureated add with scalar and assign
            // MSADDSA  - Masked staturated add with vector and assign
            // POSTINC  - Postfix increment
            // MPOSTINC - Masked postfix increment
            // PREFINC  - Prefix increment
            // MPREFINC - Masked prefix increment

            //(Subtraction operations)
            // SUBV       - Sub with vector
            // MSUBV      - Masked sub with vector
            // SUBS       - Sub with scalar
            // MSUBS      - Masked subtraction with scalar
            // SUBVA      - Sub with vector and assign
            // MSUBVA     - Masked sub with vector and assign
            // SUBSA      - Sub with scalar and assign
            // MSUBSA     - Masked sub with scalar and assign
            // SSUBV      - Saturated sub with vector
            // MSSUBV     - Masked saturated sub with vector
            // SSUBS      - Saturated sub with scalar
            // MSSUBS     - Masked saturated sub with scalar
            // SSUBVA     - Saturated sub with vector and assign
            // MSSUBVA    - Masked saturated sub with vector and assign
            // SSUBSA     - Saturated sub with scalar and assign
            // MSSUBSA    - Masked saturated sub with scalar and assign
            // SUBFROMV   - Sub from vector
            // MSUBFROMV  - Masked sub from vector
            // SUBFROMS   - Sub from scalar (promoted to vector)
            // MSUBFROMS  - Masked sub from scalar (promoted to vector)
            // SUBFROMVA  - Sub from vector and assign
            // MSUBFROMVA - Masked sub from vector and assign
            // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
            // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
            // POSTDEC    - Postfix decrement
            // MPOSTDEC   - Masked postfix decrement
            // PREFDEC    - Prefix decrement
            // MPREFDEC   - Masked prefix decrement

            //(Multiplication operations)
            // MULV   - Multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDVecAVX2_f const & b) {
                return SIMDVecAVX2_f(_mm256_mul_ps(this->mVec, b.mVec));
            }
            // MMULV  - Masked multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDMask8 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_mul_ps(this->mVec, b.mVec);
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
            }
            // MULS   - Multiplication with scalar
            inline SIMDVecAVX2_f mul(float b) {
                return SIMDVecAVX2_f(_mm256_mul_ps(this->mVec, _mm256_set1_ps(b)));
            }
            // MMULS  - Masked multiplication with scalar
            inline SIMDVecAVX2_f mul(SIMDMask8 const & mask, float b) {
                __m256 t0 = _mm256_mul_ps(this->mVec, _mm256_set1_ps(b));
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
            }
            // MULVA  - Multiplication with vector and assign
            // MMULVA - Masked multiplication with vector and assign
            // MULSA  - Multiplication with scalar and assign
            // MMULSA - Masked multiplication with scalar and assign

            //(Division operations)
            // DIVV   - Division with vector
            // MDIVV  - Masked division with vector
            // DIVS   - Division with scalar
            // MDIVS  - Masked division with scalar
            // DIVVA  - Division with vector and assign
            // MDIVVA - Masked division with vector and assign
            // DIVSA  - Division with scalar and assign
            // MDIVSA - Masked division with scalar and assign
            // RCP    - Reciprocal
            inline SIMDVecAVX2_f rcp() {
                return SIMDVecAVX2_f(_mm256_rcp_ps(mVec));
            }
            // MRCP   - Masked reciprocal
            inline SIMDVecAVX2_f rcp(SIMDMask8 const & mask) {
                __m256 t0 = _mm256_rcp_ps(mVec);
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
            }
            // RCPS   - Reciprocal with scalar numerator
            inline SIMDVecAVX2_f rcp(float b) {
                __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec), _mm256_set1_ps(b));
                return SIMDVecAVX2_f(t0);
            }
            // MRCPS  - Masked reciprocal with scalar
            inline SIMDVecAVX2_f rcp(SIMDMask8 const & mask, float b) {
                __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec), _mm256_set1_ps(b));
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
            }
            // RCPA   - Reciprocal and assign
            // MRCPA  - Masked reciprocal and assign
            // RCPSA  - Reciprocal with scalar and assign
            // MRCPSA - Masked reciprocal with scalar and assign

            //(Comparison operations)
            // CMPEQV - Element-wise 'equal' with vector
            // CMPEQS - Element-wise 'equal' with scalar
            // CMPNEV - Element-wise 'not equal' with vector
            // CMPNES - Element-wise 'not equal' with scalar
            // CMPGTV - Element-wise 'greater than' with vector
            // CMPGTS - Element-wise 'greater than' with scalar
            // CMPLTV - Element-wise 'less than' with vector
            // CMPLTS - Element-wise 'less than' with scalar
            // CMPGEV - Element-wise 'greater than or equal' with vector
            // CMPGES - Element-wise 'greater than or equal' with scalar
            // CMPLEV - Element-wise 'less than or equal' with vector
            // CMPLES - Element-wise 'less than or equal' with scalar
            // CMPEX  - Check if vectors are exact (returns scalar 'bool')

            // (Pack/Unpack operations - not available for SIMD1)
            // PACK     - assign vector with two half-length vectors
            // PACKLO   - assign lower half of a vector with a half-length vector
            // PACKHI   - assign upper half of a vector with a half-length vector
            // UNPACK   - Unpack lower and upper halfs to half-length vectors.
            // UNPACKLO - Unpack lower half and return as a half-length vector.
            // UNPACKHI - Unpack upper half and return as a half-length vector.

            //(Blend/Swizzle operations)
            // BLENDV   - Blend (mix) two vectors
            // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
            // assign
            // SWIZZLE  - Swizzle (reorder/permute) vector elements
            // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

            //(Reduction to scalar operations)
            // HADD  - Add elements of a vector (horizontal add)
            // MHADD - Masked add elements of a vector (horizontal add)
            // HMUL  - Multiply elements of a vector (horizontal mul)
            // MHMUL - Masked multiply elements of a vector (horizontal mul)

            //(Fused arithmetics)
            // FMULADDV  - Fused multiply and add (A*B + C) with vectors        
            inline SIMDVecAVX2_f fmuladd(SIMDVecAVX2_f const & a, SIMDVecAVX2_f const & b) {
#ifdef FMA
                return _mm256_fmadd_ps(this->mVec, a.mVec, b.mVec);
#else
                return _mm256_add_ps(_mm256_mul_ps(this->mVec, a.mVec), b.mVec);
#endif
            }

            // MFMULADDV
            inline SIMDVecAVX2_f fmuladd(SIMDMask8 const & mask, SIMDVecAVX2_f const & a, SIMDVecAVX2_f const & b) {
#ifdef FMA
                __m256 t0 = _mm256_fmadd_ps(this->mVec, a.mVec, b.mVec);
                return _mm256_blendv_ps(this->mVec, t0, _mm256_cvtepi32_ps(mask.mMask));
#else
                __m256 t0 = _mm256_add_ps(_mm256_mul_ps(this->mVec, a.mVec), b.mVec);
                return _mm256_blendv_ps(this->mVec, t0, _mm256_cvtepi32_ps(mask.mMask));
#endif
            }
            // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
            // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
            // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
            // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
            // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
            // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

            // (Mathematical operations)
            // MAXV   - Max with vector
            // MMAXV  - Masked max with vector
            // MAXS   - Max with scalar
            // MMAXS  - Masked max with scalar
            // MAXVA  - Max with vector and assign
            // MMAXVA - Masked max with vector and assign
            // MAXSA  - Max with scalar (promoted to vector) and assign
            // MMAXSA - Masked max with scalar (promoted to vector) and assign
            // MINV   - Min with vector
            // MMINV  - Masked min with vector
            // MINS   - Min with scalar (promoted to vector)
            // MMINS  - Masked min with scalar (promoted to vector)
            // MINVA  - Min with vector and assign
            // MMINVA - Masked min with vector and assign
            // MINSA  - Min with scalar (promoted to vector) and assign
            // MMINSA - Masked min with scalar (promoted to vector) and assign
            // HMAX   - Max of elements of a vector (horizontal max)
            // MHMAX  - Masked max of elements of a vector (horizontal max)
            // IMAX   - Index of max element of a vector
            // HMIN   - Min of elements of a vector (horizontal min)
            // MHMIN  - Masked min of elements of a vector (horizontal min)
            // IMIN   - Index of min element of a vector
            // MIMIN  - Masked index of min element of a vector

            // (Gather/Scatter operations)
            // GATHERS   - Gather from memory using indices from array
            // MGATHERS  - Masked gather from memory using indices from array
            // GATHERV   - Gather from memory using indices from vector
            // MGATHERV  - Masked gather from memory using indices from vector
            // SCATTERS  - Scatter to memory using indices from array
            // MSCATTERS - Masked scatter to memory using indices from array
            // SCATTERV  - Scatter to memory using indices from vector
            // MSCATTERV - Masked scatter to memory using indices from vector

            // 3) Operations available for Signed integer and Unsigned integer 
            // data types:

            //(Signed/Unsigned cast)
            // UTOI - Cast unsigned vector to signed vector
            // ITOU - Cast signed vector to unsigned vector

            // 4) Operations available for Signed integer and floating point SIMD types:

            // (Sign modification)
            // NEG   - Negate signed values
            // MNEG  - Masked negate signed values
            // NEGA  - Negate signed values and assign
            // MNEGA - Masked negate signed values and assign

            // (Mathematical functions)
            // ABS   - Absolute value
            // MABS  - Masked absolute value
            // ABSA  - Absolute value and assign
            // MABSA - Masked absolute value and assign

            // 5) Operations available for floating point SIMD types:

            // (Comparison operations)
            // CMPEQRV - Compare 'Equal within range' with margins from vector
            // CMPEQRS - Compare 'Equal within range' with scalar margin

            // (Mathematical functions)
            // SQR       - Square of vector values
            // MSQR      - Masked square of vector values
            // SQRA      - Square of vector values and assign
            // MSQRA     - Masked square of vector values and assign
            // SQRT      - Square root of vector values
            SIMDVecAVX2_f sqrt() {
                return SIMDVecAVX2_f(_mm256_sqrt_ps(mVec));
            }
            // MSQRT     - Masked square root of vector values 
            SIMDVecAVX2_f sqrt(SIMDMask8 const & mask) {
                __m256 mask_ps = _mm256_castsi256_ps(mask.mMask);
                __m256 ret = _mm256_sqrt_ps(mVec);
                return SIMDVecAVX2_f(_mm256_blendv_ps(mVec, ret, mask_ps));
            }
            // SQRTA     - Square root of vector values and assign
            // MSQRTA    - Masked square root of vector values and assign
            // POWV      - Power (exponents in vector)
            // MPOWV     - Masked power (exponents in vector)
            // POWS      - Power (exponent in scalar)
            // MPOWS     - Masked power (exponent in scalar) 
            // ROUND     - Round to nearest integer
            // MROUND    - Masked round to nearest integer
            // TRUNC     - Truncate to integer (returns Signed integer vector)
            SIMDVecAVX2_i<int32_t, 8> trunc() {
                __m256i t0 = _mm256_cvttps_epi32(mVec);
                return SIMDVecAVX2_i<int32_t, 8>(t0);
            }
            // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
            SIMDVecAVX2_i<int32_t, 8> trunc(SIMDMask8 const & mask) {
                __m256 mask_ps = _mm256_castsi256_ps(mask.mMask);
                __m256 t0 = _mm256_setzero_ps();
                __m256i t1 = _mm256_cvttps_epi32(_mm256_blendv_ps(t0, mVec, mask_ps));
                return SIMDVecAVX2_i<int32_t, 8>(t1);
            }
            // FLOOR     - Floor
            // MFLOOR    - Masked floor
            // CEIL      - Ceil
            // MCEIL     - Masked ceil
            // ISFIN     - Is finite
            // ISINF     - Is infinite (INF)
            // ISAN      - Is a number
            // ISNAN     - Is 'Not a Number (NaN)'
            // ISSUB     - Is subnormal
            // ISZERO    - Is zero
            // ISZEROSUB - Is zero or subnormal
            // SIN       - Sine
            // MSIN      - Masked sine
            // COS       - Cosine
            // MCOS      - Masked cosine
            // TAN       - Tangent
            // MTAN      - Masked tangent
            // CTAN      - Cotangent
            // MCTAN     - Masked cotangent
        };

        template<>
        class SIMDVecAVX2_f<float, 16> :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<float, 16>,
            SIMDVecAVX2_u<uint32_t, 16>,
            SIMDVecAVX2_i<int32_t, 16>,
            float,
            16,
            uint32_t,
            SIMDMask16,
            SIMDSwizzle16>,
            public SIMDVecPackableInterface<
            SIMDVecAVX2_f<float, 16>,
            SIMDVecAVX2_f<float, 8 >>
        {
        private:
            __m256 mVecLo;
            __m256 mVecHi;

            inline SIMDVecAVX2_f(__m256 const & xLo, __m256 const & xHi) {
                this->mVecLo = xLo;
                this->mVecHi = xHi;
            }

            typedef SIMDVecAVX2_u<uint32_t, 16>    VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 16>     VEC_INT_TYPE;
        public:
            // ZERO-CONSTR - Zero element constructor 
            inline SIMDVecAVX2_f() {}

            // SET-CONSTR  - One element constructor
            inline explicit SIMDVecAVX2_f(float f) {
                mVecLo = _mm256_set1_ps(f);
                mVecHi = _mm256_set1_ps(f);
            }

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {

            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {

            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(float const *p) { this->load(p); };

            // FULL-CONSTR - constructor with VEC_LEN scalar element 
            inline SIMDVecAVX2_f(float f0, float f1, float f2, float f3,
                float f4, float f5, float f6, float f7,
                float f8, float f9, float f10, float f11,
                float f12, float f13, float f14, float f15) {
                mVecLo = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
                mVecHi = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
            }

            // EXTRACT - Extract single element from a vector
            inline float extract(uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(32) float raw[8];
                if (index < 8) {
                    _mm256_store_ps(raw, mVecLo);
                    return raw[index];
                }
                else {
                    _mm256_store_ps(raw, mVecHi);
                    return raw[index - 8];
                }
            }

            // EXTRACT - Extract single element from a vector
            inline float operator[] (uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                return extract(index);
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, SIMDMask16> operator[] (SIMDMask16 const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, SIMDMask16>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // INSERT  - Insert single element into a vector
            inline SIMDVecAVX2_f & insert(uint32_t index, float value) {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(32) float raw[8];
                if (index < 8) {
                    _mm256_store_ps(raw, mVecLo);
                    raw[index] = value;
                    mVecLo = _mm256_load_ps(raw);
                }
                else {
                    _mm256_store_ps(raw, mVecHi);
                    raw[index - 8] = value;
                    mVecHi = _mm256_load_ps(raw);
                }

                return *this;
            }

            //(Initialization)
            // ASSIGNV     - Assignment with another vector
            // MASSIGNV    - Masked assignment with another vector
            // ASSIGNS     - Assignment with scalar
            // MASSIGNS    - Masked assign with scalar

            //(Memory access)
            // LOAD    - Load from memory (either aligned or unaligned) to vector 
            inline SIMDVecAVX2_f & load(float const * p) {
                mVecLo = _mm256_loadu_ps(p);
                mVecHi = _mm256_loadu_ps(p + 8);
                return *this;
            }
            // MLOAD   - Masked load from memory (either aligned or unaligned) to
            //        vector
            inline SIMDVecAVX2_f & load(SIMDMask16 const & mask, float const * p) {
                __m256 t0 = _mm256_loadu_ps(p);
                __m256 t1 = _mm256_loadu_ps(p + 8);
                mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                mVecHi = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return *this;
            }
            // LOADA   - Load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(float const * p) {
                mVecLo = _mm256_load_ps(p);
                mVecHi = _mm256_load_ps(p + 8);
                return *this;
            }

            // MLOADA  - Masked load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(SIMDMask16 const & mask, float const * p) {
                __m256 t0 = _mm256_load_ps(p);
                __m256 t1 = _mm256_load_ps(p + 8);
                mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                mVecHi = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return *this;
            }
            // STORE   - Store vector content into memory (either aligned or unaligned)
            // MSTORE  - Masked store vector content into memory (either aligned or
            //           unaligned)
            // STOREA  - Store vector content into aligned memory
            // MSTOREA - Masked store vector content into aligned memory
            // EXTRACT - Extract single element from a vector
            // INSERT  - Insert single element into a vector

            //(Addition operations)
            // ADDV     - Add with vector
            inline SIMDVecAVX2_f add(SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(mVecLo, b.mVecLo);
                __m256 t1 = _mm256_add_ps(mVecHi, b.mVecHi);
                return SIMDVecAVX2_f(t0, t1);
            }
            // MADDV    - Masked add with vector
            inline SIMDVecAVX2_f add(SIMDMask16 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(mVecLo, b.mVecLo);
                __m256 t1 = _mm256_add_ps(mVecHi, b.mVecHi);
                __m256 t2 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                __m256 t3 = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return SIMDVecAVX2_f(t2, t3);
            }
            // ADDS     - Add with scalar
            inline SIMDVecAVX2_f add(float b) {
                __m256 t0 = _mm256_add_ps(mVecLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_add_ps(mVecHi, _mm256_set1_ps(b));
                return SIMDVecAVX2_f(t0, t1);
            }
            // MADDS    - Masked add with scalar
            inline SIMDVecAVX2_f add(SIMDMask16 const & mask, float b) {
                __m256 t0 = _mm256_add_ps(mVecLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_add_ps(mVecHi, _mm256_set1_ps(b));
                __m256 t2 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                __m256 t3 = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return SIMDVecAVX2_f(t2, t3);
            }
            // ADDVA    - Add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDVecAVX2_f const & b) {
                mVecLo = _mm256_add_ps(mVecLo, b.mVecLo);
                mVecHi = _mm256_add_ps(mVecHi, b.mVecHi);
                return *this;
            }
            // MADDVA   - Masked add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDMask16 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(mVecLo, b.mVecLo);
                __m256 t1 = _mm256_add_ps(mVecHi, b.mVecHi);
                mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                mVecHi = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return *this;
            }
            // ADDSA    - Add with scalar and assign
            inline SIMDVecAVX2_f & adda(float b) {
                mVecLo = _mm256_add_ps(mVecLo, _mm256_set1_ps(b));
                mVecHi = _mm256_add_ps(mVecHi, _mm256_set1_ps(b));
                return *this;
            }
            // SADDV    - Saturated add with vector
            inline SIMDVecAVX2_f & adda(SIMDMask16 const & mask, float b) {
                __m256 t0 = _mm256_add_ps(mVecLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_add_ps(mVecHi, _mm256_set1_ps(b));
                mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                mVecHi = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return *this;
            }
            // MSADDV   - Masked saturated add with vector
            // SADDS    - Saturated add with scalar
            // MSADDS   - Masked saturated add with scalar
            // SADDVA   - Saturated add with vector and assign
            // MSADDVA  - Masked saturated add with vector and assign
            // SADDSA   - Satureated add with scalar and assign
            // MSADDSA  - Masked staturated add with vector and assign
            // POSTINC  - Postfix increment
            // MPOSTINC - Masked postfix increment
            // PREFINC  - Prefix increment
            // MPREFINC - Masked prefix increment

            //(Subtraction operations)
            // SUBV       - Sub with vector
            // MSUBV      - Masked sub with vector
            // SUBS       - Sub with scalar
            // MSUBS      - Masked subtraction with scalar
            // SUBVA      - Sub with vector and assign
            // MSUBVA     - Masked sub with vector and assign
            // SUBSA      - Sub with scalar and assign
            // MSUBSA     - Masked sub with scalar and assign
            // SSUBV      - Saturated sub with vector
            // MSSUBV     - Masked saturated sub with vector
            // SSUBS      - Saturated sub with scalar
            // MSSUBS     - Masked saturated sub with scalar
            // SSUBVA     - Saturated sub with vector and assign
            // MSSUBVA    - Masked saturated sub with vector and assign
            // SSUBSA     - Saturated sub with scalar and assign
            // MSSUBSA    - Masked saturated sub with scalar and assign
            // SUBFROMV   - Sub from vector
            // MSUBFROMV  - Masked sub from vector
            // SUBFROMS   - Sub from scalar (promoted to vector)
            // MSUBFROMS  - Masked sub from scalar (promoted to vector)
            // SUBFROMVA  - Sub from vector and assign
            // MSUBFROMVA - Masked sub from vector and assign
            // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
            // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
            // POSTDEC    - Postfix decrement
            // MPOSTDEC   - Masked postfix decrement
            // PREFDEC    - Prefix decrement
            // MPREFDEC   - Masked prefix decrement

            //(Multiplication operations)
            // MULV   - Multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_mul_ps(this->mVecLo, b.mVecLo);
                __m256 t1 = _mm256_mul_ps(this->mVecHi, b.mVecHi);
                return SIMDVecAVX2_f(t0, t1);
            }
            // MMULV  - Masked multiplication with vector
            inline SIMDVecAVX2_f mul(SIMDMask16 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_mul_ps(mVecLo, b.mVecLo);
                __m256 t1 = _mm256_mul_ps(mVecHi, b.mVecHi);
                __m256 t2 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                __m256 t3 = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return SIMDVecAVX2_f(t2, t3);
            }
            // MULS   - Multiplication with scalar
            inline SIMDVecAVX2_f mul(float b) {
                __m256 t0 = _mm256_mul_ps(this->mVecLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_mul_ps(this->mVecHi, _mm256_set1_ps(b));
                return SIMDVecAVX2_f(t0, t1);
            }
            // MMULS  - Masked multiplication with scalar
            inline SIMDVecAVX2_f mul(SIMDMask16 const & mask, float b) {
                __m256 t0 = _mm256_mul_ps(mVecLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_mul_ps(mVecHi, _mm256_set1_ps(b));
                __m256 t2 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
                __m256 t3 = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
                return SIMDVecAVX2_f(t2, t3);
            }
            // MULVA  - Multiplication with vector and assign
            // MMULVA - Masked multiplication with vector and assign
            // MULSA  - Multiplication with scalar and assign
            // MMULSA - Masked multiplication with scalar and assign

            //(Division operations)
            // DIVV   - Division with vector
            // MDIVV  - Masked division with vector
            // DIVS   - Division with scalar
            // MDIVS  - Masked division with scalar
            // DIVVA  - Division with vector and assign
            // MDIVVA - Masked division with vector and assign
            // DIVSA  - Division with scalar and assign
            // MDIVSA - Masked division with scalar and assign
            // RCP    - Reciprocal
            // MRCP   - Masked reciprocal
            // RCPS   - Reciprocal with scalar numerator
            // MRCPS  - Masked reciprocal with scalar
            // RCPA   - Reciprocal and assign
            // MRCPA  - Masked reciprocal and assign
            // RCPSA  - Reciprocal with scalar and assign
            // MRCPSA - Masked reciprocal with scalar and assign

            //(Comparison operations)
            // CMPEQV - Element-wise 'equal' with vector
            // CMPEQS - Element-wise 'equal' with scalar
            // CMPNEV - Element-wise 'not equal' with vector
            // CMPNES - Element-wise 'not equal' with scalar
            // CMPGTV - Element-wise 'greater than' with vector
            // CMPGTS - Element-wise 'greater than' with scalar
            // CMPLTV - Element-wise 'less than' with vector
            // CMPLTS - Element-wise 'less than' with scalar
            // CMPGEV - Element-wise 'greater than or equal' with vector
            // CMPGES - Element-wise 'greater than or equal' with scalar
            // CMPLEV - Element-wise 'less than or equal' with vector
            // CMPLES - Element-wise 'less than or equal' with scalar
            // CMPEX  - Check if vectors are exact (returns scalar 'bool')

            // (Pack/Unpack operations - not available for SIMD1)
            // PACK     - assign vector with two half-length vectors
            // PACKLO   - assign lower half of a vector with a half-length vector
            // PACKHI   - assign upper half of a vector with a half-length vector
            // UNPACK   - Unpack lower and upper halfs to half-length vectors.
            // UNPACKLO - Unpack lower half and return as a half-length vector.
            // UNPACKHI - Unpack upper half and return as a half-length vector.

            //(Blend/Swizzle operations)
            // BLENDV   - Blend (mix) two vectors
            // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
            // assign
            // SWIZZLE  - Swizzle (reorder/permute) vector elements
            // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

            //(Reduction to scalar operations)
            // HADD  - Add elements of a vector (horizontal add)
            // MHADD - Masked add elements of a vector (horizontal add)
            // HMUL  - Multiply elements of a vector (horizontal mul)
            // MHMUL - Masked multiply elements of a vector (horizontal mul)

            //(Fused arithmetics)
            // FMULADDV  - Fused multiply and add (A*B + C) with vectors
            inline SIMDVecAVX2_f fmuladd(SIMDVecAVX2_f const & a, SIMDVecAVX2_f const & b) {
#ifdef FMA
                __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
                __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
                return SIMDVecAVX2_f(t0, t1);
#else
                __m256 t0 = _mm256_add_ps(_mm256_mul_ps(this->mVecLo, a.mVecLo), b.mVecLo);
                __m256 t1 = _mm256_add_ps(_mm256_mul_ps(this->mVecHi, a.mVecHi), b.mVecHi);
#endif
                return SIMDVecAVX2_f(t0, t1);
            }

            // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
            inline SIMDVecAVX2_f fmuladd(SIMDMask16 const & mask, SIMDVecAVX2_f const & a, SIMDVecAVX2_f const & b) {
#ifdef FMA
                __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
                __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
#else
                __m256 t0 = _mm256_add_ps(_mm256_mul_ps(this->mVecLo, a.mVecLo), b.mVecLo);
                __m256 t1 = _mm256_add_ps(_mm256_mul_ps(this->mVecHi, a.mVecHi), b.mVecHi);
#endif
                __m256 t2 = _mm256_blendv_ps(this->mVecLo, t0, _mm256_cvtepi32_ps(mask.mMaskLo));
                __m256 t3 = _mm256_blendv_ps(this->mVecHi, t1, _mm256_cvtepi32_ps(mask.mMaskHi));
                return SIMDVecAVX2_f(t2, t3);
            }
            // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
            // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
            // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
            // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
            // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
            // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

            // (Mathematical operations)
            // MAXV   - Max with vector
            // MMAXV  - Masked max with vector
            // MAXS   - Max with scalar
            // MMAXS  - Masked max with scalar
            // MAXVA  - Max with vector and assign
            // MMAXVA - Masked max with vector and assign
            // MAXSA  - Max with scalar (promoted to vector) and assign
            // MMAXSA - Masked max with scalar (promoted to vector) and assign
            // MINV   - Min with vector
            // MMINV  - Masked min with vector
            // MINS   - Min with scalar (promoted to vector)
            // MMINS  - Masked min with scalar (promoted to vector)
            // MINVA  - Min with vector and assign
            // MMINVA - Masked min with vector and assign
            // MINSA  - Min with scalar (promoted to vector) and assign
            // MMINSA - Masked min with scalar (promoted to vector) and assign
            // HMAX   - Max of elements of a vector (horizontal max)
            // MHMAX  - Masked max of elements of a vector (horizontal max)
            // IMAX   - Index of max element of a vector
            // HMIN   - Min of elements of a vector (horizontal min)
            // MHMIN  - Masked min of elements of a vector (horizontal min)
            // IMIN   - Index of min element of a vector
            // MIMIN  - Masked index of min element of a vector

            // (Gather/Scatter operations)
            // GATHERS   - Gather from memory using indices from array
            // MGATHERS  - Masked gather from memory using indices from array
            // GATHERV   - Gather from memory using indices from vector
            // MGATHERV  - Masked gather from memory using indices from vector
            // SCATTERS  - Scatter to memory using indices from array
            // MSCATTERS - Masked scatter to memory using indices from array
            // SCATTERV  - Scatter to memory using indices from vector
            // MSCATTERV - Masked scatter to memory using indices from vector

            // 3) Operations available for Signed integer and Unsigned integer 
            // data types:

            //(Signed/Unsigned cast)
            // UTOI - Cast unsigned vector to signed vector
            // ITOU - Cast signed vector to unsigned vector

            // 4) Operations available for Signed integer and floating point SIMD types:

            // (Sign modification)
            // NEG   - Negate signed values
            // MNEG  - Masked negate signed values
            // NEGA  - Negate signed values and assign
            // MNEGA - Masked negate signed values and assign

            // (Mathematical functions)
            // ABS   - Absolute value
            // MABS  - Masked absolute value
            // ABSA  - Absolute value and assign
            // MABSA - Masked absolute value and assign

            // 5) Operations available for floating point SIMD types:

            // (Comparison operations)
            // CMPEQRV - Compare 'Equal within range' with margins from vector
            // CMPEQRS - Compare 'Equal within range' with scalar margin

            // (Mathematical functions)
            // SQR       - Square of vector values
            // MSQR      - Masked square of vector values
            // SQRA      - Square of vector values and assign
            // MSQRA     - Masked square of vector values and assign
            // SQRT      - Square root of vector values
            // MSQRT     - Masked square root of vector values 
            // SQRTA     - Square root of vector values and assign
            // MSQRTA    - Masked square root of vector values and assign
            // POWV      - Power (exponents in vector)
            // MPOWV     - Masked power (exponents in vector)
            // POWS      - Power (exponent in scalar)
            // MPOWS     - Masked power (exponent in scalar) 
            // ROUND     - Round to nearest integer
            // MROUND    - Masked round to nearest integer
            // TRUNC     - Truncate to integer (returns Signed integer vector)
            // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
            // FLOOR     - Floor
            // MFLOOR    - Masked floor
            // CEIL      - Ceil
            // MCEIL     - Masked ceil
            // ISFIN     - Is finite
            // ISINF     - Is infinite (INF)
            // ISAN      - Is a number
            // ISNAN     - Is 'Not a Number (NaN)'
            // ISSUB     - Is subnormal
            // ISZERO    - Is zero
            // ISZEROSUB - Is zero or subnormal
            // SIN       - Sine
            // MSIN      - Masked sine
            // COS       - Cosine
            // MCOS      - Masked cosine
            // TAN       - Tangent
            // MTAN      - Masked tangent
            // CTAN      - Cotangent
            // MCTAN     - Masked cotangent

        };

        template<>
        class SIMDVecAVX2_f<float, 32> :
            public SIMDVecFloatInterface<
            SIMDVecAVX2_f<float, 32>,
            SIMDVecAVX2_u<uint32_t, 32>,
            SIMDVecAVX2_i<int32_t, 32>,
            float,
            32,
            uint32_t,
            SIMDMask32,
            SIMDSwizzle32>,
            public SIMDVecPackableInterface<
            SIMDVecAVX2_f<float, 32>,
            SIMDVecAVX2_f<float, 16 >>
        {
        private:
            __m256 mVecLoLo;  // bits 0-255
            __m256 mVecLoHi;  // bits 256-511
            __m256 mVecHiLo;  // bits 512-767
            __m256 mVecHiHi;  // bits 768-1023

            inline SIMDVecAVX2_f(__m256 const & xLoLo, __m256 const & xLoHi, __m256 const & xHiLo, __m256 const & xHiHi) {
                this->mVecLoLo = xLoLo;
                this->mVecLoHi = xLoHi;
                this->mVecHiLo = xHiLo;
                this->mVecHiHi = xHiHi;
            }

            typedef SIMDVecAVX2_u<uint32_t, 32>    VEC_UINT_TYPE;
            typedef SIMDVecAVX2_i<int32_t, 32>     VEC_INT_TYPE;
        public:
            // ZERO-CONSTR - Zero element constructor 
            inline SIMDVecAVX2_f() {}

            // SET-CONSTR  - One element constructor
            inline explicit SIMDVecAVX2_f(float f) {
                mVecLoLo = _mm256_set1_ps(f);
                mVecLoHi = _mm256_set1_ps(f);
                mVecHiLo = _mm256_set1_ps(f);
                mVecHiHi = _mm256_set1_ps(f);
            }

            // UTOF
            inline explicit SIMDVecAVX2_f(VEC_UINT_TYPE const & vecUint) {

            }

            // ITOF
            inline explicit SIMDVecAVX2_f(VEC_INT_TYPE const & vecInt) {

            }

            // LOAD-CONSTR - Construct by loading from memory
            inline explicit SIMDVecAVX2_f(float const *p) { this->load(p); }

            // FULL-CONSTR - constructor with VEC_LEN scalar element 
            inline SIMDVecAVX2_f(float f0, float f1, float f2, float f3,
                float f4, float f5, float f6, float f7,
                float f8, float f9, float f10, float f11,
                float f12, float f13, float f14, float f15,
                float f16, float f17, float f18, float f19,
                float f20, float f21, float f22, float f23,
                float f24, float f25, float f26, float f27,
                float f28, float f29, float f30, float f31) {
                mVecLoLo = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
                mVecLoHi = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
                mVecHiLo = _mm256_setr_ps(f16, f17, f18, f19, f20, f21, f22, f23);
                mVecHiHi = _mm256_setr_ps(f24, f25, f26, f27, f28, f29, f30, f31);
            }

            // EXTRACT - Extract single element from a vector
            inline float extract(uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(32) float raw[8];
                if (index < 8) {
                    _mm256_store_ps(raw, mVecLoLo);
                    return raw[index];
                }
                else if (index < 16) {
                    _mm256_store_ps(raw, mVecLoHi);
                    return raw[index - 8];
                }
                else if (index < 24) {
                    _mm256_store_ps(raw, mVecHiLo);
                    return raw[index - 16];
                }
                else {
                    _mm256_store_ps(raw, mVecHiHi);
                    return raw[index - 24];
                }
            }

            // EXTRACT - Extract single element from a vector
            inline float operator[] (uint32_t index) const {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                return extract(index);
            }

            // Override Mask Access operators
            inline IntermediateMask<SIMDVecAVX2_f, SIMDMask32> operator[] (SIMDMask32 const & mask) {
                return IntermediateMask<SIMDVecAVX2_f, SIMDMask32>(mask, static_cast<SIMDVecAVX2_f &>(*this));
            }

            // INSERT  - Insert single element into a vector
            inline SIMDVecAVX2_f & insert(uint32_t index, float value) {
                UME_PERFORMANCE_UNOPTIMAL_WARNING();
                alignas(32) float raw[8];
                if (index < 8) {
                    _mm256_store_ps(raw, mVecLoLo);
                    raw[index] = value;
                    mVecLoLo = _mm256_load_ps(raw);
                }
                else if (index < 16) {
                    _mm256_store_ps(raw, mVecLoHi);
                    raw[index - 8] = value;
                    mVecLoHi = _mm256_load_ps(raw);
                }
                else if (index < 24) {
                    _mm256_store_ps(raw, mVecHiLo);
                    raw[index - 16] = value;
                    mVecHiLo = _mm256_load_ps(raw);
                }
                else {
                    _mm256_store_ps(raw, mVecHiHi);
                    raw[index - 24] = value;
                    mVecHiHi = _mm256_load_ps(raw);
                }

                return *this;
            }

            //(Initialization)
            // ASSIGNV     - Assignment with another vector
            // MASSIGNV    - Masked assignment with another vector
            // ASSIGNS     - Assignment with scalar
            // MASSIGNS    - Masked assign with scalar

            //(Memory access)
            // LOAD    - Load from memory (either aligned or unaligned) to vector 
            inline SIMDVecAVX2_f & load(float const * p) {
                mVecLoLo = _mm256_loadu_ps(p);
                mVecLoHi = _mm256_loadu_ps(p + 8);
                mVecHiLo = _mm256_loadu_ps(p + 16);
                mVecHiHi = _mm256_loadu_ps(p + 24);
                return *this;
            }
            // MLOAD   - Masked load from memory (either aligned or unaligned) to
            //        vector
            inline SIMDVecAVX2_f & load(SIMDMask32 const & mask, float const * p) {
                __m256 t0 = _mm256_loadu_ps(p);
                __m256 t1 = _mm256_loadu_ps(p + 8);
                __m256 t2 = _mm256_loadu_ps(p + 16);
                __m256 t3 = _mm256_loadu_ps(p + 24);
                mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
                mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
                mVecHiLo = _mm256_blendv_ps(mVecHiLo, t1, _mm256_castsi256_ps(mask.mMaskHiLo));
                mVecHiHi = _mm256_blendv_ps(mVecHiHi, t1, _mm256_castsi256_ps(mask.mMaskHiHi));
                return *this;
            }
            // LOADA   - Load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(float const * p) {
                mVecLoLo = _mm256_load_ps(p);
                mVecLoHi = _mm256_load_ps(p + 8);
                mVecHiLo = _mm256_load_ps(p + 16);
                mVecHiHi = _mm256_load_ps(p + 24);
                return *this;
            }

            // MLOADA  - Masked load from aligned memory to vector
            inline SIMDVecAVX2_f & loada(SIMDMask32 const & mask, float const * p) {
                __m256 t0 = _mm256_load_ps(p);
                __m256 t1 = _mm256_load_ps(p + 8);
                __m256 t2 = _mm256_load_ps(p + 16);
                __m256 t3 = _mm256_load_ps(p + 24);
                mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
                mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
                mVecHiLo = _mm256_blendv_ps(mVecHiLo, t1, _mm256_castsi256_ps(mask.mMaskHiLo));
                mVecHiHi = _mm256_blendv_ps(mVecHiHi, t1, _mm256_castsi256_ps(mask.mMaskHiHi));
                return *this;
            }
            // STORE   - Store vector content into memory (either aligned or unaligned)
            // MSTORE  - Masked store vector content into memory (either aligned or
            //           unaligned)
            // STOREA  - Store vector content into aligned memory
            // MSTOREA - Masked store vector content into aligned memory
            // EXTRACT - Extract single element from a vector
            // INSERT  - Insert single element into a vector

            //(Addition operations)
            // ADDV     - Add with vector
            inline SIMDVecAVX2_f add(SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
                __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
                __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
                __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
                return SIMDVecAVX2_f(t0, t1, t2, t3);
            }
            // MADDV    - Masked add with vector
            inline SIMDVecAVX2_f add(SIMDMask32 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
                __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
                __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
                __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
                __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
                __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
                __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
                __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
                return SIMDVecAVX2_f(t4, t5, t6, t7);
            }
            // ADDS     - Add with scalar
            inline SIMDVecAVX2_f add(float b) {
                __m256 t0 = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
                __m256 t2 = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
                __m256 t3 = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
                return SIMDVecAVX2_f(t0, t1, t2, t3);
            }
            // MADDS    - Masked add with scalar
            inline SIMDVecAVX2_f add(SIMDMask32 const & mask, float b) {
                __m256 t0 = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
                __m256 t2 = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
                __m256 t3 = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
                __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
                __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
                __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
                __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
                return SIMDVecAVX2_f(t4, t5, t6, t7);
            }
            // ADDVA    - Add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDVecAVX2_f const & b) {
                mVecLoLo = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
                mVecLoHi = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
                mVecHiLo = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
                mVecHiHi = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
                return *this;
            }
            // MADDVA   - Masked add with vector and assign
            inline SIMDVecAVX2_f & adda(SIMDMask32 const & mask, SIMDVecAVX2_f const & b) {
                __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
                __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
                __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
                __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
                mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
                mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
                mVecHiLo = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
                mVecHiHi = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
                return *this;
            }
            // ADDSA    - Add with scalar and assign
            inline SIMDVecAVX2_f & adda(float b) {
                mVecLoLo = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
                mVecLoHi = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
                mVecHiLo = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
                mVecHiHi = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
                return *this;
            }
            // MADDSA   - Masked add with scalar and assign
            inline SIMDVecAVX2_f & adda(SIMDMask32 const & mask, float b) {
                __m256 t0 = _mm256_add_ps(mVecLoLo, _mm256_set1_ps(b));
                __m256 t1 = _mm256_add_ps(mVecLoHi, _mm256_set1_ps(b));
                __m256 t2 = _mm256_add_ps(mVecHiLo, _mm256_set1_ps(b));
                __m256 t3 = _mm256_add_ps(mVecHiHi, _mm256_set1_ps(b));
                mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_castsi256_ps(mask.mMaskLoLo));
                mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_castsi256_ps(mask.mMaskLoHi));
                mVecHiLo = _mm256_blendv_ps(mVecHiLo, t2, _mm256_castsi256_ps(mask.mMaskHiLo));
                mVecHiHi = _mm256_blendv_ps(mVecHiHi, t3, _mm256_castsi256_ps(mask.mMaskHiHi));
                return *this;
            }
            // SADDV    - Saturated add with vector
            // MSADDV   - Masked saturated add with vector
            // SADDS    - Saturated add with scalar
            // MSADDS   - Masked saturated add with scalar
            // SADDVA   - Saturated add with vector and assign
            // MSADDVA  - Masked saturated add with vector and assign
            // SADDSA   - Satureated add with scalar and assign
            // MSADDSA  - Masked staturated add with vector and assign
            // POSTINC  - Postfix increment
            // MPOSTINC - Masked postfix increment
            // PREFINC  - Prefix increment
            // MPREFINC - Masked prefix increment

            //(Subtraction operations)
            // SUBV       - Sub with vector
            // MSUBV      - Masked sub with vector
            // SUBS       - Sub with scalar
            // MSUBS      - Masked subtraction with scalar
            // SUBVA      - Sub with vector and assign
            // MSUBVA     - Masked sub with vector and assign
            // SUBSA      - Sub with scalar and assign
            // MSUBSA     - Masked sub with scalar and assign
            // SSUBV      - Saturated sub with vector
            // MSSUBV     - Masked saturated sub with vector
            // SSUBS      - Saturated sub with scalar
            // MSSUBS     - Masked saturated sub with scalar
            // SSUBVA     - Saturated sub with vector and assign
            // MSSUBVA    - Masked saturated sub with vector and assign
            // SSUBSA     - Saturated sub with scalar and assign
            // MSSUBSA    - Masked saturated sub with scalar and assign
            // SUBFROMV   - Sub from vector
            // MSUBFROMV  - Masked sub from vector
            // SUBFROMS   - Sub from scalar (promoted to vector)
            // MSUBFROMS  - Masked sub from scalar (promoted to vector)
            // SUBFROMVA  - Sub from vector and assign
            // MSUBFROMVA - Masked sub from vector and assign
            // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
            // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
            // POSTDEC    - Postfix decrement
            // MPOSTDEC   - Masked postfix decrement
            // PREFDEC    - Prefix decrement
            // MPREFDEC   - Masked prefix decrement

            //(Multiplication operations)
            // MULV   - Multiplication with vector
            // MMULV  - Masked multiplication with vector
            // MULS   - Multiplication with scalar
            // MMULS  - Masked multiplication with scalar
            // MULVA  - Multiplication with vector and assign
            // MMULVA - Masked multiplication with vector and assign
            // MULSA  - Multiplication with scalar and assign
            // MMULSA - Masked multiplication with scalar and assign

            //(Division operations)
            // DIVV   - Division with vector
            // MDIVV  - Masked division with vector
            // DIVS   - Division with scalar
            // MDIVS  - Masked division with scalar
            // DIVVA  - Division with vector and assign
            // MDIVVA - Masked division with vector and assign
            // DIVSA  - Division with scalar and assign
            // MDIVSA - Masked division with scalar and assign
            // RCP    - Reciprocal
            // MRCP   - Masked reciprocal
            // RCPS   - Reciprocal with scalar numerator
            // MRCPS  - Masked reciprocal with scalar
            // RCPA   - Reciprocal and assign
            // MRCPA  - Masked reciprocal and assign
            // RCPSA  - Reciprocal with scalar and assign
            // MRCPSA - Masked reciprocal with scalar and assign

            //(Comparison operations)
            // CMPEQV - Element-wise 'equal' with vector
            // CMPEQS - Element-wise 'equal' with scalar
            // CMPNEV - Element-wise 'not equal' with vector
            // CMPNES - Element-wise 'not equal' with scalar
            // CMPGTV - Element-wise 'greater than' with vector
            // CMPGTS - Element-wise 'greater than' with scalar
            // CMPLTV - Element-wise 'less than' with vector
            // CMPLTS - Element-wise 'less than' with scalar
            // CMPGEV - Element-wise 'greater than or equal' with vector
            // CMPGES - Element-wise 'greater than or equal' with scalar
            // CMPLEV - Element-wise 'less than or equal' with vector
            // CMPLES - Element-wise 'less than or equal' with scalar
            // CMPEX  - Check if vectors are exact (returns scalar 'bool')

            // (Pack/Unpack operations - not available for SIMD1)
            // PACK     - assign vector with two half-length vectors
            // PACKLO   - assign lower half of a vector with a half-length vector
            // PACKHI   - assign upper half of a vector with a half-length vector
            // UNPACK   - Unpack lower and upper halfs to half-length vectors.
            // UNPACKLO - Unpack lower half and return as a half-length vector.
            // UNPACKHI - Unpack upper half and return as a half-length vector.

            //(Blend/Swizzle operations)
            // BLENDV   - Blend (mix) two vectors
            // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
            // assign
            // SWIZZLE  - Swizzle (reorder/permute) vector elements
            // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

            //(Reduction to scalar operations)
            // HADD  - Add elements of a vector (horizontal add)
            // MHADD - Masked add elements of a vector (horizontal add)
            // HMUL  - Multiply elements of a vector (horizontal mul)
            // MHMUL - Masked multiply elements of a vector (horizontal mul)

            //(Fused arithmetics)
            // FMULADDV  - Fused multiply and add (A*B + C) with vectors
            // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
            // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
            // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
            // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
            // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
            // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
            // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors

            // (Mathematical operations)
            // MAXV   - Max with vector
            // MMAXV  - Masked max with vector
            // MAXS   - Max with scalar
            // MMAXS  - Masked max with scalar
            // MAXVA  - Max with vector and assign
            // MMAXVA - Masked max with vector and assign
            // MAXSA  - Max with scalar (promoted to vector) and assign
            // MMAXSA - Masked max with scalar (promoted to vector) and assign
            // MINV   - Min with vector
            // MMINV  - Masked min with vector
            // MINS   - Min with scalar (promoted to vector)
            // MMINS  - Masked min with scalar (promoted to vector)
            // MINVA  - Min with vector and assign
            // MMINVA - Masked min with vector and assign
            // MINSA  - Min with scalar (promoted to vector) and assign
            // MMINSA - Masked min with scalar (promoted to vector) and assign
            // HMAX   - Max of elements of a vector (horizontal max)
            // MHMAX  - Masked max of elements of a vector (horizontal max)
            // IMAX   - Index of max element of a vector
            // HMIN   - Min of elements of a vector (horizontal min)
            // MHMIN  - Masked min of elements of a vector (horizontal min)
            // IMIN   - Index of min element of a vector
            // MIMIN  - Masked index of min element of a vector

            // (Gather/Scatter operations)
            // GATHERS   - Gather from memory using indices from array
            // MGATHERS  - Masked gather from memory using indices from array
            // GATHERV   - Gather from memory using indices from vector
            // MGATHERV  - Masked gather from memory using indices from vector
            // SCATTERS  - Scatter to memory using indices from array
            // MSCATTERS - Masked scatter to memory using indices from array
            // SCATTERV  - Scatter to memory using indices from vector
            // MSCATTERV - Masked scatter to memory using indices from vector

            // 3) Operations available for Signed integer and Unsigned integer 
            // data types:

            //(Signed/Unsigned cast)
            // UTOI - Cast unsigned vector to signed vector
            // ITOU - Cast signed vector to unsigned vector

            // 4) Operations available for Signed integer and floating point SIMD types:

            // (Sign modification)
            // NEG   - Negate signed values
            // MNEG  - Masked negate signed values
            // NEGA  - Negate signed values and assign
            // MNEGA - Masked negate signed values and assign

            // (Mathematical functions)
            // ABS   - Absolute value
            // MABS  - Masked absolute value
            // ABSA  - Absolute value and assign
            // MABSA - Masked absolute value and assign

            // 5) Operations available for floating point SIMD types:

            // (Comparison operations)
            // CMPEQRV - Compare 'Equal within range' with margins from vector
            // CMPEQRS - Compare 'Equal within range' with scalar margin

            // (Mathematical functions)
            // SQR       - Square of vector values
            // MSQR      - Masked square of vector values
            // SQRA      - Square of vector values and assign
            // MSQRA     - Masked square of vector values and assign
            // SQRT      - Square root of vector values
            // MSQRT     - Masked square root of vector values 
            // SQRTA     - Square root of vector values and assign
            // MSQRTA    - Masked square root of vector values and assign
            // POWV      - Power (exponents in vector)
            // MPOWV     - Masked power (exponents in vector)
            // POWS      - Power (exponent in scalar)
            // MPOWS     - Masked power (exponent in scalar) 
            // ROUND     - Round to nearest integer
            // MROUND    - Masked round to nearest integer
            // TRUNC     - Truncate to integer (returns Signed integer vector)
            // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
            // FLOOR     - Floor
            // MFLOOR    - Masked floor
            // CEIL      - Ceil
            // MCEIL     - Masked ceil
            // ISFIN     - Is finite
            // ISINF     - Is infinite (INF)
            // ISAN      - Is a number
            // ISNAN     - Is 'Not a Number (NaN)'
            // ISSUB     - Is subnormal
            // ISZERO    - Is zero
            // ISZEROSUB - Is zero or subnormal
            // SIN       - Sine
            // MSIN      - Masked sine
            // COS       - Cosine
            // MCOS      - Masked cosine
            // TAN       - Tangent
            // MTAN      - Masked tangent
            // CTAN      - Cotangent
            // MCTAN     - Masked cotangent
        };
    }
}
#endif
