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

#ifndef UME_SIMD_VEC_FLOAT_KNC_H_
#define UME_SIMD_VEC_FLOAT_KNC_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

#include "UMESimdMaskKNC.h"
#include "UMESimdSwizzleKNC.h"
#include "UMESimdVecUintKNC.h"
#include "UMESimdVecFloatKNC.h"

namespace UME {
namespace SIMD {

    // ********************************************************************************************
    // FLOATING POINT VECTORS
    // ********************************************************************************************

    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    struct SIMDVecKNC_f_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 32b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 1> {
        typedef SIMDVecKNC_u<uint32_t, 1>  VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 1>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 2> {
        typedef SIMDVecKNC_f<float, 1>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 2>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 1> {
        typedef SIMDVecKNC_u<uint64_t, 1> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 1>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask1                 MASK_TYPE;
        typedef SIMDSwizzle1              SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 4> {
        typedef SIMDVecKNC_f<float, 2>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 4> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 4>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 2> {
        typedef SIMDVecKNC_f<double, 1>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 2> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 2>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask2                 MASK_TYPE;
        typedef SIMDSwizzle2              SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 8> {
        typedef SIMDVecKNC_f<float, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 8> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 8>  VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 4> {
        typedef SIMDVecKNC_f<double, 2>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 4> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 4>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask4                 MASK_TYPE;
        typedef SIMDSwizzle4              SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 16> {
        typedef SIMDVecKNC_f<float, 8>     HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 16> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 16>  VEC_INT_TYPE;
        typedef int32_t                    SCALAR_INT_TYPE;
        typedef uint32_t                   SCALAR_UINT_TYPE;
        typedef float*                     SCALAR_TYPE_PTR;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 8> {
        typedef SIMDVecKNC_f<double, 4>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 8> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 8>  VEC_INT_TYPE;
        typedef int64_t                   SCALAR_INT_TYPE;
        typedef uint64_t                  SCALAR_UINT_TYPE;
        typedef double*                   SCALAR_TYPE_PTR;
        typedef SIMDMask8                 MASK_TYPE;
        typedef SIMDSwizzle8              SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVecKNC_f_traits<float, 32> {
        typedef SIMDVecKNC_f<float, 16>   HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint32_t, 32> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int32_t, 32> VEC_INT_TYPE;
        typedef int32_t                   SCALAR_INT_TYPE;
        typedef uint32_t                  SCALAR_UINT_TYPE;
        typedef float*                    SCALAR_TYPE_PTR;
        typedef SIMDMask32                MASK_TYPE;
        typedef SIMDSwizzle32             SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVecKNC_f_traits<double, 16> {
        typedef SIMDVecKNC_f<double, 8>    HALF_LEN_VEC_TYPE;
        typedef SIMDVecKNC_u<uint64_t, 16> VEC_UINT_TYPE;
        typedef SIMDVecKNC_i<int64_t, 16>  VEC_INT_TYPE;
        typedef int64_t                    SCALAR_INT_TYPE;
        typedef uint64_t                   SCALAR_UINT_TYPE;
        typedef double*                    SCALAR_TYPE_PTR;
        typedef SIMDMask16                 MASK_TYPE;
        typedef SIMDSwizzle16              SWIZZLE_MASK_TYPE;
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
    class SIMDVecKNC_f final :
        public SIMDVecFloatInterface<
        SIMDVecKNC_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
        SCALAR_FLOAT_TYPE,
        VEC_LEN,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
        SIMDVecKNC_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, VEC_LEN>                            VEC_EMU_REG;

        typedef SIMDVecKNC_f VEC_TYPE;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE    VEC_UINT_TYPE;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE     VEC_INT_TYPE;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE       MASK_TYPE;
    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecKNC_f() : mVec() {};

        inline explicit SIMDVecKNC_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};

        // UTOF
        inline explicit SIMDVecKNC_f(VEC_UINT_TYPE const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVecKNC_f(VEC_INT_TYPE const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); }

        inline SIMDVecKNC_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec.insert(0, f0); mVec.insert(1, f1);
        }

        inline SIMDVecKNC_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3) {
            mVec.insert(0, f0);  mVec.insert(1, f1);  mVec.insert(2, f2);  mVec.insert(3, f3);
        }

        inline SIMDVecKNC_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3,
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5,
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7)
        {
            mVec.insert(0, f0);  mVec.insert(1, f1);
            mVec.insert(2, f2);  mVec.insert(3, f3);
            mVec.insert(4, f4);  mVec.insert(5, f5);
            mVec.insert(6, f6);  mVec.insert(7, f7);
        }

        inline SIMDVecKNC_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3,
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5,
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7,
            SCALAR_FLOAT_TYPE f8, SCALAR_FLOAT_TYPE f9,
            SCALAR_FLOAT_TYPE f10, SCALAR_FLOAT_TYPE f11,
            SCALAR_FLOAT_TYPE f12, SCALAR_FLOAT_TYPE f13,
            SCALAR_FLOAT_TYPE f14, SCALAR_FLOAT_TYPE f15)
        {
            mVec.insert(0, f0);    mVec.insert(1, f1);
            mVec.insert(2, f2);    mVec.insert(3, f3);
            mVec.insert(4, f4);    mVec.insert(5, f5);
            mVec.insert(6, f6);    mVec.insert(7, f7);
            mVec.insert(8, f8);    mVec.insert(9, f9);
            mVec.insert(10, f10);  mVec.insert(11, f11);
            mVec.insert(12, f12);  mVec.insert(13, f13);
            mVec.insert(14, f14);  mVec.insert(15, f15);
        }

        inline SIMDVecKNC_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3,
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5,
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7,
            SCALAR_FLOAT_TYPE f8, SCALAR_FLOAT_TYPE f9,
            SCALAR_FLOAT_TYPE f10, SCALAR_FLOAT_TYPE f11,
            SCALAR_FLOAT_TYPE f12, SCALAR_FLOAT_TYPE f13,
            SCALAR_FLOAT_TYPE f14, SCALAR_FLOAT_TYPE f15,
            SCALAR_FLOAT_TYPE f16, SCALAR_FLOAT_TYPE f17,
            SCALAR_FLOAT_TYPE f18, SCALAR_FLOAT_TYPE f19,
            SCALAR_FLOAT_TYPE f20, SCALAR_FLOAT_TYPE f21,
            SCALAR_FLOAT_TYPE f22, SCALAR_FLOAT_TYPE f23,
            SCALAR_FLOAT_TYPE f24, SCALAR_FLOAT_TYPE f25,
            SCALAR_FLOAT_TYPE f26, SCALAR_FLOAT_TYPE f27,
            SCALAR_FLOAT_TYPE f28, SCALAR_FLOAT_TYPE f29,
            SCALAR_FLOAT_TYPE f30, SCALAR_FLOAT_TYPE f31)
        {
            mVec.insert(0, f0);    mVec.insert(1, f1);
            mVec.insert(2, f2);    mVec.insert(3, f3);
            mVec.insert(4, f4);    mVec.insert(5, f5);
            mVec.insert(6, f6);    mVec.insert(7, f7);
            mVec.insert(8, f8);    mVec.insert(9, f9);
            mVec.insert(10, f10);  mVec.insert(11, f11);
            mVec.insert(12, f12);  mVec.insert(13, f13);
            mVec.insert(14, f14);  mVec.insert(15, f15);
            mVec.insert(16, f16);  mVec.insert(17, f17);
            mVec.insert(18, f18);  mVec.insert(19, f19);
            mVec.insert(20, f20);  mVec.insert(21, f21);
            mVec.insert(22, f22);  mVec.insert(23, f23);
            mVec.insert(24, f24);  mVec.insert(25, f25);
            mVec.insert(26, f26);  mVec.insert(27, f27);
            mVec.insert(28, f28);  mVec.insert(29, f29);
            mVec.insert(30, f30);  mVec.insert(31, f31);
        }

        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_f, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVecKNC_f, MASK_TYPE>(mask, static_cast<SIMDVecKNC_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecKNC_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
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
    class SIMDVecKNC_f<SCALAR_FLOAT_TYPE, 1> final :
        public SIMDVecFloatInterface<
        SIMDVecKNC_f<SCALAR_FLOAT_TYPE, 1>,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE,
        SCALAR_FLOAT_TYPE,
        1,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE,
        typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, 1>                            VEC_EMU_REG;

        typedef SIMDVecKNC_f VEC_TYPE;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE    VEC_UINT_TYPE;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE     VEC_INT_TYPE;
        typedef typename SIMDVecKNC_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE       MASK_TYPE;

    private:
        VEC_EMU_REG mVec;

    public:
        inline SIMDVecKNC_f() : mVec() {};

        inline explicit SIMDVecKNC_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};

        // UTOF
        inline explicit SIMDVecKNC_f(VEC_UINT_TYPE const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVecKNC_f(VEC_INT_TYPE const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); }

        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_f, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVecKNC_f, MASK_TYPE>(mask, static_cast<SIMDVecKNC_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVecKNC_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }
    };

    // ********************************************************************************************
    // FLOATING POINT VECTOR specializations
    // ********************************************************************************************

    template<>
    class SIMDVecKNC_f<float, 8> :
        public SIMDVecFloatInterface<
        SIMDVecKNC_f<float, 8>,
        SIMDVecKNC_u<uint32_t, 8>,
        SIMDVecKNC_i<int32_t, 8>,
        float,
        8,
        uint32_t,
        SIMDMask8,
        SIMDSwizzle8>,
        public SIMDVecPackableInterface<
        SIMDVecKNC_f<float, 8>,
        SIMDVecKNC_f<float, 4 >>
    {
    public:
        typedef typename SIMDVecKNC_f_traits<float, 8>::VEC_UINT_TYPE    VEC_UINT_TYPE;
        typedef typename SIMDVecKNC_f_traits<float, 8>::VEC_INT_TYPE     VEC_INT_TYPE;
        typedef typename SIMDVecKNC_f_traits<float, 8>::MASK_TYPE        MASK_TYPE;

    private:
        __m512 mVec;

        inline SIMDVecKNC_f(__m512 & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecKNC_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecKNC_f(float f) {
            mVec = _mm512_set1_ps(f);
        }

        // UTOF
        inline explicit SIMDVecKNC_f(VEC_UINT_TYPE const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVecKNC_f(VEC_INT_TYPE const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_f(float const * p) { this->load(p); }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecKNC_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7) {
            mVec = _mm512_setr_ps(f0, f1, f2, f3,
                f4, f5, f6, f7,
                0.0f, 0.0f, 0.0f, 0.0f,
                0.0f, 0.0f, 0.0f, 0.0f);
        }

        // EXTRACT - Extract single element from a vector
        inline float extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return raw[index];
        }

        // EXTRACT - Extract single element from a vector
        inline float operator[] (uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_f, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVecKNC_f, MASK_TYPE>(mask, static_cast<SIMDVecKNC_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecKNC_f & insert(uint32_t index, float value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_ps(raw);
            return *this;
        }

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
        inline SIMDVecKNC_f & load(float const * p) {
            if ((uint64_t(p) % 64) == 0) {

                mVec = _mm512_mask_load_ps(_mm512_setzero_ps(),
                    0x00FF,
                    p);
            }
            else {
                alignas(64) float raw[8];
                memcpy(raw, p, 8 * sizeof(float));
                mVec = _mm512_mask_load_ps(_mm512_setzero_ps(),
                    0x00FF,
                    raw);
            }
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        inline SIMDVecKNC_f & load(SIMDMask8 const & mask, float const * p) {
            if ((uint64_t(p) % 64) == 0) {
                mVec = _mm512_mask_load_ps(mVec, mask.mMask, p);
            }
            else {
                alignas(64) float raw[8];
                memcpy(raw, p, 8 * sizeof(float));
                mVec = _mm512_mask_load_ps(mVec,
                    mask.mMask,
                    raw);
            }
            return *this;
        }
        // LOADA   - Load from aligned memory to vector
        // For this class alignment is 32B!!!
        inline SIMDVecKNC_f & loada(float const * p) {
            mVec = _mm512_mask_load_ps(mVec,
                0x00FF,
                p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecKNC_f & loada(SIMDMask8 const & mask, float const * p) {
            mVec = _mm512_mask_load_ps(mVec, mask.mMask, p);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float * store(float * p)
        {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_mask_store_ps(p,
                    0x00FF, // Only store 8 lower elements!
                    mVec);
            }
            else {
                alignas(64) float raw[8];
                _mm512_mask_store_ps(raw,
                    0x00FF, // Only store 8 lower elements!
                    mVec);

                memcpy(p, raw, 8 * sizeof(float));
                return p;
            }
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        inline float * store(SIMDMask8 const & mask, float *p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_mask_store_ps(p, mask.mMask, mVec);
            }
            else {
                alignas(64) float raw[8];
                _mm512_mask_store_ps(p, mask.mMask, mVec);
            }
            return p;
        }

        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) {
            _mm512_store_ps(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDMask8 const & mask, float* p) {
            _mm512_mask_store_ps(p, mask.mMask, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVecKNC_f add(SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MADDV    - Masked add with vector
        inline SIMDVecKNC_f add(SIMDMask8 const & mask, SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // ADDS     - Add with scalar 
        inline SIMDVecKNC_f add(float b) const {
            __m512 t0 = _mm512_add_ps(mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecKNC_f add(SIMDMask8 const & mask, float b) const {
            __m512 t0 = _mm512_mask_add_ps(mVec,
                mask.mMask,
                mVec,
                _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecKNC_f & adda(SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_add_ps(mVec,
                0x00FF,
                mVec,
                b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVecKNC_f & adda(SIMDMask8 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }

        // ADDSA    - Add with scalar and assign
        inline SIMDVecKNC_f & adda(float b) {
            mVec = _mm512_mask_add_ps(mVec,
                0x00FF,
                mVec,
                _mm512_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecKNC_f & adda(SIMDMask8 const & mask, float b) {
            mVec = _mm512_mask_add_ps(mVec,
                mask.mMask,
                mVec,
                _mm512_set1_ps(b));
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
        inline SIMDVecKNC_f postinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_mask_add_ps(mVec, 0xFF, mVec, t0);
            return SIMDVecKNC_f(t1);
        }

        inline SIMDVecKNC_f operator++ (int) {
            return postinc();
        }

        // MPOSTINC - Masked postfix increment
        inline SIMDVecKNC_f postinc(SIMDMask8 const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }
        // PREFINC  - Prefix increment
        inline SIMDVecKNC_f & prefinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_add_ps(mVec, 0xFF, mVec, t0);
            return *this;
        }

        inline SIMDVecKNC_f & operator++ () {
            return prefinc();
        }

        // MPREFINC - Masked prefix increment
        inline SIMDVecKNC_f & prefinc(SIMDMask8 const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        //(Subtraction operations)
        // SUBV       - Sub with vector
        inline SIMDVecKNC_f sub(SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, 0xFF, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MSUBV      - Masked sub with vector
        inline SIMDVecKNC_f sub(SIMDMask8 const & mask, SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // SUBS       - Sub with scalar
        inline SIMDVecKNC_f sub(float b) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, 0xFF, mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MSUBS      - Masked subtraction with scalar
        inline SIMDVecKNC_f sub(SIMDMask8 const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }
        // SUBVA      - Sub with vector and assign
        inline SIMDVecKNC_f & suba(SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_sub_ps(mVec, 0xFF, mVec, b.mVec);
            return *this;
        }
        // MSUBVA     - Masked sub with vector and assign
        inline SIMDVecKNC_f & suba(SIMDMask8 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA      - Sub with scalar and assign
        inline SIMDVecKNC_f & suba(float b) {
            mVec = _mm512_mask_sub_ps(mVec, 0xFF, mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MSUBSA     - Masked sub with scalar and assign
        inline SIMDVecKNC_f & suba(SIMDMask8 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
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
        inline SIMDVecKNC_f mul(SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mask_mul_ps(mVec, 0x00FF, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecKNC_f mul(SIMDMask8 const & mask, SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVecKNC_f mul(float b) {
            __m512 t0 = _mm512_mask_mul_ps(mVec,
                0x00FF,
                mVec,
                _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecKNC_f mul(SIMDMask8 const & mask, float b) {
            __m512 t0 = _mm512_mask_mul_ps(mVec,
                mask.mMask,
                mVec,
                _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
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
        inline SIMDVecKNC_f fmuladd(SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) {
            __m512 t0 = _mm512_mask_fmadd_ps(mVec, 0x00FF, b.mVec, c.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecKNC_f fmuladd(SIMDMask8 const & mask, SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) {
            __m512 t0 = _mm512_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVecKNC_f(t0);
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
    class SIMDVecKNC_f<float, 16> :
        public SIMDVecFloatInterface<
        SIMDVecKNC_f<float, 16>,
        SIMDVecKNC_u<uint32_t, 16>,
        SIMDVecKNC_i<int32_t, 16>,
        float,
        16,
        uint32_t,
        SIMDMask16,
        SIMDSwizzle16>,
        public SIMDVecPackableInterface<
        SIMDVecKNC_f<float, 16>,
        SIMDVecKNC_f<float, 8 >>
    {
    public:
        typedef typename SIMDVecKNC_f_traits<float, 16>::VEC_UINT_TYPE  VEC_UINT_TYPE;
        typedef typename SIMDVecKNC_f_traits<float, 16>::VEC_INT_TYPE   VEC_INT_TYPE;
        typedef typename SIMDVecKNC_f_traits<float, 16>::MASK_TYPE      MASK_TYPE;

    private:
        __m512 mVec;

        inline SIMDVecKNC_f(__m512 x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecKNC_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecKNC_f(float f) {
            mVec = _mm512_set1_ps(f);
        }

        // UTOF
        inline explicit SIMDVecKNC_f(VEC_UINT_TYPE const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVecKNC_f(VEC_INT_TYPE const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_f(float const * p) { this->load(p); }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecKNC_f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15) {
            mVec = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
        }

        // EXTRACT - Extract single element from a vector
        inline float extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return raw[index];
        }

        // EXTRACT - Extract single element from a vector
        inline float operator[] (uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_f, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVecKNC_f, MASK_TYPE>(mask, static_cast<SIMDVecKNC_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecKNC_f & insert(uint32_t index, float value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_ps(raw);
            return *this;
        }

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        inline SIMDVecKNC_f & assign(SIMDVecKNC_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV    - Masked assignment with another vector
        inline SIMDVecKNC_f & assign(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS     - Assignment with scalar
        inline SIMDVecKNC_f & assign(float b) {
            mVec = _mm512_set1_ps(b);
            return *this;
        }
        // MASSIGNS    - Masked assign with scalar
        inline SIMDVecKNC_f & assign(SIMDMask16 const & mask, float b) {
            mVec = _mm512_mask_mov_ps(mVec, mask.mMask, _mm512_set1_ps(b));
            return *this;
        }

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVecKNC_f & load(float const * p) {
            if ((uint64_t(p) % 64) == 0) {

                mVec = _mm512_load_ps(p);
            }
            else {
                alignas(64) float raw[16];
                memcpy(raw, p, 16 * sizeof(float));
                mVec = _mm512_load_ps(raw);
            }
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        inline SIMDVecKNC_f & load(SIMDMask16 const & mask, float const * p) {
            if ((uint64_t(p) % 64) == 0) {
                mVec = _mm512_mask_load_ps(mVec, mask.mMask, p);
            }
            else {
                alignas(64) float raw[16];
                memcpy(raw, p, 16 * sizeof(float));
                mVec = _mm512_mask_load_ps(mVec,
                    mask.mMask,
                    raw);
            }
            return *this;
        }
        // LOADA   - Load from aligned memory to vector
        // For this class alignment is 32B!!!
        inline SIMDVecKNC_f & loada(float const * p) {
            mVec = _mm512_load_ps(p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVecKNC_f & loada(SIMDMask16 const & mask, float const * p) {
            mVec = _mm512_mask_load_ps(mVec, mask.mMask, p);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float * store(float * p)
        {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_ps(p, mVec);
            }
            else {
                alignas(64) float raw[16];
                _mm512_store_ps(raw, mVec);
                memcpy(p, raw, 16 * sizeof(float));
                return p;
            }
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        inline float * store(SIMDMask16 const & mask, float *p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_mask_store_ps(p, mask.mMask, mVec);
            }
            else {
                alignas(64) float raw[8];
                _mm512_mask_store_ps(p, mask.mMask, mVec);
            }
            return p;
        }

        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) {
            _mm512_store_ps(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDMask16 const & mask, float* p) {
            _mm512_mask_store_ps(p, mask.mMask, mVec);
            return p;
        }

        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVecKNC_f add(SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        inline SIMDVecKNC_f operator+ (SIMDVecKNC_f const & b) {
            return this->add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVecKNC_f add(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // ADDS     - Add with scalar
        inline SIMDVecKNC_f add(float b) {
            __m512 t0 = _mm512_add_ps(mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        inline SIMDVecKNC_f operator+ (float b) {
            return this->add(b);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecKNC_f add(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecKNC_f & adda(SIMDVecKNC_f const & b) {
            mVec = _mm512_add_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_f & operator+= (SIMDVecKNC_f const & b) {
            return this->adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVecKNC_f & adda(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVecKNC_f & adda(float b) {
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        inline SIMDVecKNC_f & operator+= (float b) {
            return this->adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecKNC_f & adda(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
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
        inline SIMDVecKNC_f postinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_add_ps(t0, t1);
            return SIMDVecKNC_f(t1);
        }
        inline SIMDVecKNC_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC - Masked postfix increment
        inline SIMDVecKNC_f postinc(SIMDMask16 const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }
        // PREFINC  - Prefix increment
        inline SIMDVecKNC_f & prefinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_add_ps(mVec, t0);
            return *this;
        }
        inline SIMDVecKNC_f & operator++ () {
            return prefinc();
        }
        // MPREFINC - Masked prefix increment
        inline SIMDVecKNC_f & prefinc(SIMDMask16 const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        //(Subtraction operations)
        // SUBV       - Sub with vector
        inline SIMDVecKNC_f sub(SIMDVecKNC_f const & b) {
            return SIMDVecKNC_f(_mm512_sub_ps(mVec, b.mVec));
        }
        inline SIMDVecKNC_f operator- (SIMDVecKNC_f const & b) {
            return this->sub(b);
        }
        // MSUBV      - Masked sub with vector
        inline SIMDVecKNC_f sub(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // SUBS       - Sub with scalar
        inline SIMDVecKNC_f sub(float b) {
            return SIMDVecKNC_f(_mm512_sub_ps(mVec, _mm512_set1_ps(b)));
        }
        inline SIMDVecKNC_f operator- (float b) {
            return this->sub(b);
        }
        // MSUBS      - Masked subtraction with scalar
        inline SIMDVecKNC_f sub(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }
        // SUBVA      - Sub with vector and assign
        inline SIMDVecKNC_f & suba(SIMDVecKNC_f const & b) {
            mVec = _mm512_sub_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_f & operator-= (SIMDVecKNC_f const & b) {
            return suba(b);
        }
        // MSUBVA     - Masked sub with vector and assign
        inline SIMDVecKNC_f & suba(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA      - Sub with scalar and assign
        inline SIMDVecKNC_f & suba(float b) {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        inline SIMDVecKNC_f & operator-= (float b) {
            return this->suba(b);
        }
        // MSUBSA     - Masked sub with scalar and assign
        inline SIMDVecKNC_f & suba(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        inline SIMDVecKNC_f subfrom(SIMDVecKNC_f const & a) {
            return SIMDVecKNC_f(_mm512_sub_ps(a.mVec, mVec));
        }
        // MSUBFROMV  - Masked sub from vector
        inline SIMDVecKNC_f subfrom(SIMDMask16 const & mask, SIMDVecKNC_f const & a) {
            __m512 t0 = _mm512_mask_sub_ps(a.mVec, mask.mMask, a.mVec, mVec);
            return SIMDVecKNC_f(t0);
        }
        // SUBFROMS   - Sub from scalar (promoted to vector)
        inline SIMDVecKNC_f subfrom(float a) {
            return SIMDVecKNC_f(_mm512_sub_ps(_mm512_set1_ps(a), mVec));
        }
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        inline SIMDVecKNC_f subfrom(SIMDMask16 const & mask, float a) {
            __m512 t0 = _mm512_set1_ps(a);
            __m512 t1 = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return SIMDVecKNC_f(t1);
        }
        // SUBFROMVA  - Sub from vector and assign
        inline SIMDVecKNC_f & subfroma(SIMDVecKNC_f const & a) {
            mVec = _mm512_sub_ps(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA - Masked sub from vector and assign
        inline SIMDVecKNC_f & subfroma(SIMDMask16 const & mask, SIMDVecKNC_f const & a) {
            mVec = _mm512_mask_sub_ps(a.mVec, mask.mMask, a.mVec, mVec);
            return *this;
        }
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        inline SIMDVecKNC_f subfroma(float a) {
            mVec = _mm512_sub_ps(_mm512_set1_ps(a), mVec);
            return *this;
        }
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        inline SIMDVecKNC_f & subfroma(SIMDMask16 const & mask, float a) {
            __m512 t0 = _mm512_set1_ps(a);
            mVec = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC    - Postfix decrement
        inline SIMDVecKNC_f postdec() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_sub_ps(mVec, t0);
            return t1;
        }
        inline SIMDVecKNC_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC   - Masked postfix decrement
        inline SIMDVecKNC_f postdec(SIMDMask16 const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return t1;
        }
        // PREFDEC    - Prefix decrement
        inline SIMDVecKNC_f & prefdec() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_sub_ps(mVec, t0);
            return *this;
        }
        inline SIMDVecKNC_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC   - Masked prefix decrement
        inline SIMDVecKNC_f & prefdec(SIMDMask16 const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVecKNC_f mul(SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mul_ps(mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        inline SIMDVecKNC_f operator* (SIMDVecKNC_f const & b) {
            return this->mul(b);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecKNC_f mul(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }

        // MULS   - Multiplication with scalar
        inline SIMDVecKNC_f mul(float b) {
            __m512 t0 = _mm512_mul_ps(mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        inline SIMDVecKNC_f operator* (float b) {
            return this->mul(b);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecKNC_f mul(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }

        // MULVA  - Multiplication with vector and assign
        inline SIMDVecKNC_f & mula(SIMDVecKNC_f const & b) {
            mVec = _mm512_mul_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVecKNC_f & operator*= (SIMDVecKNC_f const & b) {
            return mula(b);
        }
        // MMULVA - Masked multiplication with vector and assign
        inline SIMDVecKNC_f & mula(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA  - Multiplication with scalar and assign
        inline SIMDVecKNC_f & mula(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mul_ps(mVec, t0);
            return *this;
        }
        inline SIMDVecKNC_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA - Masked multiplication with scalar and assign
        inline SIMDVecKNC_f & mula(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        //(Division operations)
        // DIVV   - Division with vector
        inline SIMDVecKNC_f div(SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_div_ps(mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        inline SIMDVecKNC_f operator/ (SIMDVecKNC_f const & b) {
            return this->div(b);
        }
        // MDIVV  - Masked division with vector
        inline SIMDVecKNC_f div(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            __m512 t0 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // DIVS   - Division with scalar
        inline SIMDVecKNC_f div(float b) {
            __m512 t0 = _mm512_div_ps(mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        inline SIMDVecKNC_f operator/ (float b) {
            return this->div(b);
        }
        // MDIVS  - Masked division with scalar
        inline SIMDVecKNC_f div(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVecKNC_f(t1);
        }
        // DIVVA  - Division with vector and assign
        inline SIMDVecKNC_f & diva(SIMDVecKNC_f const & b) {
            mVec = _mm512_div_ps(mVec, b.mVec);
            return *this;
        }
        // MDIVVA - Masked division with vector and assign
        inline SIMDVecKNC_f & diva(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA  - Division with scalar and assign
        inline SIMDVecKNC_f & diva(float b) {
            mVec = _mm512_div_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MDIVSA - Masked division with scalar and assign
        inline SIMDVecKNC_f & diva(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RCP    - Reciprocal
        inline SIMDVecKNC_f rcp() {
            __m512 t0 = _mm512_rcp23_ps(mVec);
            return SIMDVecKNC_f(t0);
        }
        // MRCP   - Masked reciprocal
        inline SIMDVecKNC_f rcp(SIMDMask16 const & mask) {
            __m512 t0 = _mm512_mask_rcp23_ps(mVec, mask.mMask, mVec);
            return SIMDVecKNC_f(t0);
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVecKNC_f rcp(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            __m512 t2 = _mm512_mul_ps(t0, t1);
            return SIMDVecKNC_f(t2);
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVecKNC_f rcp(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            __m512 t2 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return SIMDVecKNC_f(t2);
        }
        // RCPA   - Reciprocal and assign
        inline SIMDVecKNC_f & rcpa() {
            mVec = _mm512_rcp23_ps(mVec);
            return *this;
        }
        // MRCPA  - Masked reciprocal and assign
        inline SIMDVecKNC_f & rcpa(SIMDMask16 const & mask) {
            mVec = _mm512_mask_rcp23_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RCPSA  - Reciprocal with scalar and assign
        inline SIMDVecKNC_f & rcpa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            mVec = _mm512_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA - Masked reciprocal with scalar and assign
        inline SIMDVecKNC_f & rcpa(SIMDMask16 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return *this;
        }

        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        inline SIMDMask16 cmpeq(SIMDVecKNC_f const & b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPEQS - Element-wise 'equal' with scalar
        inline SIMDMask16 cmpeq(float b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, _mm512_set1_ps(b));
            return SIMDMask16(m0);
        }
        // CMPNEV - Element-wise 'not equal' with vector
        inline SIMDMask16 cmpne(SIMDVecKNC_f const & b) const {
            __mmask16 m0 = _mm512_cmpneq_ps_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPNES - Element-wise 'not equal' with scalar
        inline SIMDMask16 cmpne(float b) const {
            __mmask16 m0 = _mm512_cmpneq_ps_mask(mVec, _mm512_set1_ps(b));
            return SIMDMask16(m0);
        }
        // CMPGTV - Element-wise 'greater than' with vector
        inline SIMDMask16 cmpgt(SIMDVecKNC_f const & b) const {
            //__mmask16 m0 = _mm512_cmpgt_ps_mask(mVec, b.mVec);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, b.mVec, 14);
            return SIMDMask16(m0);
        }
        // CMPGTS - Element-wise 'greater than' with scalar
        inline SIMDMask16 cmpgt(float b) const {
            //__mmask16 m0 = _mm512_cmpgt_ps_mask(mVec, _mm512_set1_ps(b));
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 14);
            return SIMDMask16(m0);
        }
        // CMPLTV - Element-wise 'less than' with vector
        inline SIMDMask16 cmplt(SIMDVecKNC_f const & b) const {
            __mmask16 m0 = _mm512_cmplt_ps_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPLTS - Element-wise 'less than' with scalar
        inline SIMDMask16 cmplt(float b) const {
            __mmask16 m0 = _mm512_cmplt_ps_mask(mVec, _mm512_set1_ps(b));
            return SIMDMask16(m0);
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        inline SIMDMask16 cmpge(SIMDVecKNC_f const & b) const {
            //__mmask16 m0 = _mm512_cmpge_ps_mask(mVec, b.mVec);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, b.mVec, 13);
            return SIMDMask16(m0);
        }
        // CMPGES - Element-wise 'greater than or equal' with scalar
        inline SIMDMask16 cmpge(float b) const {
            //__mmask16 m0 = _mm512_cmpge_ps_mask(mVec, _mm512_set1_ps(b));
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 13);
            return SIMDMask16(m0);
        }
        // CMPLEV - Element-wise 'less than or equal' with vector
        inline SIMDMask16 cmple(SIMDVecKNC_f const & b) const {
            __mmask16 m0 = _mm512_cmple_ps_mask(mVec, b.mVec);
            return SIMDMask16(m0);
        }
        // CMPLES - Element-wise 'less than or equal' with scalar
        inline SIMDMask16 cmple(float b) const {
            __mmask16 m0 = _mm512_cmple_ps_mask(mVec, _mm512_set1_ps(b));
            return SIMDMask16(m0);
        }
        // CMPEV  - Check if vectors are exact (returns scalar 'bool')
        inline bool cmpe(SIMDVecKNC_f const & b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, b.mVec);
            return m0 == 0xFF;
        }
        // CMPES - Check if all vector elements are equal to scalar value
        inline bool cmpe(float b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, _mm512_set1_ps(b));
            return m0 == 0xFF;
        }

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
        inline float hadd() const {
            alignas(64) uint32_t raw[16];
            union {
                float    retval_f;
                uint32_t retval_u;
            };
            retval_u = 0;

            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                retval_u += raw[i];
            }
            return retval_f;
        }
        // MHADD - Masked add elements of a vector (horizontal add)
        inline float hadd(SIMDMask16 const & mask) const {
            alignas(64) uint32_t raw[16];
            union {
                float    retval_f;
                uint32_t retval_u;
            };
            retval_u = 0;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                if (mask.mMask & (1 << i)) retval_u += raw[i];
            }
            return retval_f;
        }
        // HMUL  - Multiply elements of a vector (horizontal mul)
        inline float hmul() const {
            alignas(64) uint32_t raw[16];
            union {
                float    retval_f;
                uint32_t retval_u;
            };
            retval_u = 1;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                retval_u *= raw[i];
            }
            return retval_f;
        }
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        inline float hmul(SIMDMask16 const & mask) const {
            alignas(64) uint32_t raw[16];
            union {
                float    retval_f;
                uint32_t retval_u;
            };
            retval_u = 1;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                if (mask.mMask & (1 << i)) retval_u *= raw[i];
            }
            return retval_f;
        }

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVecKNC_f fmuladd(SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) const {
            __m512 t0 = _mm512_fmadd_ps(mVec, b.mVec, c.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecKNC_f fmuladd(SIMDMask16 const & mask, SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) const {
            __m512 t0 = _mm512_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVecKNC_f(t0);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        inline SIMDVecKNC_f fmulsub(SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) {
            __m512 t0 = _mm512_fmsub_ps(mVec, b.mVec, c.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        inline SIMDVecKNC_f fmulsub(SIMDMask16 const & mask, SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) {
            __m512 t0 = _mm512_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVecKNC_f(t0);
        }
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        inline SIMDVecKNC_f faddmul(SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVecKNC_f(t1);
        }
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        inline SIMDVecKNC_f faddmul(SIMDMask16 const & mask, SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(t0, mask.mMask, t0, c.mVec);
            return SIMDVecKNC_f(t1);
        }
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        inline SIMDVecKNC_f fsubmul(SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) const {
            __m512 t0 = _mm512_sub_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVecKNC_f(t1);
        }
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
        inline SIMDVecKNC_f fsubmul(SIMDMask16 const & mask, SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(t0, mask.mMask, t0, c.mVec);
            return SIMDVecKNC_f(t1);
        }

        // (Mathematical operations)
        // MAXV   - Max with vector
        inline SIMDVecKNC_f max(SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_gmax_ps(mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MMAXV  - Masked max with vector
        inline SIMDVecKNC_f max(SIMDMask16 const & mask, SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MAXS   - Max with scalar
        inline SIMDVecKNC_f max(float b) const {
            __m512 t0 = _mm512_gmax_ps(mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MMAXS  - Masked max with scalar
        inline SIMDVecKNC_f max(SIMDMask16 const & mask, float b) const {
            __m512 t0 = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MAXVA  - Max with vector and assign
        inline SIMDVecKNC_f & maxa(SIMDVecKNC_f const & b) {
            mVec = _mm512_gmax_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA - Masked max with vector and assign
        inline SIMDVecKNC_f & maxa(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA  - Max with scalar (promoted to vector) and assign
        inline SIMDVecKNC_f & maxa(float b) {
            mVec = _mm512_gmax_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        inline SIMDVecKNC_f & maxa(SIMDMask16 const & mask, float b) {
            mVec = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MINV   - Min with vector
        inline SIMDVecKNC_f min(SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_gmin_ps(mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MMINV  - Masked min with vector
        inline SIMDVecKNC_f min(SIMDMask16 const & mask, SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVecKNC_f(t0);
        }
        // MINS   - Min with scalar (promoted to vector)
        inline SIMDVecKNC_f min(float b) const {
            __m512 t0 = _mm512_gmin_ps(mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MMINS  - Masked min with scalar (promoted to vector)
        inline SIMDVecKNC_f min(SIMDMask16 const & mask, float b) const {
            __m512 t0 = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0);
        }
        // MINVA  - Min with vector and assign
        inline SIMDVecKNC_f & mina(SIMDVecKNC_f const & b) {
            mVec = _mm512_gmin_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA - Masked min with vector and assign
        inline SIMDVecKNC_f & mina(SIMDMask16 const & mask, SIMDVecKNC_f const & b) {
            mVec = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA  - Min with scalar (promoted to vector) and assign
        inline SIMDVecKNC_f & mina(float b) {
            mVec = _mm512_gmin_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        inline SIMDVecKNC_f & mina(SIMDMask16 const & mask, float b) {
            mVec = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return *this;
        }
        // HMAX   - Max of elements of a vector (horizontal max)
        inline float hmax() {
            return _mm512_reduce_gmax_ps(mVec);
        }
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        inline float hmax(SIMDMask16 const & mask) {
            return _mm512_mask_reduce_gmax_ps(mask.mMask, mVec);
        }
        // IMAX   - Index of max element of a vector
        // MIMAX  - Masked index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        inline float hmin() {
            return _mm512_reduce_gmin_ps(mVec);
        }
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        inline float hmin(SIMDMask16 const & mask) {
            return _mm512_mask_reduce_gmin_ps(mask.mMask, mVec);
        }
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
        inline SIMDVecKNC_f operator-  () const {
            return neg();
        }
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign

        // (Mathematical functions)
        // ABS   - Absolute value
        inline SIMDVecKNC_f abs() const {
            return SIMDVecKNC_f(_mm512_abs_ps(mVec));
        }
        // MABS  - Masked absolute value
        inline SIMDVecKNC_f abs(SIMDMask16 const & mask) const {
            __m512 t0 = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return SIMDVecKNC_f(t0);
        }
        // ABSA  - Absolute value and assign
        inline SIMDVecKNC_f & abs() {
            mVec = _mm512_abs_ps(mVec);
            return *this;
        }
        // MABSA - Masked absolute value and assign
        inline SIMDVecKNC_f & abx(SIMDMask16 const & mask) {
            mVec = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return *this;
        }

        // 5) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR       - Square of vector values
        inline SIMDVecKNC_f sqr() const {
            __m512 t0 = _mm512_mul_ps(mVec, mVec);
            return SIMDVecKNC_f(t0);
        }
        // MSQR      - Masked square of vector values
        inline SIMDVecKNC_f sqr(SIMDMask16 const & mask) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return SIMDVecKNC_f(t0);
        }
        // SQRA      - Square of vector values and assign
        inline SIMDVecKNC_f & sqra() {
            mVec = _mm512_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA     - Masked square of vector values and assign
        inline SIMDVecKNC_f & sqra(SIMDMask16 const & mask) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT      - Square root of vector values
        inline SIMDVecKNC_f sqrt() const {
            return SIMDVecKNC_f(_mm512_sqrt_ps(mVec));
        }
        // MSQRT     - Masked square root of vector values 
        inline SIMDVecKNC_f sqrt(SIMDMask16 const & mask) const {
            __m512 t0 = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return SIMDVecKNC_f(t0);
        }
        // SQRTA     - Square root of vector values and assign
        inline SIMDVecKNC_f & sqrta() {
            mVec = _mm512_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA    - Masked square root of vector values and assign
        inline SIMDVecKNC_f & sqrta(SIMDMask16 const & mask) {
            mVec = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RSQRT     - Reciprocal square root
        inline SIMDVecKNC_f rsqr() const {
            return SIMDVecKNC_f(_mm512_rsqrt23_ps(mVec));
        }
        // MRSQRT    - Masked reciprocal square root
        inline SIMDVecKNC_f rsqrt(SIMDMask16 const & mask) const {
            return SIMDVecKNC_f(_mm512_mask_rsqrt23_ps(mVec, mask.mMask, mVec));
        }
        // RSQRTA    - Reciprocal square root and assign
        inline SIMDVecKNC_f & rsqrta() {
            mVec = _mm512_rsqrt23_ps(mVec);
            return *this;
        }
        // MRSQRTA   - Masked reciprocal square root and assign
        inline SIMDVecKNC_f & rsqrta(SIMDMask16 const & mask) {
            mVec = _mm512_mask_rsqrt23_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        inline SIMDVecKNC_f round() const {
            __m512 t0 = _mm512_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
            return SIMDVecKNC_f(t0);
        }
        // MROUND    - Masked round to nearest integer
        inline SIMDVecKNC_f round(SIMDMask16 const & mask) const {
            __m512 t0 = _mm512_mask_round_ps(mVec, mask.mMask, mVec, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
            return SIMDVecKNC_f(t0);
        }
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        inline SIMDVecKNC_f floor() const {
            __m512 t0 = _mm512_round_ps(mVec, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE);
            return SIMDVecKNC_f(t0);
        }
        // MFLOOR    - Masked floor
        inline SIMDVecKNC_f floor(SIMDMask16 const & mask) const {
            __m512 t0 = _mm512_mask_round_ps(mVec, mask.mMask, mVec, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE);
            return SIMDVecKNC_f(t0);
        }
        // CEIL      - Ceil
        inline SIMDVecKNC_f ceil() const {
            __m512 t0 = _mm512_round_ps(mVec, _MM_FROUND_TO_POS_INF, _MM_EXPADJ_NONE);
            return SIMDVecKNC_f(t0);
        }
        // MCEIL     - Masked ceil
        inline SIMDVecKNC_f ceil(SIMDMask16 const & mask) const {
            __m512 t0 = _mm512_mask_round_ps(mVec, mask.mMask, mVec, _MM_FROUND_TO_POS_INF, _MM_EXPADJ_NONE);
            return SIMDVecKNC_f(t0);
        }
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        inline SIMDMask16 isinf() const {
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_slli_epi32(t0, 1);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(t1, _mm512_set1_epi32(0xFF000000));
            return SIMDMask16(m0);
        }
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        inline SIMDMask16 isnan() const {
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_slli_epi32(t0, 1);
            __m512i t2 = _mm512_set1_epi32(0xFF000000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512i t4 = _mm512_andnot_epi32(t1, t2);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(t4, _mm512_set1_epi32(0));
            return SIMDMask16(m0 && m1);
        }
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
    class SIMDVecKNC_f<float, 32> :
        public SIMDVecFloatInterface<
        SIMDVecKNC_f<float, 32>,
        SIMDVecKNC_u<uint32_t, 32>,
        SIMDVecKNC_i<int32_t, 32>,
        float,
        32,
        uint32_t,
        SIMDMask32,
        SIMDSwizzle32>,
        public SIMDVecPackableInterface<
        SIMDVecKNC_f<float, 32>,
        SIMDVecKNC_f<float, 16 >>
    {
    public:
        typedef typename SIMDVecKNC_f_traits<float, 32>::VEC_UINT_TYPE  VEC_UINT_TYPE;
        typedef typename SIMDVecKNC_f_traits<float, 32>::VEC_INT_TYPE   VEC_INT_TYPE;
        typedef typename SIMDVecKNC_f_traits<float, 32>::MASK_TYPE      MASK_TYPE;

    private:
        __m512 mVecLo;
        __m512 mVecHi;

        inline SIMDVecKNC_f(__m512 & xLo, __m512 & xHi) {
            this->mVecLo = xLo;
            this->mVecHi = xHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVecKNC_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVecKNC_f(float f) {
            mVecLo = _mm512_set1_ps(f);
            mVecHi = _mm512_set1_ps(f);
        }

        // UTOF
        inline explicit SIMDVecKNC_f(VEC_UINT_TYPE const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVecKNC_f(VEC_INT_TYPE const & vecInt) {

        }


        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecKNC_f(float const * p) { this->load(p); }


        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVecKNC_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11,
            float f12, float f13, float f14, float f15,
            float f16, float f17, float f18, float f19,
            float f20, float f21, float f22, float f23,
            float f24, float f25, float f26, float f27,
            float f28, float f29, float f30, float f31) {
            mVecLo = _mm512_setr_ps(f0, f1, f2, f3,
                f4, f5, f6, f7,
                f8, f9, f10, f11,
                f12, f13, f14, f15);
            mVecHi = _mm512_setr_ps(f16, f17, f18, f19,
                f20, f21, f22, f23,
                f24, f25, f26, f27,
                f28, f29, f30, f31);
        }

        // EXTRACT - Extract single element from a vector
        inline float extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) float raw[16];
            if (index < 16) {
                _mm512_store_ps(raw, mVecLo);
                return raw[index];
            }
            else {
                _mm512_store_ps(raw, mVecHi);
                return raw[index - 16];
            }
        }

        // EXTRACT - Extract single element from a vector
        inline float operator[] (uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVecKNC_f, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVecKNC_f, MASK_TYPE>(mask, static_cast<SIMDVecKNC_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVecKNC_f & insert(uint32_t index, float value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(64) float raw[16];
            if (index < 16) {
                _mm512_store_ps(raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm512_load_ps(raw);
            }
            else {
                _mm512_store_ps(raw, mVecHi);
                raw[index - 16] = value;
                mVecHi = _mm512_load_ps(raw);
            }
            return *this;
        }

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
        inline SIMDVecKNC_f & load(float const * p) {
            if ((uint64_t(p) % 64) == 0) {
                mVecLo = _mm512_load_ps(p);
                mVecHi = _mm512_load_ps(p + 16);
            }
            else {
                alignas(64) float raw[32];
                memcpy(raw, p, 32 * sizeof(float));
                mVecLo = _mm512_load_ps(raw);
                mVecHi = _mm512_load_ps(raw + 16);
            }
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVecKNC_f & loada(float const * p) {
            mVecLo = _mm512_load_ps(p);
            mVecHi = _mm512_load_ps(p + 16);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float * store(float * p)
        {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_ps(p, mVecLo);
                _mm512_store_ps(p + 16, mVecHi);
            }
            else {
                alignas(64) float raw[32];
                _mm512_store_ps(raw, mVecLo);
                _mm512_store_ps(raw + 16, mVecHi);
                memcpy(p, raw, 32 * sizeof(float));
                return p;
            }
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) {
            _mm512_store_ps(p, mVecLo);
            _mm512_store_ps(p + 16, mVecHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory

        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVecKNC_f add(SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVecLo, b.mVecLo);
            __m512 t1 = _mm512_add_ps(mVecHi, b.mVecHi);
            return SIMDVecKNC_f(t0, t1);
        }
        // MADDV    - Masked add with vector
        inline SIMDVecKNC_f add(SIMDMask32 const & mask, SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, b.mVecLo);
            __m512 t1 = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, b.mVecHi);
            return SIMDVecKNC_f(t0, t1);
        }
        // ADDS     - Add with scalar
        inline SIMDVecKNC_f add(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_add_ps(mVecLo, t0);
            __m512 t2 = _mm512_add_ps(mVecHi, t0);
            return SIMDVecKNC_f(t1, t2);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVecKNC_f add(SIMDMask32 const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, t0);
            __m512 t2 = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, t0);
            return SIMDVecKNC_f(t1, t2);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVecKNC_f & adda(SIMDVecKNC_f const & b) {
            mVecLo = _mm512_add_ps(mVecLo, b.mVecLo);
            mVecHi = _mm512_add_ps(mVecHi, b.mVecHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVecKNC_f & adda(SIMDMask32 const & mask, SIMDVecKNC_f const & b) {
            mVecLo = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, b.mVecLo);
            mVecHi = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, b.mVecHi);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVecKNC_f & adda(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVecLo = _mm512_add_ps(mVecLo, t0);
            mVecHi = _mm512_add_ps(mVecHi, t0);
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVecKNC_f & adda(SIMDMask32 const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVecLo = _mm512_mask_add_ps(mVecLo, mask.mMaskLo, mVecLo, t0);
            mVecHi = _mm512_mask_add_ps(mVecHi, mask.mMaskHi, mVecHi, t0);
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
        inline SIMDVecKNC_f mul(SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mul_ps(mVecLo, b.mVecLo);
            __m512 t1 = _mm512_mul_ps(mVecHi, b.mVecHi);
            return SIMDVecKNC_f(t0, t1);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVecKNC_f mul(SIMDMask32 const & mask, SIMDVecKNC_f const & b) const {
            __m512 t0 = _mm512_mask_mul_ps(mVecLo, mask.mMaskLo, mVecLo, b.mVecLo);
            __m512 t1 = _mm512_mask_mul_ps(mVecHi, mask.mMaskHi, mVecHi, b.mVecHi);
            return SIMDVecKNC_f(t0, t1);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVecKNC_f mul(float b) {
            __m512 t0 = _mm512_mul_ps(mVecLo, _mm512_set1_ps(b));
            __m512 t1 = _mm512_mul_ps(mVecHi, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0, t1);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVecKNC_f mul(SIMDMask32 const & mask, float b) {
            __m512 t0 = _mm512_mask_mul_ps(mVecLo, mask.mMaskLo, mVecLo, _mm512_set1_ps(b));
            __m512 t1 = _mm512_mask_mul_ps(mVecHi, mask.mMaskHi, mVecHi, _mm512_set1_ps(b));
            return SIMDVecKNC_f(t0, t1);
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
        inline SIMDVecKNC_f fmuladd(SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) {
            __m512 t0 = _mm512_fmadd_ps(mVecLo, b.mVecLo, c.mVecLo);
            __m512 t1 = _mm512_fmadd_ps(mVecHi, b.mVecHi, c.mVecHi);
            return SIMDVecKNC_f(t0, t1);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVecKNC_f fmuladd(SIMDMask32 const & mask, SIMDVecKNC_f const & b, SIMDVecKNC_f const & c) {
            __m512 t0 = _mm512_mask_fmadd_ps(mVecLo, mask.mMaskLo, b.mVecLo, c.mVecLo);
            __m512 t1 = _mm512_mask_fmadd_ps(mVecHi, mask.mMaskHi, b.mVecHi, c.mVecHi);
            return SIMDVecKNC_f(t0, t1);
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
}
}

#endif
