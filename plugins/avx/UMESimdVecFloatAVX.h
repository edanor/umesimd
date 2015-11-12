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

#ifndef UME_SIMD_VEC_FLOAT_H_
#define UME_SIMD_VEC_FLOAT_H_

#include <type_traits>
#include "../../UMESimdInterface.h"
#include "../UMESimdPluginScalarEmulation.h"
#include <immintrin.h>

#include "UMESimdMaskAVX.h"
#include "UMESimdSwizzleAVX.h"
#include "UMESimdVecUintAVX.h"
#include "UMESimdVecIntAVX.h"

namespace UME {
namespace SIMD {

    // ********************************************************************************************
    // FLOATING POINT VECTORS
    // ********************************************************************************************

    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    struct SIMDVec_f_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 32b vectors
    template<>
    struct SIMDVec_f_traits<float, 1> {
        typedef SIMDVec_u<uint32_t, 1>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 1>   VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDVecSwizzle<1>       SWIZZLE_MASK_TYPE;
    };

    // 64b vectors
    template<>
    struct SIMDVec_f_traits<float, 2> {
        typedef SIMDVec_f<float, 1>     HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 2>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 2>   VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDVecSwizzle<2>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_f_traits<double, 1> {
        typedef SIMDVec_u<uint64_t, 1>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 1>   VEC_INT_TYPE;
        typedef int64_t                 SCALAR_INT_TYPE;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double*                 SCALAR_TYPE_PTR;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDVecSwizzle<1>       SWIZZLE_MASK_TYPE;
    };

    // 128b vectors
    template<>
    struct SIMDVec_f_traits<float, 4> {
        typedef SIMDVec_f<float, 2>     HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 4>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 4>   VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDVecSwizzle<4>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_f_traits<double, 2> {
        typedef SIMDVec_f<double, 1>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 2>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 2>   VEC_INT_TYPE;
        typedef int64_t                 SCALAR_INT_TYPE;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double*                 SCALAR_TYPE_PTR;
        typedef SIMDVecMask<2>          MASK_TYPE;
        typedef SIMDVecSwizzle<2>       SWIZZLE_MASK_TYPE;
    };

    // 256b vectors
    template<>
    struct SIMDVec_f_traits<float, 8> {
        typedef SIMDVec_f<float, 4>     HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 8>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 8>   VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDVecSwizzle<8>       SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_f_traits<double, 4> {
        typedef SIMDVec_f<double, 2>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 4>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 4>   VEC_INT_TYPE;
        typedef int64_t                 SCALAR_INT_TYPE;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double*                 SCALAR_TYPE_PTR;
        typedef SIMDVecMask<4>          MASK_TYPE;
        typedef SIMDVecSwizzle<4>       SWIZZLE_MASK_TYPE;
    };

    // 512b vectors
    template<>
    struct SIMDVec_f_traits<float, 16> {
        typedef SIMDVec_f<float, 8>     HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 16> VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 16>  VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDVecSwizzle<16>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_f_traits<double, 8> {
        typedef SIMDVec_f<double, 4>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 8>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 8>   VEC_INT_TYPE;
        typedef int64_t                 SCALAR_INT_TYPE;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double*                 SCALAR_TYPE_PTR;
        typedef SIMDVecMask<8>          MASK_TYPE;
        typedef SIMDVecSwizzle<8>       SWIZZLE_MASK_TYPE;
    };

    // 1024b vectors
    template<>
    struct SIMDVec_f_traits<float, 32> {
        typedef SIMDVec_f<float, 16>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 32> VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 32>  VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<32>         MASK_TYPE;
        typedef SIMDVecSwizzle<32>      SWIZZLE_MASK_TYPE;
    };

    template<>
    struct SIMDVec_f_traits<double, 16> {
        typedef SIMDVec_f<double, 8>    HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 16> VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 16>  VEC_INT_TYPE;
        typedef int64_t                 SCALAR_INT_TYPE;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double*                 SCALAR_TYPE_PTR;
        typedef SIMDVecMask<16>         MASK_TYPE;
        typedef SIMDVecSwizzle<16>      SWIZZLE_MASK_TYPE;
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
    class SIMDVec_f final :
        public SIMDVecFloatInterface<
        SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
        typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
        typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
        SCALAR_FLOAT_TYPE,
        VEC_LEN,
        typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
        typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE,
        typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SWIZZLE_MASK_TYPE>,
        public SIMDVecPackableInterface<
        SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
        typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::HALF_LEN_VEC_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, VEC_LEN> VEC_EMU_REG;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE SCALAR_UINT_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_INT_TYPE SCALAR_INT_TYPE;
        typedef SIMDVec_f VEC_TYPE;

        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE VEC_UINT_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE  VEC_INT_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::MASK_TYPE     MASK_TYPE;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVec_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};

        // UTOF
        inline explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {
            alignas(alignment()) SCALAR_UINT_TYPE raw[VEC_LEN];
            vecUint.storea(raw);
            for (int i = 0; i < VEC_LEN; i++) {
                mVec[i] = SCALAR_FLOAT_TYPE(raw[i]);
            }
        }

        // ITOF
        inline explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {
            alignas(alignment()) SCALAR_INT_TYPE raw[VEC_LEN];
            vecInt.storea(raw);
            for (int i = 0; i < VEC_LEN; i++) {
                mVec[i] = SCALAR_FLOAT_TYPE(raw[i]);
            }
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(SCALAR_FLOAT_TYPE const *p) { this->load(p); };

        inline SIMDVec_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec.insert(0, f0); mVec.insert(1, f1);
        }

        inline SIMDVec_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3) {
            mVec.insert(0, f0);  mVec.insert(1, f1);  mVec.insert(2, f2);  mVec.insert(3, f3);
        }

        inline SIMDVec_f(
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

        inline SIMDVec_f(
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

        inline SIMDVec_f(
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
        inline IntermediateMask<SIMDVec_f, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>() const {
            SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> retval;
            for (uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }

        inline operator SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>() const {
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            for (uint32_t i = 0; i < VEC_LEN; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
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
    class SIMDVec_f<SCALAR_FLOAT_TYPE, 1> final :
        public SIMDVecFloatInterface<
            SIMDVec_f<SCALAR_FLOAT_TYPE, 1>,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE,
            1,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::SWIZZLE_MASK_TYPE>
    {
    public:
        typedef SIMDVecEmuRegister<SCALAR_FLOAT_TYPE, 1> VEC_EMU_REG;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_UINT_TYPE SCALAR_UINT_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::SCALAR_INT_TYPE SCALAR_INT_TYPE;

        typedef SIMDVec_f VEC_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_UINT_TYPE VEC_UINT_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::VEC_INT_TYPE VEC_INT_TYPE;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, 1>::MASK_TYPE       MASK_TYPE;
    private:
        VEC_EMU_REG mVec;

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() : mVec() {};

        // SET-CONSTR
        inline explicit SIMDVec_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};

        // UTOF
        inline explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {
            mVec.insert(0, SCALAR_FLOAT_TYPE(vecUint[0]));
        }

        // ITOF
        inline explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {
            mVec.insert(0, SCALAR_FLOAT_TYPE(vecInt[0]));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(SCALAR_FLOAT_TYPE const *p) { this->load(p); }

        // Override Access operators
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return mVec[index];
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<1>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }

        inline operator SIMDVec_u<SCALAR_UINT_TYPE, 1>() const {
            SIMDVec_u<SCALAR_UINT_TYPE, 1> retval;
            for (uint32_t i = 0; i < 1; i++) {
                retval.insert(i, (SCALAR_UINT_TYPE)mVec[i]);
            }
            return retval;
        }

        inline operator SIMDVec_i<SCALAR_INT_TYPE, 1>() const {
            SIMDVec_i<SCALAR_INT_TYPE, 1> retval;
            for (uint32_t i = 0; i < 1; i++) {
                retval.insert(i, (SCALAR_INT_TYPE)mVec[i]);
            }
            return retval;
        }
    };

    // ********************************************************************************************
    // FLOATING POINT VECTOR specializations
    // ********************************************************************************************
    template<>
    class SIMDVec_f<float, 1> final:
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 1>,
            SIMDVec_u<uint32_t, 1>,
            SIMDVec_i<int32_t, 1>,
            float,
            1,
            uint32_t,
            SIMDVecMask<1>,
            SIMDVecSwizzle<1>>
    {
    private:
        float mVec;

        typedef SIMDVec_u<uint32_t, 1>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 1>     VEC_INT_TYPE;
        typedef SIMDVec_f<float, 1>       HALF_LEN_VEC_TYPE;
    public:

        constexpr static uint32_t alignment() {
            return 4;
        }

        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline explicit SIMDVec_f(float f) {
            mVec = f;
        }

        // UTOF
        inline explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {
            mVec = float(vecUint[0]);
        }

        // FTOU
        inline VEC_UINT_TYPE ftou() const {
            return VEC_UINT_TYPE(uint32_t(mVec));
        }

        // ITOF
        inline explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {
            mVec = float(vecInt[0]);
        }

        // FTOI
        inline VEC_INT_TYPE ftoi() const {
            return VEC_UINT_TYPE(int32_t(mVec));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const *p) {
            mVec = p[0];
        }
        
        // EXTRACT
        inline float extract(uint32_t index) const {
            return mVec;
        }

        // EXTRACT
        inline float operator[] (uint32_t index) const {
            return mVec;
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<1>> operator[] (SIMDVecMask<1> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<1>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            mVec = value;
            return *this;
        }
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV    - Masked assignment with another vector
        inline SIMDVec_f & assign(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec = b.mVec;
            return *this;
        }
        // ASSIGNS     - Assignment with scalar
        inline SIMDVec_f & assign(float b) {
            mVec = b;
            return *this;
        }
        // MASSIGNS    - Masked assign with scalar
        inline SIMDVec_f & assign(SIMDVecMask<1> const & mask, float b) {
            if (mask.mMask == true) mVec = b;
            return *this;
        }
        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVec_f & load(float const * p) {
            mVec = p[0];
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //        vector
        inline SIMDVec_f & load(SIMDVecMask<1> const & mask, float const * p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVec = p[0];
            return *this;
        }

        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<1> const & mask, float const * p) {
            if (mask.mMask == true) mVec = p[0];
            return *this;
        }

        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float * p) const {
            p[0] = mVec;
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //        unaligned)
        inline float* store(SIMDVecMask<1> const & mask, float * p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float * p) const {
            p[0] = mVec;
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDVecMask<1> const & mask, float * p) const {
            if (mask.mMask == true) p[0] = mVec;
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec + b.mVec;
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask ? mVec + b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float a) const {
            float t0 = mVec + a;
            return SIMDVec_f(t0);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<1> const & mask, float b) const {
            float t0 = mask.mMask ? mVec + b : mVec;
            return SIMDVec_f(t0);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec += b.mVec;
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            mVec = mask.mMask ? mVec + b.mVec : mVec;
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float a) {
            mVec += a;
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<1> const & mask, float b) {
            mVec = mask.mMask ? mVec + b : mVec;
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
        inline SIMDVec_f postinc() {
            float t0 = mVec++;
            return SIMDVec_f(t0);
        }
        // MPOSTINC - Masked postfix increment
        inline SIMDVec_f postinc(SIMDVecMask<1> const & mask) {
            float t0 = (mask.mMask == true) ? mVec++ : mVec;
            return SIMDVec_f(t0);
        }
        // PREFINC  - Prefix increment
        inline SIMDVec_f & prefinc() {
            ++mVec;
            return *this;
        }
        // MPREFINC - Masked prefix increment
        inline SIMDVec_f & prefinc(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) ++mVec;
            return *this;
        }
        //(Subtraction operations)
        // SUBV       - Sub with vector
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec - b.mVec;
            return SIMDVec_f(t0);
        }
        // MSUBV      - Masked sub with vector
        inline SIMDVec_f sub(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = (mask.mMask == true) ? (mVec - b.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // SUBS       - Sub with scalar
        inline SIMDVec_f sub(float b) const {
            float t0 = mVec - b;
            return SIMDVec_f(t0);
        }
        // MSUBS      - Masked subtraction with scalar
        inline SIMDVec_f sub(SIMDVecMask<1> const & mask, float b) const {
            float t0 = (mask.mMask == true) ? (mVec - b) : mVec;
            return SIMDVec_f(t0);
        }
        // SUBVA      - Sub with vector and assign
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = mVec - b.mVec;
            return *this;
        }
        // MSUBVA     - Masked sub with vector and assign
        inline SIMDVec_f & suba(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec = mVec - b.mVec;
            return *this;
        }
        // SUBSA      - Sub with scalar and assign
        inline SIMDVec_f & suba(const float b) {
            mVec = mVec - b;
            return *this;
        }
        // MSUBSA     - Masked sub with scalar and assign
        inline SIMDVec_f & suba(SIMDVecMask<1> const & mask, const float b) {
            if (mask.mMask == true) mVec = mVec - b;
            return *this;
        }
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        inline SIMDVec_f subfrom(SIMDVec_f const & a) const {
            float t0 = a.mVec - mVec;
            return SIMDVec_f(t0);
        }
        // MSUBFROMV  - Masked sub from vector
        inline SIMDVec_f subfrom(SIMDVecMask<1> const & mask, SIMDVec_f const & a) const {
            float t0 = (mask.mMask == true) ? (a.mVec - mVec) : a[0];
            return SIMDVec_f(t0);
        }
        // SUBFROMS   - Sub from scalar (promoted to vector)
        inline SIMDVec_f subfrom(float a) const {
            float t0 = a - mVec;
            return SIMDVec_f(t0);
        }
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        inline SIMDVec_f subfrom(SIMDVecMask<1> const & mask, float a) const {
            float t0 = (mask.mMask == true) ? (a - mVec) : a;
            return SIMDVec_f(t0);
        }
        // SUBFROMVA  - Sub from vector and assign
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = a.mVec - mVec;
            return *this;
        }
        // MSUBFROMVA - Masked sub from vector and assign
        inline SIMDVec_f & subfroma(SIMDVecMask<1> const & mask, SIMDVec_f const & a) {
            mVec = (mask.mMask == true) ? (a.mVec - mVec) : a.mVec;
            return *this;
        }
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        inline SIMDVec_f & subfroma(float a) {
            mVec = a - mVec;
            return *this;
        }
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        inline SIMDVec_f & subfroma(SIMDVecMask<1> const & mask, float a) {
            mVec = (mask.mMask == true) ? (a - mVec) : a;
            return *this;
        }
        // POSTDEC    - Postfix decrement
        inline SIMDVec_f postdec() {
            float t0 = mVec--;
            return SIMDVec_f(t0);
        }
        // MPOSTDEC   - Masked postfix decrement
        inline SIMDVec_f postdec(SIMDVecMask<1> const & mask) {
            float t0 = (mask.mMask == true) ? mVec-- : mVec;
            return SIMDVec_f(t0);
        }
        // PREFDEC    - Prefix decrement
        inline SIMDVec_f & prefdec() {
            --mVec;
            return *this;
        }
        // MPREFDEC   - Masked prefix decrement
        inline SIMDVec_f & prefdec(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) --mVec;
            return *this;
        }
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec * b.mVec;
            return SIMDVec_f(t0);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask ? mVec * b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) const {
            float t0 = mVec * b;
            return SIMDVec_f(t0);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<1> const & mask, float b) const {
            float t0 = mask.mMask ? mVec * b : mVec;
            return SIMDVec_f(t0);
        }
        // MULVA  - Multiplication with vector and assign
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec *= b.mVec;
            return *this;
        }
        // MMULVA - Masked multiplication with vector and assign
        inline SIMDVec_f & mula(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec *= b.mVec;
            return *this;
        }
        // MULSA  - Multiplication with scalar and assign
        inline SIMDVec_f & mula(float b) {
            mVec *= b;
            return *this;
        }
        // MMULSA - Masked multiplication with scalar and assign
        inline SIMDVec_f & mula(SIMDVecMask<1> const & mask, float b) {
            if (mask.mMask == true) mVec *= b;
            return *this;
        }

        //(Division operations)
        // DIVV   - Division with vector
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec / b.mVec;
            return SIMDVec_f(t0);
        }
        // MDIVV  - Masked division with vector
        inline SIMDVec_f div(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask ? mVec / b.mVec : mVec;
            return SIMDVec_f(t0);
        }
        // DIVS   - Division with scalar
        inline SIMDVec_f div(float b) const {
            float t0 = mVec / b;
            return SIMDVec_f(t0);
        }
        // MDIVS  - Masked division with scalar
        inline SIMDVec_f div(SIMDVecMask<1> const & mask, float b) const {
            float t0 = mask.mMask ? mVec / b : mVec;
            return SIMDVec_f(t0);
        }
        // DIVVA  - Division with vector and assign
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec /= b.mVec;
            return *this;
        }
        // MDIVVA - Masked division with vector and assign
        inline SIMDVec_f & diva(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if (mask.mMask == true) mVec /= b.mVec;
            return *this;
        }
        // DIVSA  - Division with scalar and assign
        inline SIMDVec_f & diva(float b) {
            mVec /= b;
            return *this;
        }
        // MDIVSA - Masked division with scalar and assign
        inline SIMDVec_f & diva(SIMDVecMask<1> const & mask, float b) {
            if (mask.mMask == true) mVec /= b;
            return *this;
        }
        // RCP    - Reciprocal
        inline SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec;
            return SIMDVec_f(t0);
        }
        // MRCP   - Masked reciprocal
        inline SIMDVec_f rcp(SIMDVecMask<1> const & mask) const {
            float t0 = mask.mMask ? 1.0f / mVec : mVec;
            return SIMDVec_f(t0);
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVec_f rcp(float b) const {
            float t0 = b / mVec;
            return SIMDVec_f(t0);
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVec_f rcp(SIMDVecMask<1> const & mask, float b) const {
            float t0 = mask.mMask ? b / mVec : mVec;
            return SIMDVec_f(t0);
        }
        // RCPA   - Reciprocal and assign
        inline SIMDVec_f & rcpa() {
            mVec = 1.0f / mVec;
            return *this;
        }
        // MRCPA  - Masked reciprocal and assign
        inline SIMDVec_f & rcpa(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec = 1.0f / mVec;
            return *this;
        }
        // RCPSA  - Reciprocal with scalar and assign
        inline SIMDVec_f & rcpa(float b) {
            mVec = b / mVec;
            return *this;
        }
        // MRCPSA - Masked reciprocal with scalar and assign
        inline SIMDVec_f & rcpa(SIMDVecMask<1> const & mask, float b) {
            if (mask.mMask == true) mVec = b / mVec;
            return *this;
        }

        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        inline SIMDVecMask<1> cmpeq(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec == b.mVec;
            return mask;
        }
        // CMPEQS - Element-wise 'equal' with scalar
        inline SIMDVecMask<1> cmpeq(float b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec == b;
            return mask;
        }
        // CMPNEV - Element-wise 'not equal' with vector
        inline SIMDVecMask<1> cmpne(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec != b.mVec;
            return mask;
        }
        // CMPNES - Element-wise 'not equal' with scalar
        inline SIMDVecMask<1> cmpne(float b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec != b;
            return mask;
        }
        // CMPGTV - Element-wise 'greater than' with vector
        inline SIMDVecMask<1> cmpgt(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec > b.mVec;
            return mask;
        }
        // CMPGTS - Element-wise 'greater than' with scalar
        inline SIMDVecMask<1> cmpgt(float b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec > b;
            return mask;
        }
        // CMPLTV - Element-wise 'less than' with vector
        inline SIMDVecMask<1> cmplt(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec < b.mVec;
            return mask;
        }
        // CMPLTS - Element-wise 'less than' with scalar
        inline SIMDVecMask<1> cmplt(float b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec < b;
            return mask;
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        inline SIMDVecMask<1> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec >= b.mVec;
            return mask;
        }
        // CMPGES - Element-wise 'greater than or equal' with scalar
        inline SIMDVecMask<1> cmpge(float b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec >= b;
            return mask;
        }
        // CMPLEV - Element-wise 'less than or equal' with vector
        inline SIMDVecMask<1> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec <= b.mVec;
            return mask;
        }
        // CMPLES - Element-wise 'less than or equal' with scalar
        inline SIMDVecMask<1> cmple(float b) const {
            SIMDVecMask<1> mask;
            mask.mMask = mVec <= b;
            return mask;
        }
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
        inline bool cmpex(SIMDVec_f const & b) const {
            return (b.mVec == mVec);
        }

        // (Pack/Unpack operations - not available for SIMD1)

        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        inline SIMDVec_f blend(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = (mask.mMask == true) ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        //         assign
        inline SIMDVec_f blend(SIMDVecMask<1> const & mask, float b) const {
            float t0 = (mask.mMask == true) ? mVec : b;
            return SIMDVec_f(t0);
        }
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        inline float hadd() const {
            return mVec;
        }
        // MHADD - Masked add elements of a vector (horizontal add)
        inline float hadd(SIMDVecMask<1> const & mask) const {
            float t0 = 0.0f;
            if (mask.mMask == true) t0 += mVec;
            return t0;
        }
        // HMUL  - Multiply elements of a vector (horizontal mul)
        inline float hmul() const {
            return mVec;
        }
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        inline float hmul(SIMDVecMask<1> const & mask) const {
            float t0 = 1.0f;
            if (mask.mMask == true) t0 *= mVec;
            return t0;
        }

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec * b.mVec + c.mVec;
            return SIMDVec_f(t0);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask == true) ? (mVec * b.mVec + c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec * b.mVec - c.mVec;
            return SIMDVec_f(t0);
        }
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        inline SIMDVec_f fmulsub(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask == true) ? (mVec * b.mVec - c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec + b.mVec) * c.mVec;
            return SIMDVec_f(t0);
        }
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        inline SIMDVec_f faddmul(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask == true) ? ((mVec + b.mVec) * c.mVec) : mVec;
            return SIMDVec_f(t0);
        }
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec - b.mVec) * c.mVec;
            return SIMDVec_f(t0);
        }
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
        inline SIMDVec_f fsubmul(SIMDVecMask<1> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask == true) ? ((mVec - b.mVec) * c.mVec) : mVec;
            return SIMDVec_f(t0);
        }

        // (Mathematical operations)
        // MAXV   - Max with vector
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec > b.mVec ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // MMAXV  - Masked max with vector
        inline SIMDVec_f max(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > b.mVec) ? mVec : b.mVec;
            }
            return SIMDVec_f(t0);
        }
        // MAXS   - Max with scalar
        inline SIMDVec_f max(float b) const {
            float t0 = mVec > b ? mVec : b;
            return SIMDVec_f(t0);
        }
        // MMAXS  - Masked max with scalar
        inline SIMDVec_f max(SIMDVecMask<1> const & mask, float b) const {
            float t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec > b) ? mVec : b;
            }
            return SIMDVec_f(t0);
        }
        // MAXVA  - Max with vector and assign
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            if (mVec < b.mVec) mVec = b.mVec;
            return *this;
        }
        // MMAXVA - Masked max with vector and assign
        inline SIMDVec_f & maxa(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask == true) && (mVec < b.mVec)) mVec = b.mVec;
            return *this;
        }
        // MAXSA  - Max with scalar (promoted to vector) and assign
        inline SIMDVec_f & maxa(float b) {
            if (mVec < b) mVec = b;
            return *this;
        }
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        inline SIMDVec_f & maxa(SIMDVecMask<1> const & mask, float b) {
            if ((mask.mMask == true) && (mVec < b)) mVec = b;
            return *this;
        }
        // MINV   - Min with vector
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec < b.mVec ? mVec : b.mVec;
            return SIMDVec_f(t0);
        }
        // MMINV  - Masked min with vector
        inline SIMDVec_f min(SIMDVecMask<1> const & mask, SIMDVec_f const & b) const {
            float t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b.mVec) ? mVec : b.mVec;
            }
            return SIMDVec_f(t0);
        }
        // MINS   - Min with scalar (promoted to vector)
        inline SIMDVec_f min(float b) const {
            float t0 = mVec < b ? mVec : b;
            return SIMDVec_f(t0);
        }
        // MMINS  - Masked min with scalar (promoted to vector)
        inline SIMDVec_f min(SIMDVecMask<1> const & mask, float b) const {
            float t0 = mVec;
            if (mask.mMask == true) {
                t0 = (mVec < b) ? mVec : b;
            }
            return SIMDVec_f(t0);
        }
        // MINVA  - Min with vector and assign
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            if (mVec > b.mVec) mVec = b.mVec;
            return *this;
        }
        // MMINVA - Masked min with vector and assign
        inline SIMDVec_f & mina(SIMDVecMask<1> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask == true) && (mVec > b.mVec)) mVec = b.mVec;
            return *this;
        }
        // MINSA  - Min with scalar (promoted to vector) and assign
        inline SIMDVec_f & mina(float b) {
            if (mVec > b) mVec = b;
            return *this;
        }
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        inline SIMDVec_f & mina(SIMDVecMask<1> const & mask, float b) {
            if ((mask.mMask == true) && (mVec > b)) mVec = b;
            return *this;
        }
        // HMAX   - Max of elements of a vector (horizontal max)
        inline float hmax() const {
            return mVec;
        }
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        inline float hmax(SIMDVecMask<1> const & mask) const {
            float t0 = std::numeric_limits<float>::min();
            if (mask.mMask == true) t0 = mVec;
            return t0;
        }
        // IMAX   - Index of max element of a vector
        inline uint32_t imax() const {
            return 0;
        }
        // MIMAX  - Masked index of max element of a vector
        inline uint32_t mimax(SIMDVecMask<1> const & mask) const {
            return 0;
        }
        // HMIN   - Min of elements of a vector (horizontal min)
        inline float hmin() const {
            return mVec;
        }
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        inline float mhmin(SIMDVecMask<1> const & mask) const {
            float t0 = std::numeric_limits<float>::max();
            if (mask.mMask == true) t0 = mVec;
            return t0;
        }
        // IMIN   - Index of min element of a vector
        inline uint32_t imin() const {
            return 0;
        }
        // MIMIN  - Masked index of min element of a vector
        inline uint32_t mimin(SIMDVecMask<1> const & mask) const {
            return 0;
        }

        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        inline SIMDVec_f & gather(float * baseAddr, uint64_t * indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERS  - Masked gather from memory using indices from array
        inline SIMDVec_f & gather(SIMDVecMask<1> const & mask, float * baseAddr, uint64_t * indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // GATHERV   - Gather from memory using indices from vector
        inline SIMDVec_f & gather(float * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec = baseAddr[indices[0]];
            return *this;
        }
        // MGATHERV  - Masked gather from memory using indices from vector
        inline SIMDVec_f & gather(SIMDVecMask<1> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask == true) mVec = baseAddr[indices[0]];
            return *this;
        }
        // SCATTERS  - Scatter to memory using indices from array
        inline float * scatter(float * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTERS - Masked scatter to memory using indices from array
        inline float * scatter(SIMDVecMask<1> const & mask, float * baseAddr, uint64_t * indices) const {
            if (mask.mMask == true) baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // SCATTERV  - Scatter to memory using indices from vector
        inline float * scatter(float * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // MSCATTERV - Masked scatter to memory using indices from vector
        inline float * scatter(SIMDVecMask<1> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask == true)  baseAddr[indices[0]] = mVec;
            return baseAddr;
        }
        // NEG   - Negate signed values
        inline SIMDVec_f neg() const {
            return SIMDVec_f(-mVec);
        }
        // MNEG  - Masked negate signed values
        inline SIMDVec_f neg(SIMDVecMask<1> const & mask) const {
            float t0 = (mask.mMask == true) ? -mVec : mVec;
            return SIMDVec_f(t0);
        }
        // NEGA  - Negate signed values and assign
        inline SIMDVec_f & nega() {
            mVec = -mVec;
            return *this;
        }
        // MNEGA - Masked negate signed values and assign
        inline SIMDVec_f & nega(SIMDVecMask<1> const & mask) {
            if (mask.mMask == true) mVec = -mVec;
            return *this;
        }

        // (Mathematical functions)
        // ABS   - Absolute value
        inline SIMDVec_f abs() const {
            float t0 = (mVec > 0.0f) ? mVec : -mVec;
            return SIMDVec_f(t0);
        }
        // MABS  - Masked absolute value
        inline SIMDVec_f abs(SIMDVecMask<1> const & mask) const {
            float t0 = ((mask.mMask == true) && (mVec < 0.0f)) ? -mVec : mVec;
            return SIMDVec_f(t0);
        }
        // ABSA  - Absolute value and assign
        inline SIMDVec_f & absa() {
            if (mVec < 0.0f) mVec = -mVec;
            return *this;
        }
        // MABSA - Masked absolute value and assign
        inline SIMDVec_f & absa(SIMDVecMask<1> const & mask) {
            if ((mask.mMask == true) && (mVec < 0.0f)) mVec = -mVec;
            return *this;
        }

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
        inline SIMDVec_i<int32_t, 1> trunc() {
            int32_t t0 = (int32_t)mVec;
            return SIMDVec_i<int32_t, 1>(t0);
        }
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        inline SIMDVec_i<int32_t, 1> trunc(SIMDVecMask<1> const & mask) {
            int32_t t0 = mask.mMask ? (int32_t)mVec : 0;
            return SIMDVec_i<int32_t, 1>(t0);
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
    class SIMDVec_f<float, 2> final :
        public SIMDVecFloatInterface<
        SIMDVec_f<float, 2>,
        SIMDVec_u<uint32_t, 2>,
        SIMDVec_i<int32_t, 2>,
        float,
        2,
        uint32_t,
        SIMDVecMask<2>,
        SIMDVecSwizzle<2 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_f<float, 2>,
        SIMDVec_f<float, 1 >>
    {
    private:
        float mVec[2];

        typedef SIMDVec_u<uint32_t, 2>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 2>     VEC_INT_TYPE;
        typedef SIMDVec_f<float, 1>       HALF_LEN_VEC_TYPE;
    public:

        constexpr static uint32_t alignment() {
            return 4;
        }

        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(float f) {
            mVec[0] = f;
            mVec[1] = f;
        }

        // UTOF
        inline explicit SIMDVec_f(VEC_UINT_TYPE const & vecUint) {
            mVec[0] = float(vecUint[0]);
            mVec[1] = float(vecUint[1]);
        }

        // FTOU
        inline VEC_UINT_TYPE ftou() const {
            return VEC_UINT_TYPE(uint32_t(mVec[0]), uint32_t(mVec[1]));
        }

        // ITOF
        inline explicit SIMDVec_f(VEC_INT_TYPE const & vecInt) {
            mVec[0] = float(vecInt[0]);
            mVec[1] = float(vecInt[1]);
        }

        // FTOI
        inline VEC_INT_TYPE ftoi() const {
            return VEC_UINT_TYPE(int32_t(mVec[0]), int32_t(mVec[1]));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const *p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(float x_lo, float x_hi) {
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
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<2>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            mVec[index & 1] = value;
            return *this;
        }
        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        // MASSIGNV    - Masked assignment with another vector
        inline SIMDVec_f & assign(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] = b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b.mVec[1];
            return *this;
        }
        // ASSIGNS     - Assignment with scalar
        inline SIMDVec_f & assign(float b) {
            mVec[0] = b;
            mVec[1] = b;
            return *this;
        }
        // MASSIGNS    - Masked assign with scalar
        inline SIMDVec_f & assign(SIMDVecMask<2> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] = b;
            if (mask.mMask[1] == true) mVec[1] = b;
            return *this;
        }
        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVec_f & load(float const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //        vector
        inline SIMDVec_f & load(SIMDVecMask<2> const & mask, float const * p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            return *this;
        }
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVec[0] = p[0];
            mVec[1] = p[1];
            return *this;
        }

        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<2> const & mask, float const * p) {
            if (mask.mMask[0] == true) mVec[0] = p[0];
            if (mask.mMask[1] == true) mVec[1] = p[1];
            return *this;
        }

        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //        unaligned)
        inline float* store(SIMDVecMask<2> const & mask, float * p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            return p;
        }
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float * p) const {
            p[0] = mVec[0];
            p[1] = mVec[1];
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDVecMask<2> const & mask, float * p) const {
            if (mask.mMask[0] == true) p[0] = mVec[0];
            if (mask.mMask[1] == true) p[1] = mVec[1];
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            float t0 = mVec[0] + b.mVec[0];
            float t1 = mVec[1] + b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float a) const {
            float t0 = mVec[0] + a;
            float t1 = mVec[1] + a;
            return SIMDVec_f(t0, t1);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<2> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] + b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] + b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] += b.mVec[0];
            mVec[1] += b.mVec[1];
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] + b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] + b.mVec[1] : mVec[1];
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float a) {
            mVec[0] += a;
            mVec[1] += a;
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<2> const & mask, float b) {
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
        inline SIMDVec_f postinc() {
            float t0 = mVec[0]++;
            float t1 = mVec[1]++;
            return SIMDVec_f(t0, t1);
        }
        // MPOSTINC - Masked postfix increment
        inline SIMDVec_f postinc(SIMDVecMask<2> const & mask) {
            float t0 = (mask.mMask[0] == true) ? mVec[0]++ : mVec[0];
            float t1 = (mask.mMask[1] == true) ? mVec[1]++ : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // PREFINC  - Prefix increment
        inline SIMDVec_f & prefinc() {
            ++mVec[0];
            ++mVec[1];
            return *this;
        }
        // MPREFINC - Masked prefix increment
        inline SIMDVec_f & prefinc(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) ++mVec[0];
            if (mask.mMask[1] == true) ++mVec[1];
            return *this;
        }
        //(Subtraction operations)
        // SUBV       - Sub with vector
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            float t0 = mVec[0] - b.mVec[0];
            float t1 = mVec[1] - b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBV      - Masked sub with vector
        inline SIMDVec_f sub(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = (mask.mMask[0] == true) ? (mVec[0] - b.mVec[0]) : mVec[0];
            float t1 = (mask.mMask[1] == true) ? (mVec[1] - b.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBS       - Sub with scalar
        inline SIMDVec_f sub(float b) const {
            float t0 = mVec[0] - b;
            float t1 = mVec[1] - b;
            return SIMDVec_f(t0, t1);
        }
        // MSUBS      - Masked subtraction with scalar
        inline SIMDVec_f sub(SIMDVecMask<2> const & mask, float b) const {
            float t0 = (mask.mMask[0] == true) ? (mVec[0] - b) : mVec[0];
            float t1 = (mask.mMask[1] == true) ? (mVec[1] - b) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBVA      - Sub with vector and assign
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = mVec[0] - b.mVec[0];
            mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        // MSUBVA     - Masked sub with vector and assign
        inline SIMDVec_f & suba(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] = mVec[0] - b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] = mVec[1] - b.mVec[1];
            return *this;
        }
        // SUBSA      - Sub with scalar and assign
        inline SIMDVec_f & suba(const float b) {
            mVec[0] = mVec[0] - b;
            mVec[1] = mVec[1] - b;
            return *this;
        }
        // MSUBSA     - Masked sub with scalar and assign
        inline SIMDVec_f & suba(SIMDVecMask<2> const & mask, const float b) {
            if (mask.mMask[0] == true) mVec[0] = mVec[0] - b;
            if (mask.mMask[1] == true) mVec[1] = mVec[1] - b;
            return *this;
        }
        // SSUBV      - Saturated sub with vector
        // MSSUBV     - Masked saturated sub with vector
        // SSUBS      - Saturated sub with scalar
        // MSSUBS     - Masked saturated sub with scalar
        // SSUBVA     - Saturated sub with vector and assign
        // MSSUBVA    - Masked saturated sub with vector and assign
        // SSUBSA     - Saturated sub with scalar and assign
        // MSSUBSA    - Masked saturated sub with scalar and assign
        // SUBFROMV   - Sub from vector
        inline SIMDVec_f subfrom(SIMDVec_f const & a) const {
            float t0 = a.mVec[0] - mVec[0];
            float t1 = a.mVec[1] - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV  - Masked sub from vector
        inline SIMDVec_f subfrom(SIMDVecMask<2> const & mask, SIMDVec_f const & a) const {
            float t0 = (mask.mMask[0] == true) ? (a.mVec[0] - mVec[0]) : a[0];
            float t1 = (mask.mMask[1] == true) ? (a.mVec[1] - mVec[1]) : a[1];
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS   - Sub from scalar (promoted to vector)
        inline SIMDVec_f subfrom(float a) const {
            float t0 = a - mVec[0];
            float t1 = a - mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        inline SIMDVec_f subfrom(SIMDVecMask<2> const & mask, float a) const {
            float t0 = (mask.mMask[0] == true) ? (a - mVec[0]) : a;
            float t1 = (mask.mMask[1] == true) ? (a - mVec[1]) : a;
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMVA  - Sub from vector and assign
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec[0] = a.mVec[0] - mVec[0];
            mVec[1] = a.mVec[1] - mVec[1];
            return *this;
        }
        // MSUBFROMVA - Masked sub from vector and assign
        inline SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, SIMDVec_f const & a) {
            mVec[0] = (mask.mMask[0] == true) ? (a.mVec[0] - mVec[0]) : a.mVec[0];
            mVec[1] = (mask.mMask[1] == true) ? (a.mVec[1] - mVec[1]) : a.mVec[1];
            return *this;
        }
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        inline SIMDVec_f & subfroma(float a) {
            mVec[0] = a - mVec[0];
            mVec[1] = a - mVec[1];
            return *this;
        }
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        inline SIMDVec_f & subfroma(SIMDVecMask<2> const & mask, float a) {
            mVec[0] = (mask.mMask[0] == true) ? (a - mVec[0]) : a;
            mVec[1] = (mask.mMask[1] == true) ? (a - mVec[1]) : a;
            return *this;
        }
        // POSTDEC    - Postfix decrement
        inline SIMDVec_f postdec() {
            float t0 = mVec[0]--;
            float t1 = mVec[1]--;
            return SIMDVec_f(t0, t1);
        }
        // MPOSTDEC   - Masked postfix decrement
        inline SIMDVec_f postdec(SIMDVecMask<2> const & mask) {
            float t0 = (mask.mMask[0] == true) ? mVec[0]-- : mVec[0];
            float t1 = (mask.mMask[1] == true) ? mVec[1]-- : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // PREFDEC    - Prefix decrement
        inline SIMDVec_f & prefdec() {
            --mVec[0];
            --mVec[1];
            return *this;
        }
        // MPREFDEC   - Masked prefix decrement
        inline SIMDVec_f & prefdec(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) --mVec[0];
            if (mask.mMask[1] == true) --mVec[1];
            return *this;
        }
        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            float t0 = mVec[0] * b.mVec[0];
            float t1 = mVec[1] * b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) const {
            float t0 = mVec[0] * b;
            float t1 = mVec[1] * b;
            return SIMDVec_f(t0, t1);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<2> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] * b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] * b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MULVA  - Multiplication with vector and assign
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] *= b.mVec[0];
            mVec[1] *= b.mVec[1];
            return *this;
        }
        // MMULVA - Masked multiplication with vector and assign
        inline SIMDVec_f & mula(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] *= b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] *= b.mVec[1];
            return *this;
        }
        // MULSA  - Multiplication with scalar and assign
        inline SIMDVec_f & mula(float b) {
            mVec[0] *= b;
            mVec[1] *= b;
            return *this;
        }
        // MMULSA - Masked multiplication with scalar and assign
        inline SIMDVec_f & mula(SIMDVecMask<2> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] *= b;
            if (mask.mMask[1] == true) mVec[1] *= b;
            return *this;
        }

        //(Division operations)
        // DIVV   - Division with vector
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            float t0 = mVec[0] / b.mVec[0];
            float t1 = mVec[1] / b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MDIVV  - Masked division with vector
        inline SIMDVec_f div(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVS   - Division with scalar
        inline SIMDVec_f div(float b) const {
            float t0 = mVec[0] / b;
            float t1 = mVec[1] / b;
            return SIMDVec_f(t0, t1);
        }
        // MDIVS  - Masked division with scalar
        inline SIMDVec_f div(SIMDVecMask<2> const & mask, float b) const {
            float t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            float t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // DIVVA  - Division with vector and assign
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            return *this;
        }
        // MDIVVA - Masked division with vector and assign
        inline SIMDVec_f & diva(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if (mask.mMask[0] == true) mVec[0] /= b.mVec[0];
            if (mask.mMask[1] == true) mVec[1] /= b.mVec[1];
            return *this;
        }
        // DIVSA  - Division with scalar and assign
        inline SIMDVec_f & diva(float b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        // MDIVSA - Masked division with scalar and assign
        inline SIMDVec_f & diva(SIMDVecMask<2> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] /= b;
            if (mask.mMask[1] == true) mVec[1] /= b;
            return *this;
        }
        // RCP    - Reciprocal
        inline SIMDVec_f rcp() const {
            float t0 = 1.0f / mVec[0];
            float t1 = 1.0f / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCP   - Masked reciprocal
        inline SIMDVec_f rcp(SIMDVecMask<2> const & mask) const {
            float t0 = mask.mMask[0] ? 1.0f / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? 1.0f / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVec_f rcp(float b) const {
            float t0 = b / mVec[0];
            float t1 = b / mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVec_f rcp(SIMDVecMask<2> const & mask, float b) const {
            float t0 = mask.mMask[0] ? b / mVec[0] : mVec[0];
            float t1 = mask.mMask[1] ? b / mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // RCPA   - Reciprocal and assign
        inline SIMDVec_f & rcpa() {
            mVec[0] = 1.0f / mVec[0];
            mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // MRCPA  - Masked reciprocal and assign
        inline SIMDVec_f & rcpa(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = 1.0f / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = 1.0f / mVec[1];
            return *this;
        }
        // RCPSA  - Reciprocal with scalar and assign
        inline SIMDVec_f & rcpa(float b) {
            mVec[0] = b / mVec[0];
            mVec[1] = b / mVec[1];
            return *this;
        }
        // MRCPSA - Masked reciprocal with scalar and assign
        inline SIMDVec_f & rcpa(SIMDVecMask<2> const & mask, float b) {
            if (mask.mMask[0] == true) mVec[0] = b / mVec[0];
            if (mask.mMask[1] == true) mVec[1] = b / mVec[1];
            return *this;
        }

        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        inline SIMDVecMask<2> cmpeq(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] == b.mVec[0];
            mask.mMask[1] = mVec[1] == b.mVec[1];
            return mask;
        }
        // CMPEQS - Element-wise 'equal' with scalar
        inline SIMDVecMask<2> cmpeq(float b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] == b;
            mask.mMask[1] = mVec[1] == b;
            return mask;
        }
        // CMPNEV - Element-wise 'not equal' with vector
        inline SIMDVecMask<2> cmpne(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] != b.mVec[0];
            mask.mMask[1] = mVec[1] != b.mVec[1];
            return mask;
        }
        // CMPNES - Element-wise 'not equal' with scalar
        inline SIMDVecMask<2> cmpne(float b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] != b;
            mask.mMask[1] = mVec[1] != b;
            return mask;
        }
        // CMPGTV - Element-wise 'greater than' with vector
        inline SIMDVecMask<2> cmpgt(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] > b.mVec[0];
            mask.mMask[1] = mVec[1] > b.mVec[1];
            return mask;
        }
        // CMPGTS - Element-wise 'greater than' with scalar
        inline SIMDVecMask<2> cmpgt(float b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] > b;
            mask.mMask[1] = mVec[1] > b;
            return mask;
        }
        // CMPLTV - Element-wise 'less than' with vector
        inline SIMDVecMask<2> cmplt(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] < b.mVec[0];
            mask.mMask[1] = mVec[1] < b.mVec[1];
            return mask;
        }
        // CMPLTS - Element-wise 'less than' with scalar
        inline SIMDVecMask<2> cmplt(float b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] < b;
            mask.mMask[1] = mVec[1] < b;
            return mask;
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        inline SIMDVecMask<2> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] >= b.mVec[0];
            mask.mMask[1] = mVec[1] >= b.mVec[1];
            return mask;
        }
        // CMPGES - Element-wise 'greater than or equal' with scalar
        inline SIMDVecMask<2> cmpge(float b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] >= b;
            mask.mMask[1] = mVec[1] >= b;
            return mask;
        }
        // CMPLEV - Element-wise 'less than or equal' with vector
        inline SIMDVecMask<2> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] <= b.mVec[0];
            mask.mMask[1] = mVec[1] <= b.mVec[1];
            return mask;
        }
        // CMPLES - Element-wise 'less than or equal' with scalar
        inline SIMDVecMask<2> cmple(float b) const {
            SIMDVecMask<2> mask;
            mask.mMask[0] = mVec[0] <= b;
            mask.mMask[1] = mVec[1] <= b;
            return mask;
        }
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')
        inline bool cmpex(SIMDVec_f const & b) const {
            bool t0 = (b.mVec[0] == mVec[0]) && (b.mVec[1] == mVec[1]);
            return t0;
        }
        // (Pack/Unpack operations - not available for SIMD1)
        // PACK     - assign vector with two half-length vectors
        inline SIMDVec_f & pack(HALF_LEN_VEC_TYPE const & a, HALF_LEN_VEC_TYPE const & b) {
            mVec[0] = a[0];
            mVec[1] = b[0];
            return *this;
        }
        // PACKLO   - assign lower half of a vector with a half-length vector
        inline SIMDVec_f packlo(HALF_LEN_VEC_TYPE const & a) {
            return SIMDVec_f(a[0], mVec[1]);
        }
        // PACKHI   - assign upper half of a vector with a half-length vector
        inline SIMDVec_f packhi(HALF_LEN_VEC_TYPE const & b) {
            return SIMDVec_f(mVec[0], b[0]);
        }
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        inline void unpack(HALF_LEN_VEC_TYPE & a, HALF_LEN_VEC_TYPE & b) {
            a.insert(0, mVec[0]);
            b.insert(0, mVec[1]);
        }
        // UNPACKLO - Unpack lower half and return as a half-length vector.
        inline HALF_LEN_VEC_TYPE unpacklo() const {
            return HALF_LEN_VEC_TYPE(mVec[0]);
        }
        // UNPACKHI - Unpack upper half and return as a half-length vector.
        inline HALF_LEN_VEC_TYPE unpackhi() const {
            return HALF_LEN_VEC_TYPE(mVec[1]);
        }

        //(Blend/Swizzle operations)
        // BLENDV   - Blend (mix) two vectors
        inline SIMDVec_f blend(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0 = (mask.mMask[0] == true) ? mVec[0] : b.mVec[0];
            float t1 = (mask.mMask[1] == true) ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // BLENDS   - Blend (mix) vector with scalar (promoted to vector)
        //         assign
        inline SIMDVec_f blend(SIMDVecMask<2> const & mask, float b) const {
            float t0 = (mask.mMask[0] == true) ? mVec[0] : b;
            float t1 = (mask.mMask[1] == true) ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        inline float hadd() const {
            return mVec[0] + mVec[1];
        }
        // MHADD - Masked add elements of a vector (horizontal add)
        inline float hadd(SIMDVecMask<2> const & mask) const {
            float t0 = 0.0f;
            if (mask.mMask[0] == true) t0 += mVec[0];
            if (mask.mMask[1] == true) t0 += mVec[1];
            return t0;
        }
        // HMUL  - Multiply elements of a vector (horizontal mul)
        inline float hmul() const {
            return mVec[0] * mVec[1];
        }
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        inline float hmul(SIMDVecMask<2> const & mask) const {
            float t0 = 1.0f;
            if (mask.mMask[0] == true) t0 *= mVec[0];
            if (mask.mMask[1] == true) t0 *= mVec[1];
            return t0;
        }

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask[0] == true) ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            float t1 = (mask.mMask[1] == true) ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            float t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        inline SIMDVec_f fmulsub(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask[0] == true) ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            float t1 = (mask.mMask[1] == true) ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        inline SIMDVec_f faddmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask[0] == true) ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = (mask.mMask[1] == true) ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            float t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
        inline SIMDVec_f fsubmul(SIMDVecMask<2> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            float t0 = (mask.mMask[0] == true) ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            float t1 = (mask.mMask[1] == true) ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_f(t0, t1);
        }

        // (Mathematical operations)
        // MAXV   - Max with vector
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            float t0 = mVec[0] > b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] > b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMAXV  - Masked max with vector
        inline SIMDVec_f max(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0, t1;
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > b.mVec[0]) ? mVec[0] : b.mVec[0];
            }
            else {
                t0 = mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > b.mVec[1]) ? mVec[1] : b.mVec[1];
            }
            else {
                t1 = mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MAXS   - Max with scalar
        inline SIMDVec_f max(float b) const {
            float t0 = mVec[0] > b ? mVec[0] : b;
            float t1 = mVec[1] > b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMAXS  - Masked max with scalar
        inline SIMDVec_f max(SIMDVecMask<2> const & mask, float b) const {
            float t0, t1;
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > b) ? mVec[0] : b;
            }
            else {
                t0 = mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > b) ? mVec[1] : b;
            }
            else {
                t1 = mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MAXVA  - Max with vector and assign
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            if (mVec[0] < b.mVec[0]) mVec[0] = b.mVec[0];
            if (mVec[1] < b.mVec[1]) mVec[1] = b.mVec[1];
            return *this;
        }
        // MMAXVA - Masked max with vector and assign
        inline SIMDVec_f & maxa(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask[0] == true) && (mVec[0] < b.mVec[0])) mVec[0] = b.mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < b.mVec[1])) mVec[1] = b.mVec[1];
            return *this;
        }
        // MAXSA  - Max with scalar (promoted to vector) and assign
        inline SIMDVec_f & maxa(float b) {
            if (mVec[0] < b) mVec[0] = b;
            if (mVec[1] < b) mVec[1] = b;
            return *this;
        }
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        inline SIMDVec_f & maxa(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask[0] == true) && (mVec[0] < b)) mVec[0] = b;
            if ((mask.mMask[1] == true) && (mVec[1] < b)) mVec[1] = b;
            return *this;
        }
        // MINV   - Min with vector
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            float t0 = mVec[0] < b.mVec[0] ? mVec[0] : b.mVec[0];
            float t1 = mVec[1] < b.mVec[1] ? mVec[1] : b.mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MMINV  - Masked min with vector
        inline SIMDVec_f min(SIMDVecMask<2> const & mask, SIMDVec_f const & b) const {
            float t0, t1;
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b.mVec[0]) ? mVec[0] : b.mVec[0];
            }
            else {
                t0 = mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b.mVec[1]) ? mVec[1] : b.mVec[1];
            }
            else {
                t1 = mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MINS   - Min with scalar (promoted to vector)
        inline SIMDVec_f min(float b) const {
            float t0 = mVec[0] < b ? mVec[0] : b;
            float t1 = mVec[1] < b ? mVec[1] : b;
            return SIMDVec_f(t0, t1);
        }
        // MMINS  - Masked min with scalar (promoted to vector)
        inline SIMDVec_f min(SIMDVecMask<2> const & mask, float b) const {
            float t0, t1;
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] < b) ? mVec[0] : b;
            }
            else {
                t0 = mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] < b) ? mVec[1] : b;
            }
            else {
                t1 = mVec[1];
            }
            return SIMDVec_f(t0, t1);
        }
        // MINVA  - Min with vector and assign
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            if (mVec[0] > b.mVec[0]) mVec[0] = b.mVec[0];
            if (mVec[1] > b.mVec[1]) mVec[1] = b.mVec[1];
            return *this;
        }
        // MMINVA - Masked min with vector and assign
        inline SIMDVec_f & mina(SIMDVecMask<2> const & mask, SIMDVec_f const & b) {
            if ((mask.mMask[0] == true) && (mVec[0] > b.mVec[0])) mVec[0] = b.mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] > b.mVec[1])) mVec[1] = b.mVec[1];
            return *this;
        }
        // MINSA  - Min with scalar (promoted to vector) and assign
        inline SIMDVec_f & mina(float b) {
            if (mVec[0] > b) mVec[0] = b;
            if (mVec[1] > b) mVec[1] = b;
            return *this;
        }
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        inline SIMDVec_f & mina(SIMDVecMask<2> const & mask, float b) {
            if ((mask.mMask[0] == true) && (mVec[0] > b)) mVec[0] = b;
            if ((mask.mMask[1] == true) && (mVec[1] > b)) mVec[1] = b;
            return *this;
        }
        // HMAX   - Max of elements of a vector (horizontal max)
        inline float hmax() const {
            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
        }
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        inline float hmax(SIMDVecMask<2> const & mask) const {
            float t0 = std::numeric_limits<float>::min();
            if (mask.mMask[0] == true) t0 = mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] > t0)) t0 = mVec[1];
            return t0;
        }
        // IMAX   - Index of max element of a vector
        inline uint32_t imax() const {
            uint32_t t0 = 0;
            if (mVec[0] < mVec[1]) t0 = 1;
            return t0;
        }
        // MIMAX  - Masked index of max element of a vector
        inline uint32_t mimax(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = 0;
            if (mask.mMask[1] == true) {
                if (mVec[0] < mVec[1]) t0 = 1;
            }
            return t0;
        }
        // HMIN   - Min of elements of a vector (horizontal min)
        inline float hmin() const {
            float t0 = mVec[0];
            if (mVec[0] > mVec[1]) t0 = mVec[1];
            return t0;
        }
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        inline float mhmin(SIMDVecMask<2> const & mask) const {
            float t0 = std::numeric_limits<float>::max();
            if (mask.mMask[0] == true) t0 = mVec[0];
            if (mask.mMask[1] == true) {
                if (t0 < mVec[1]) {
                    t0 = mVec[1];
                }
            }
            return t0;
        }
        // IMIN   - Index of min element of a vector
        inline uint32_t imin() const {
            uint32_t t0 = 0;
            if (mVec[0] > mVec[1]) t0 = 1;
            return t0;
        }
        // MIMIN  - Masked index of min element of a vector
        inline uint32_t mimin(SIMDVecMask<2> const & mask) const {
            uint32_t t0 = 0;
            if (mask.mMask[1] == true) {
                if (mVec[0] > mVec[1]) t0 = 1;
            }
            return t0;
        }

        // (Gather/Scatter operations)
        // GATHERS   - Gather from memory using indices from array
        inline SIMDVec_f & gather(float * baseAddr, uint64_t * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS  - Masked gather from memory using indices from array
        inline SIMDVec_f & gather(SIMDVecMask<2> const & mask, float * baseAddr, uint64_t * indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV   - Gather from memory using indices from vector
        inline SIMDVec_f & gather(float * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERV  - Masked gather from memory using indices from vector
        inline SIMDVec_f & gather(SIMDVecMask<2> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // SCATTERS  - Scatter to memory using indices from array
        inline float * scatter(float * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS - Masked scatter to memory using indices from array
        inline float * scatter(SIMDVecMask<2> const & mask, float * baseAddr, uint64_t * indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV  - Scatter to memory using indices from vector
        inline float * scatter(float * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV - Masked scatter to memory using indices from vector
        inline float * scatter(SIMDVecMask<2> const & mask, float * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask[0] == true)  baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // NEG   - Negate signed values
        inline SIMDVec_f neg() const {
            return SIMDVec_f(-mVec[0], -mVec[1]);
        }
        // MNEG  - Masked negate signed values
        inline SIMDVec_f neg(SIMDVecMask<2> const & mask) const {
            float t0 = (mask.mMask[0] == true) ? -mVec[0] : mVec[0];
            float t1 = (mask.mMask[1] == true) ? -mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // NEGA  - Negate signed values and assign
        inline SIMDVec_f & nega() {
            mVec[0] = -mVec[0];
            mVec[1] = -mVec[1];
            return *this;
        }
        // MNEGA - Masked negate signed values and assign
        inline SIMDVec_f & nega(SIMDVecMask<2> const & mask) {
            if (mask.mMask[0] == true) mVec[0] = -mVec[0];
            if (mask.mMask[1] == true) mVec[1] = -mVec[1];
            return *this;
        }

        // (Mathematical functions)
        // ABS   - Absolute value
        inline SIMDVec_f abs() const {
            float t0 = (mVec[0] > 0.0f) ? mVec[0] : -mVec[0];
            float t1 = (mVec[1] > 0.0f) ? mVec[1] : -mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // MABS  - Masked absolute value
        inline SIMDVec_f abs(SIMDVecMask<2> const & mask) const {
            float t0 = ((mask.mMask[0] == true) && (mVec[0] < 0.0f)) ? -mVec[0] : mVec[0];
            float t1 = ((mask.mMask[1] == true) && (mVec[1] < 0.0f)) ? -mVec[1] : mVec[1];
            return SIMDVec_f(t0, t1);
        }
        // ABSA  - Absolute value and assign
        inline SIMDVec_f & absa() {
            if (mVec[0] < 0.0f) mVec[0] = -mVec[0];
            if (mVec[1] < 0.0f) mVec[1] = -mVec[1];
            return *this;
        }
        // MABSA - Masked absolute value and assign
        inline SIMDVec_f & absa(SIMDVecMask<2> const & mask) {
            if ((mask.mMask[0] == true) && (mVec[0] < 0.0f)) mVec[0] = -mVec[0];
            if ((mask.mMask[1] == true) && (mVec[1] < 0.0f)) mVec[1] = -mVec[1];
            return *this;
        }

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
        inline SIMDVec_i<int32_t, 2> trunc() {
            int32_t t0 = (int32_t)mVec[0];
            int32_t t1 = (int32_t)mVec[1];
            return SIMDVec_i<int32_t, 2>(t0, t1);
        }
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        inline SIMDVec_i<int32_t, 2> trunc(SIMDVecMask<2> const & mask) {
            int32_t t0 = mask.mMask[0] ? (int32_t)mVec[0] : 0;
            int32_t t1 = mask.mMask[1] ? (int32_t)mVec[1] : 0;
            return SIMDVec_i<int32_t, 2>(t0, t1);
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
    class SIMDVec_f<float, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_i<int32_t, 4>,
            float,
            4,
            uint32_t,
            SIMDVecMask<4>,
            SIMDVecSwizzle<4>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2 >>
    {
    private:
        __m128 mVec;

        inline SIMDVec_f(__m128 const & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() {}

        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec = _mm_set1_ps(f);
        }

        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const * p) {
            mVec = _mm_loadu_ps(p);
        }

        // UTOF
        inline explicit SIMDVec_f(SIMDVec_u<uint32_t, 4> const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVec_f(SIMDVec_i<int32_t, 4>  const & vecInt) {

        }

        // FULL-CONSTR
        inline SIMDVec_f(float f0, float f1, float f2, float f3) {
            mVec = _mm_setr_ps(f0, f1, f2, f3);
        }

        // EXTRACT
        inline float extract(uint32_t index) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[index];
        }

        // EXTRACT
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
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

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVec_f & load(float const * p) {
            mVec = _mm_loadu_ps(p);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVec = _mm_load_ps(p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const * p) {
            __m128 t0 = _mm_load_ps(p);
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) {
            _mm_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) const {
            _mm_store_ps(p, mVec);
            return p;
        }

        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDVecMask<4> const & mask, float* p) const {
            _mm_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }

        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVec_f(t0);
        }

        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVec_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) const {
            return SIMDVec_f(_mm_add_ps(this->mVec, _mm_set1_ps(b)));
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return SIMDVec_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm_add_ps(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            mVec = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
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
        inline SIMDVec_f mul(SIMDVec_f const & b) {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVec_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) {
            return SIMDVec_f(_mm_mul_ps(mVec, _mm_set1_ps(b)));
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_mul_ps(mVec, _mm_set1_ps(b));
            return SIMDVec_f(_mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask)));
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

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

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
        inline float hadd() {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & a, SIMDVec_f const & b) {
#ifdef FMA
            return _mm_fmadd_ps(this->mVec, a.mVec, b.mVec);
#else
            return _mm_add_ps(_mm_mul_ps(this->mVec, a.mVec), b.mVec);
#endif
        }

        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & a, SIMDVec_f const & b) {
#ifdef FMA
            __m128 temp = _mm_fmadd_ps(this->mVec, a.mVec, b.mVec);
            return _mm_blendv_ps(temp, this->mVec, mask.mMask);
#else


            __m128 temp = _mm_add_ps(_mm_mul_ps(this->mVec, a.mVec), b.mVec);
            return _mm_blendv_ps(this->mVec, temp, _mm_cvtepi32_ps(mask.mMask));
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

        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign

        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign

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
    class SIMDVec_f<float, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_u<uint32_t, 8>,
            SIMDVec_i<int32_t, 8>,
            float,
            8,
            uint32_t,
            SIMDVecMask<8>,
            SIMDVecSwizzle<8>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_f<float, 4 >>
    {
    private:
        __m256 mVec;

        inline SIMDVec_f(__m256 const & x) {
            this->mVec = x; // TODO: should this be replaced with mov?
        }

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() {}

        // UTOF
        inline explicit SIMDVec_f(SIMDVec_u<uint32_t, 8> const & uintVec) {
            for (int i = 0; i < 8; i++) this->insert(i, (float)uintVec[i]);
        }

        // ITOF
        inline explicit SIMDVec_f(SIMDVec_i<int32_t, 8> const & intVec) {
            for (int i = 0; i < 8; i++) this->insert(i, (float)intVec[i]);
        }

        // SET-CONSTR
        inline explicit SIMDVec_f(float f) {
            mVec = _mm256_set1_ps(f);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const * p) {
            mVec = _mm256_loadu_ps(p);
        }

        // FULL-CONSTR
        inline SIMDVec_f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
            mVec = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
        }

        // EXTRACT
        inline float extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            return raw[index];
        }

        // EXTRACT
        inline float operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_ps(raw);
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
        inline SIMDVec_f & load(float const * p) {
            mVec = _mm256_loadu_ps(p);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVec = _mm256_load_ps(p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<8> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) const {
            _mm256_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) const {
            _mm256_store_ps(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDVecMask<8> const & mask, float* p) const {
            _mm256_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVec_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // ADDS     - Add with scalarr
        inline SIMDVec_f add(float b) const {
            return SIMDVec_f(_mm256_add_ps(mVec, _mm256_set1_ps(b)));
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_add_ps(mVec, _mm256_set1_ps(b));
            return SIMDVec_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_ps(mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            mVec = _mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            mVec = _mm256_add_ps(mVec, _mm256_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_add_ps(mVec, _mm256_set1_ps(b));
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
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            return SIMDVec_f(_mm256_mul_ps(mVec, b.mVec));
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            return SIMDVec_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b)  const {
            return SIMDVec_f(_mm256_mul_ps(mVec, _mm256_set1_ps(b)));
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_mul_ps(mVec, _mm256_set1_ps(b));
            return SIMDVec_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
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
        inline SIMDVec_f rcp() const {
            return SIMDVec_f(_mm256_rcp_ps(mVec));
        }
        // MRCP   - Masked reciprocal
        inline SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_rcp_ps(mVec);
            return SIMDVec_f(_mm256_blendv_ps(mVec, t0, _mm256_castsi256_ps(mask.mMask)));
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVec_f rcp(float b) {
            __m256 t0 = _mm256_rcp_ps(mVec);
            return SIMDVec_f(_mm256_mul_ps(t0, _mm256_set1_ps(b)));
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVec_f rcp(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_rcp_ps(mVec);
            __m256 t1 = _mm256_mul_ps(t0, _mm256_set1_ps(b));
            return SIMDVec_f(_mm256_blendv_ps(mVec, t1, _mm256_castsi256_ps(mask.mMask)));
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

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

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
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & a, SIMDVec_f const & b) {
#ifdef FMA
            return _mm256_fmadd_ps(this->mVec, a.mVec, b.mVec);
#else
            return _mm256_add_ps(b.mVec, _mm256_mul_ps(this->mVec, a.mVec));
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & a, SIMDVec_f const & b) {
            __m256 temp = _mm256_add_ps(_mm256_mul_ps(this->mVec, a.mVec), b.mVec);
            return _mm256_blendv_ps(this->mVec, temp, _mm256_cvtepi32_ps(mask.mMask));
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

        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign

        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign

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
        SIMDVec_f sqrt() {
            return SIMDVec_f(_mm256_sqrt_ps(mVec));
        }
        // MSQRT     - Masked square root of vector values 
        SIMDVec_f sqrt(SIMDVecMask<8> const & mask) {
            __m256 mask_ps = _mm256_castsi256_ps(mask.mMask);
            __m256 ret = _mm256_sqrt_ps(mVec);
            return SIMDVec_f(_mm256_blendv_ps(mVec, ret, mask_ps));
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
        SIMDVec_i<int32_t, 8> trunc() {
            __m256i t0 = _mm256_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 8>(t0);
        }
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        SIMDVec_i<int32_t, 8> trunc(SIMDVecMask<8> const & mask) {
            __m256 mask_ps = _mm256_castsi256_ps(mask.mMask);
            __m256 t0 = _mm256_blendv_ps(_mm256_setzero_ps(), mVec, mask_ps);
            __m256i t1 = _mm256_cvttps_epi32(t0);
            return SIMDVec_i<int32_t, 8>(t1);
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
        /*SIMDVec_f sin() {
        SIMDVec_f ret = UME::SIMD::genericSin<float,SIMDVec_f, SIMDVec_i<int32_t, 8>, SIMDMask8>(*this);
        return ret;
        }*/
        // MSIN      - Masked sine     
        //SIMDVec_f sin(SIMDMask8 const & mask) {
        //    SIMDVec_f ret; //= UME::SIMD::genericSin<float,SIMDVec_f, SIMDVec_i<int32_t, 8>, SIMDMask8>(*this);
        //    return ret;
        //}
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent

        // inline operator SIMDVec_u<uint32_t, 8> const () ;
        // inline operator SIMDVec_i<int32_t, 8> const () ;
    };

    template<>
    class SIMDVec_f<float, 16> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_u<uint32_t, 16>,
            SIMDVec_i<int32_t, 16>,
            float,
            16,
            uint32_t,
            SIMDVecMask<16>,
            SIMDVecSwizzle<16>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_f<float, 8 >>
    {
    private:
        __m256 mVecLo, mVecHi;

        inline SIMDVec_f(__m256 const & lo, __m256 const & hi) {
            this->mVecLo = lo;
            this->mVecHi = hi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(float f) {
            mVecLo = _mm256_set1_ps(f);
            mVecHi = _mm256_set1_ps(f);
        }

        // UTOF
        inline explicit SIMDVec_f(SIMDVec_u<uint32_t, 16> const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVec_f(SIMDVec_i<int32_t, 16>  const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const * p) {
            mVecLo = _mm256_loadu_ps(p);
            mVecHi = _mm256_loadu_ps(p + 8);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11,
            float f12, float f13, float f14, float f15) {
            mVecLo = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
            mVecHi = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
        }

        // EXTRACT - Extract single element from a vector
        inline float extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
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
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
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
        inline SIMDVec_f & load(float const * p) {
            mVecLo = _mm256_loadu_ps(p);
            mVecHi = _mm256_loadu_ps(p + 8);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVecLo = _mm256_load_ps(p);
            mVecHi = _mm256_load_ps(p + 8);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<16> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t1 = _mm256_load_ps(p + 8);
            mVecHi = _mm256_blendv_ps(mVecHi, t0, _mm256_castsi256_ps(mask.mMaskHi));
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) {
            _mm256_storeu_ps(p, mVecLo);
            _mm256_storeu_ps(p + 8, mVecHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) {
            _mm256_store_ps(p, mVecLo);
            _mm256_store_ps(p + 8, mVecHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDVecMask<16> const & mask, float* p) {
            _mm256_maskstore_ps(p, mask.mMaskLo, mVecLo);
            _mm256_maskstore_ps(p + 8, mask.mMaskHi, mVecHi);
            return p;
        }
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_add_ps(this->mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) {
            return this->add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t2 = _mm256_add_ps(this->mVecHi, b.mVecHi);
            __m256 t3 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVec_f(t1, t3);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(this->mVecLo, t0);
            __m256 t2 = _mm256_add_ps(this->mVecHi, t0);
            return SIMDVec_f(t1, t2);
        }
        inline SIMDVec_f operator+ (float b) {
            return this->add(b);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLo, t0);
            __m256 t2 = _mm256_add_ps(mVecHi, t0);
            __m256 t3 = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t4 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVec_f(t3, t4);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVecLo = _mm256_add_ps(mVecLo, b.mVecLo);
            mVecHi = _mm256_add_ps(mVecHi, b.mVecHi);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return this->adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVecLo, b.mVecLo);
            mVecLo = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t1 = _mm256_add_ps(mVecHi, b.mVecHi);
            mVecHi = _mm256_blendv_ps(mVecHi, t1, _mm256_castsi256_ps(mask.mMaskHi));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVecLo = _mm256_add_ps(mVecLo, t0);
            mVecHi = _mm256_add_ps(mVecHi, t0);
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return this->adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLo, t0);
            __m256 t2 = _mm256_add_ps(mVecHi, t0);
            mVecLo = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            mVecHi = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
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
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_mul_ps(this->mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(this->mVecLo, b.mVecLo);
            __m256 t1 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t2 = _mm256_mul_ps(this->mVecHi, b.mVecHi);
            __m256 t3 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVec_f(t1, t3);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(this->mVecLo, t0);
            __m256 t2 = _mm256_mul_ps(this->mVecHi, t0);
            return SIMDVec_f(t1, t2);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVecLo, t0);
            __m256 t2 = _mm256_mul_ps(mVecHi, t0);
            __m256 t3 = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t4 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVec_f(t3, t4);
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
        inline SIMDVec_f rcp() {
            __m256 t0 = _mm256_rcp_ps(this->mVecLo);
            __m256 t1 = _mm256_rcp_ps(this->mVecHi);
            return SIMDVec_f(t0, t1);
        }
        // MRCP   - Masked reciprocal
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask) {
            __m256 t0 = _mm256_rcp_ps(this->mVecLo);
            __m256 t1 = _mm256_blendv_ps(mVecLo, t0, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t2 = _mm256_rcp_ps(this->mVecHi);
            __m256 t3 = _mm256_blendv_ps(mVecHi, t2, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVec_f(t1, t3);
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVec_f rcp(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(t0, this->mVecLo);
            __m256 t2 = _mm256_div_ps(t0, this->mVecHi);
            return SIMDVec_f(t1, t2);
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(t0, mVecLo);
            __m256 t2 = _mm256_blendv_ps(mVecLo, t1, _mm256_castsi256_ps(mask.mMaskLo));
            __m256 t3 = _mm256_div_ps(t0, mVecHi);
            __m256 t4 = _mm256_blendv_ps(mVecHi, t3, _mm256_castsi256_ps(mask.mMaskHi));
            return SIMDVec_f(t2, t4);
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

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

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
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & a, SIMDVec_f const & b) {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
            __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
#else
            __m256 t0 = _mm256_add_ps(b.mVecLo, _mm256_mul_ps(this->mVecLo, a.mVecLo));
            __m256 t1 = _mm256_add_ps(b.mVecHi, _mm256_mul_ps(this->mVecHi, a.mVecHi));
            return SIMDVec_f(t0, t1);
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & a, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVecLo, a.mVecLo), b.mVecLo);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVecHi, a.mVecHi), b.mVecHi);
            __m256 t2 = _mm256_blendv_ps(mVecLo, t0, _mm256_cvtepi32_ps(mask.mMaskLo));
            __m256 t3 = _mm256_blendv_ps(mVecHi, t1, _mm256_cvtepi32_ps(mask.mMaskHi));
            return SIMDVec_f(t2, t3);
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

        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign

        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign

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
        inline SIMDVec_i<int32_t, 16> trunc() const {
            __m256i t0 = _mm256_cvtps_epi32(_mm256_round_ps(mVecLo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
            __m256i t1 = _mm256_cvtps_epi32(_mm256_round_ps(mVecHi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

            return SIMDVec_i<int32_t, 16>(t0, t1);
        }
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        inline SIMDVec_i<int32_t, 16> trunc(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_round_ps(mVecLo, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            __m256 t1 = _mm256_round_ps(mVecHi, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            __m256 t2 = _mm256_setzero_ps();
            __m256 t3 = _mm256_blendv_ps(t2, t0, _mm256_cvtepi32_ps(mask.mMaskLo));
            __m256 t4 = _mm256_blendv_ps(t2, t1, _mm256_cvtepi32_ps(mask.mMaskHi));

            __m256i t5 = _mm256_cvtps_epi32(t3);
            __m256i t6 = _mm256_cvtps_epi32(t4);
            return SIMDVec_i<int32_t, 16>(t5, t6);
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
    class SIMDVec_f<float, 32> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_u<uint32_t, 32>,
            SIMDVec_i<int32_t, 32>,
            float,
            32,
            uint32_t,
            SIMDVecMask<32>,
            SIMDVecSwizzle<32>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_f<float, 16 >>
    {
    private:
        __m256 mVecLoLo, mVecLoHi, mVecHiLo, mVecHiHi;

        inline SIMDVec_f(__m256 const & xLoLo,
            __m256 const & xLoHi,
            __m256 const & xHiLo,
            __m256 const & xHiHi) {
            this->mVecLoLo = xLoLo;
            this->mVecLoHi = xLoHi;
            this->mVecHiLo = xHiLo;
            this->mVecHiHi = xHiHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(float f) {
            mVecLoLo = _mm256_set1_ps(f);
            mVecLoHi = _mm256_set1_ps(f);
            mVecHiLo = _mm256_set1_ps(f);
            mVecHiHi = _mm256_set1_ps(f);
        }

        // UTOF
        inline explicit SIMDVec_f(SIMDVec_u<uint32_t, 32> const & vecUint) {

        }

        // ITOF
        inline explicit SIMDVec_f(SIMDVec_i<int32_t, 32>  const & vecInt) {

        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(float const * p) {
            mVecLoLo = _mm256_loadu_ps(p);
            mVecLoHi = _mm256_loadu_ps(p + 8);
            mVecHiLo = _mm256_loadu_ps(p + 16);
            mVecHiHi = _mm256_loadu_ps(p + 24);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
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
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVecLoLo);
                return raw[index];
            }
            else if (index < 16) {
                _mm256_store_ps(raw, mVecLoHi);
                return raw[index - 8];
            }
            else if (index <24) {
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
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, float value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
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

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV     - Assignment with another vector
        // MASSIGNV    - Masked assignment with another vector
        // ASSIGNS     - Assignment with scalar
        // MASSIGNS    - Masked assign with scalar

        // PREFETCH0  
        static inline void prefetch0(float *p) {
            _mm_prefetch((const char *)p, _MM_HINT_T0);
        }

        // PREFETCH1
        static inline void prefetch1(float *p) {
            _mm_prefetch((const char *)p, _MM_HINT_T1);
        }

        // PREFETCH2
        static inline void prefetch2(float *p) {
            _mm_prefetch((const char *)p, _MM_HINT_T2);
        }

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVec_f & load(float const * p) {
            mVecLoLo = _mm256_loadu_ps(p);
            mVecLoHi = _mm256_loadu_ps(p + 8);
            mVecHiLo = _mm256_loadu_ps(p + 16);
            mVecHiHi = _mm256_loadu_ps(p + 24);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVecLoLo = _mm256_load_ps(p);
            mVecLoHi = _mm256_load_ps(p + 8);
            mVecHiLo = _mm256_load_ps(p + 16);
            mVecHiHi = _mm256_load_ps(p + 24);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float* p) {
            _mm256_storeu_ps(p, mVecLoLo);
            _mm256_storeu_ps(p + 8, mVecLoHi);
            _mm256_storeu_ps(p + 16, mVecHiLo);
            _mm256_storeu_ps(p + 24, mVecHiHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float* p) const {
            _mm256_store_ps(p, mVecLoLo);
            _mm256_store_ps(p + 8, mVecLoHi);
            _mm256_store_ps(p + 16, mVecHiLo);
            _mm256_store_ps(p + 24, mVecHiHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        // ADDV     - Add with vector 
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVec_f(t4, t5, t6, t7);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_add_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_add_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_add_ps(mVecHiHi, t0);
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_add_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_add_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_add_ps(mVecHiHi, t0);
            __m256 t5 = _mm256_blendv_ps(mVecLoLo, t1, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t6 = _mm256_blendv_ps(mVecLoHi, t2, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t7 = _mm256_blendv_ps(mVecHiLo, t3, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t8 = _mm256_blendv_ps(mVecHiHi, t4, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVec_f(t5, t6, t7, t8);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            this->mVecLoLo = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            this->mVecLoHi = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            this->mVecHiLo = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            this->mVecHiHi = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(mVecHiHi, b.mVecHiHi);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            this->mVecLoLo = _mm256_add_ps(mVecLoLo, t0);
            this->mVecLoHi = _mm256_add_ps(mVecLoHi, t0);
            this->mVecHiLo = _mm256_add_ps(mVecHiLo, t0);
            this->mVecHiHi = _mm256_add_ps(mVecHiHi, t0);
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_add_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_add_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_add_ps(mVecHiHi, t0);
            mVecLoLo = _mm256_blendv_ps(mVecLoLo, t1, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            mVecLoHi = _mm256_blendv_ps(mVecLoHi, t2, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            mVecHiLo = _mm256_blendv_ps(mVecHiLo, t3, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            mVecHiHi = _mm256_blendv_ps(mVecHiHi, t4, _mm256_cvtepi32_ps(mask.mMaskHiHi));
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
        inline SIMDVec_f mul(SIMDVec_f const & b) {
            __m256 t0 = _mm256_mul_ps(this->mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_mul_ps(this->mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_mul_ps(this->mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_mul_ps(this->mVecHiHi, b.mVecHiHi);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_mul_ps(mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_mul_ps(mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_mul_ps(mVecHiHi, b.mVecHiHi);
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVec_f(t4, t5, t6, t7);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(this->mVecLoLo, t0);
            __m256 t2 = _mm256_mul_ps(this->mVecLoHi, t0);
            __m256 t3 = _mm256_mul_ps(this->mVecHiLo, t0);
            __m256 t4 = _mm256_mul_ps(this->mVecHiHi, t0);
            return SIMDVec_f(t1, t2, t3, t4);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVecLoLo, t0);
            __m256 t2 = _mm256_mul_ps(mVecLoHi, t0);
            __m256 t3 = _mm256_mul_ps(mVecHiLo, t0);
            __m256 t4 = _mm256_mul_ps(mVecHiHi, t0);
            __m256 t5 = _mm256_blendv_ps(mVecLoLo, t1, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t6 = _mm256_blendv_ps(mVecLoHi, t2, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t7 = _mm256_blendv_ps(mVecHiLo, t3, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t8 = _mm256_blendv_ps(mVecHiHi, t4, _mm256_cvtepi32_ps(mask.mMaskHiHi));
            return SIMDVec_f(t5, t6, t7, t8);
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

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

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
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVec_f const & a, SIMDVec_f const & b) {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
            __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
#else
            __m256 t0 = _mm256_add_ps(b.mVecLoLo, _mm256_mul_ps(this->mVecLoLo, a.mVecLoLo));
            __m256 t1 = _mm256_add_ps(b.mVecLoHi, _mm256_mul_ps(this->mVecLoHi, a.mVecLoHi));
            __m256 t2 = _mm256_add_ps(b.mVecHiLo, _mm256_mul_ps(this->mVecHiLo, a.mVecHiLo));
            __m256 t3 = _mm256_add_ps(b.mVecHiHi, _mm256_mul_ps(this->mVecHiHi, a.mVecHiHi));
            return SIMDVec_f(t0, t1, t2, t3);
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<32> const & mask, SIMDVec_f const & a, SIMDVec_f const & b) {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(mVecLoLo, a.mVecLoLo, b.mVecLoLo);
            __m256 t1 = _mm256_fmadd_ps(mVecLoHi, a.mVecLoHi, b.mVecLoHi);
            __m256 t2 = _mm256_fmadd_ps(mVecHiLo, a.mVecHiLo, b.mVecHiLo);
            __m256 t3 = _mm256_fmadd_ps(mVecHiHi, a.mVecHiHi, b.mVecHiHi);
#else
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVecLoLo, a.mVecLoLo), b.mVecLoLo);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVecLoHi, a.mVecLoHi), b.mVecLoHi);
            __m256 t2 = _mm256_add_ps(_mm256_mul_ps(mVecHiLo, a.mVecHiLo), b.mVecHiLo);
            __m256 t3 = _mm256_add_ps(_mm256_mul_ps(mVecHiHi, a.mVecHiHi), b.mVecHiHi);
#endif
            __m256 t4 = _mm256_blendv_ps(mVecLoLo, t0, _mm256_cvtepi32_ps(mask.mMaskLoLo));
            __m256 t5 = _mm256_blendv_ps(mVecLoHi, t1, _mm256_cvtepi32_ps(mask.mMaskLoHi));
            __m256 t6 = _mm256_blendv_ps(mVecHiLo, t2, _mm256_cvtepi32_ps(mask.mMaskHiLo));
            __m256 t7 = _mm256_blendv_ps(mVecHiHi, t3, _mm256_cvtepi32_ps(mask.mMaskHiHi));

            return SIMDVec_f(t4, t5, t6, t7);
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
    class SIMDVec_f<double, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 4>,
            SIMDVec_u<uint64_t, 4>,
            SIMDVec_i<int64_t, 4>,
            double,
            4,
            uint64_t,
            SIMDVecMask<4>, // Using non-standard mask!
            SIMDVecSwizzle<4>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2 >>
    {
    private:
        __m256d mVec;

        inline SIMDVec_f(__m256d const & x) {
            this->mVec = x;
        }

    public:

        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(double d) {
            mVec = _mm256_set1_pd(d);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(double const * d) {
            mVec = _mm256_loadu_pd(d);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_setr_pd(d0, d1, d2, d3);
        }

        // EXTRACT - Extract single element from a vector
        inline double extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[index];
        }

        // EXTRACT - Extract single element from a vector
        inline double operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_pd(raw);
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
        inline SIMDVec_f & load(double const * p) {
            mVec = _mm256_loadu_pd(p);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(double const * p) {
            mVec = _mm256_load_pd(p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d mask_pd = _mm256_cvtepi32_pd(mask.mMask);
            mVec = _mm256_blendv_pd(mVec, t0, mask_pd);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline double* storea(double* p) {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline double* storea(SIMDVecMask<4> const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;
            x.pd = _mm256_cvtepi32_pd(mask.mMask);

            _mm256_maskstore_pd(p, x.epi64, mVec);
            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        // MADDV    - Masked add with vector
        // ADDS     - Add with scalar
        // MADDS    - Masked add with scalar
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_pd(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(double b) {
            mVec = _mm256_add_pd(this->mVec, _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
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

    template<>
    class SIMDVec_f<double, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 8>,
            SIMDVec_u<uint64_t, 8>,
            SIMDVec_i<int64_t, 8>,
            double,
            8,
            uint64_t,
            SIMDVecMask<8>, // Using non-standard mask!
            SIMDVecSwizzle<4>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 8>,
            SIMDVec_f<double, 4 >>
    {
    private:
        __m256d mVecLo;
        __m256d mVecHi;

        inline SIMDVec_f(__m256d const & xLo, __m256d const & xHi) {
            this->mVecLo = xLo;
            this->mVecHi = xHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(const double *p) { this->load(p); }

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(double d) {
            mVecLo = _mm256_set1_pd(d);
            mVecHi = _mm256_set1_pd(d);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(double d0, double d1, double d2, double d3,
            double d4, double d5, double d6, double d7) {
            mVecLo = _mm256_setr_pd(d0, d1, d2, d3);
            mVecHi = _mm256_setr_pd(d4, d5, d6, d7);
        }

        // EXTRACT - Extract single element from a vector
        inline double extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];

            if (index < 4) {
                _mm256_store_pd(raw, mVecLo);
                return raw[index];
            }
            else {
                _mm256_store_pd(raw, mVecHi);
                return raw[index - 4];
            }
        }

        // EXTRACT - Extract single element from a vector
        inline double operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            if (index < 4) {
                _mm256_store_pd(raw, mVecLo);
                raw[index] = value;
                mVecLo = _mm256_load_pd(raw);
            }
            else {
                _mm256_store_pd(raw, mVecHi);
                raw[index - 4] = value;
                mVecHi = _mm256_load_pd(raw);
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
        inline SIMDVec_f & load(double const * p) {
            mVecLo = _mm256_loadu_pd(p);
            mVecHi = _mm256_loadu_pd(p + 4);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(double const * p) {
            mVecLo = _mm256_load_pd(p);
            mVecHi = _mm256_load_pd(p + 4);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<8> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d t1 = _mm256_load_pd(p + 4);

            __m128i t2 = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask, 1);

            __m256d mask_pd_lo = _mm256_cvtepi32_pd(t2);
            __m256d mask_pd_hi = _mm256_cvtepi32_pd(t3);
            mVecLo = _mm256_blendv_pd(mVecLo, t0, mask_pd_lo);
            mVecHi = _mm256_blendv_pd(mVecHi, t1, mask_pd_hi);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_storeu_pd(p, mVecLo);
            _mm256_storeu_pd(p + 4, mVecHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline double* storea(double* p) {
            _mm256_store_pd(p, mVecLo);
            _mm256_store_pd(p + 4, mVecHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline double* storea(SIMDVecMask<8> const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;

            __m128i t0 = _mm256_extractf128_si256(mask.mMask, 0);
            x.pd = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p, x.epi64, mVecLo);

            __m128i t1 = _mm256_extractf128_si256(mask.mMask, 1);
            x.pd = _mm256_cvtepi32_pd(t1);
            _mm256_maskstore_pd(p + 4, x.epi64, mVecHi);

            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        // MADDV    - Masked add with vector
        // ADDS     - Add with scalar
        // MADDS    - Masked add with scalar
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVecLo = _mm256_add_pd(this->mVecLo, b.mVecLo);
            mVecHi = _mm256_add_pd(this->mVecHi, b.mVecHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(double b) {
            mVecLo = _mm256_add_pd(this->mVecLo, _mm256_set1_pd(b));
            mVecHi = _mm256_add_pd(this->mVecHi, _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
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

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

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
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

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

    template<>
    class SIMDVec_f<double, 16> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 16>,
            SIMDVec_u<uint64_t, 16>,
            SIMDVec_i<int64_t, 16>,
            double,
            16,
            uint64_t,
            SIMDVecMask<16>, // Using non-standard mask!
            SIMDVecSwizzle<16>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 16>,
            SIMDVec_f<double, 8 >>
    {
    private:
        __m256d mVecLoLo;
        __m256d mVecLoHi;
        __m256d mVecHiLo;
        __m256d mVecHiHi;

        inline SIMDVec_f(__m256d const & xLoLo, __m256d const & xLoHi,
            __m256d const & xHiLo, __m256d const & xHiHi) {
            this->mVecLoLo = xLoLo;
            this->mVecLoHi = xLoHi;
            this->mVecHiLo = xHiLo;
            this->mVecHiHi = xHiHi;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline explicit SIMDVec_f(double d) {
            mVecLoLo = _mm256_set1_pd(d);
            mVecLoHi = _mm256_set1_pd(d);
            mVecHiLo = _mm256_set1_pd(d);
            mVecHiHi = _mm256_set1_pd(d);
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVec_f(const double* d) {
            mVecLoLo = _mm256_loadu_pd(d);
            mVecLoHi = _mm256_loadu_pd(d + 4);
            mVecHiLo = _mm256_loadu_pd(d + 8);
            mVecHiHi = _mm256_loadu_pd(d + 12);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(double d0, double d1, double d2, double d3,
            double d4, double d5, double d6, double d7,
            double d8, double d9, double d10, double d11,
            double d12, double d13, double d14, double d15) {
            mVecLoLo = _mm256_setr_pd(d0, d1, d2, d3);
            mVecLoHi = _mm256_setr_pd(d4, d5, d6, d7);
            mVecHiLo = _mm256_setr_pd(d8, d9, d10, d11);
            mVecHiHi = _mm256_setr_pd(d12, d13, d14, d15);
        }

        // EXTRACT - Extract single element from a vector
        inline double extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];

            if (index < 4) {
                _mm256_store_pd(raw, mVecLoLo);
                return raw[index];
            }
            else if (index < 8) {
                _mm256_store_pd(raw, mVecLoHi);
                return raw[index - 4];
            }
            else if (index < 12) {
                _mm256_store_pd(raw, mVecHiLo);
                return raw[index - 8];
            }
            else {
                _mm256_store_pd(raw, mVecHiHi);
                return raw[index - 12];
            }
        }

        // EXTRACT - Extract single element from a vector
        inline double operator[] (uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_f, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }

        // INSERT  - Insert single element into a vector
        inline SIMDVec_f & insert(uint32_t index, double value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) double raw[4];
            if (index < 4) {
                _mm256_store_pd(raw, mVecLoLo);
                raw[index] = value;
                mVecLoLo = _mm256_load_pd(raw);
            }
            else if (index < 8) {
                _mm256_store_pd(raw, mVecLoHi);
                raw[index - 4] = value;
                mVecLoHi = _mm256_load_pd(raw);
            }
            else if (index < 12) {
                _mm256_store_pd(raw, mVecHiLo);
                raw[index - 8] = value;
                mVecHiLo = _mm256_load_pd(raw);
            }
            else {
                _mm256_store_pd(raw, mVecHiHi);
                raw[index - 12] = value;
                mVecHiHi = _mm256_load_pd(raw);
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
        inline SIMDVec_f & load(double const * p) {
            mVecLoLo = _mm256_load_pd(p);
            mVecLoHi = _mm256_load_pd(p + 4);
            mVecHiLo = _mm256_load_pd(p + 8);
            mVecHiHi = _mm256_load_pd(p + 12);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //           vector
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(double const * p) {
            mVecLoLo = _mm256_load_pd(p);
            mVecLoHi = _mm256_load_pd(p + 4);
            mVecHiLo = _mm256_load_pd(p + 8);
            mVecHiHi = _mm256_load_pd(p + 12);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<16> const & mask, double const * p) {
            __m256d t0 = _mm256_load_pd(p);
            __m256d t1 = _mm256_load_pd(p + 4);
            __m256d t2 = _mm256_load_pd(p + 8);
            __m256d t3 = _mm256_load_pd(p + 12);

            __m128i t4 = _mm256_extractf128_si256(mask.mMaskLo, 0);
            __m128i t5 = _mm256_extractf128_si256(mask.mMaskLo, 1);
            __m256d mask_pd_lo = _mm256_cvtepi32_pd(t4);
            __m256d mask_pd_hi = _mm256_cvtepi32_pd(t5);
            mVecLoLo = _mm256_blendv_pd(mVecLoLo, t0, mask_pd_lo);
            mVecLoHi = _mm256_blendv_pd(mVecLoHi, t1, mask_pd_hi);

            t4 = _mm256_extractf128_si256(mask.mMaskHi, 0);
            t5 = _mm256_extractf128_si256(mask.mMaskHi, 1);
            mask_pd_lo = _mm256_cvtepi32_pd(t4);
            mask_pd_hi = _mm256_cvtepi32_pd(t5);
            mVecHiLo = _mm256_blendv_pd(mVecLoLo, t2, mask_pd_lo);
            mVecHiHi = _mm256_blendv_pd(mVecLoHi, t3, mask_pd_hi);

            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline double* store(double* p) {
            _mm256_store_pd(p, mVecLoLo);
            _mm256_store_pd(p + 4, mVecLoHi);
            _mm256_store_pd(p + 8, mVecHiLo);
            _mm256_store_pd(p + 12, mVecHiHi);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        // STOREA  - Store vector content into aligned memory
        inline double* storea(double* p) {
            _mm256_store_pd(p, mVecLoLo);
            _mm256_store_pd(p + 4, mVecLoHi);
            _mm256_store_pd(p + 8, mVecHiLo);
            _mm256_store_pd(p + 12, mVecHiHi);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline double* storea(SIMDVecMask<16> const & mask, double* p) {
            union {
                __m256d pd;
                __m256i epi64;
            }x;

            __m128i t0 = _mm256_extractf128_si256(mask.mMaskLo, 0);
            x.pd = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p, x.epi64, mVecLoLo);

            t0 = _mm256_extractf128_si256(mask.mMaskLo, 1);
            x.pd = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p + 4, x.epi64, mVecLoHi);

            t0 = _mm256_extractf128_si256(mask.mMaskHi, 0);
            x.pd = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p + 8, x.epi64, mVecHiLo);

            t0 = _mm256_extractf128_si256(mask.mMaskHi, 1);
            x.pd = _mm256_cvtepi32_pd(t0);
            _mm256_maskstore_pd(p + 12, x.epi64, mVecHiHi);

            return p;
        }
        //(Addition operations)
        // ADDV     - Add with vector 
        // MADDV    - Masked add with vector
        // ADDS     - Add with scalar
        // MADDS    - Masked add with scalar
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVecLoLo = _mm256_add_pd(this->mVecLoLo, b.mVecLoLo);
            mVecLoHi = _mm256_add_pd(this->mVecLoHi, b.mVecLoHi);
            mVecHiLo = _mm256_add_pd(this->mVecHiLo, b.mVecHiLo);
            mVecHiHi = _mm256_add_pd(this->mVecHiHi, b.mVecHiHi);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(double b) {
            mVecLoLo = _mm256_add_pd(this->mVecLoLo, _mm256_set1_pd(b));
            mVecLoHi = _mm256_add_pd(this->mVecLoHi, _mm256_set1_pd(b));
            mVecHiLo = _mm256_add_pd(this->mVecHiLo, _mm256_set1_pd(b));
            mVecHiHi = _mm256_add_pd(this->mVecHiHi, _mm256_set1_pd(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
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
