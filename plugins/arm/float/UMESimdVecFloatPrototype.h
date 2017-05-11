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

#ifndef UME_SIMD_VEC_FLOAT_PROTOTYPE_H_
#define UME_SIMD_VEC_FLOAT_PROTOTYPE_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#include "../UMESimdMask.h"
#include "../UMESimdSwizzle.h"
#include "../UMESimdVecUint.h"

namespace UME {
namespace SIMD {

    template<typename SCALAR_FLOAT_TYPE, uint32_t VEC_LEN>
    struct SIMDVec_f_traits {
        // Generic trait class not containing type definition so that only correct explicit
        // type definitions are compiled correctly
    };

    // 32b vectors
    template<>
    struct SIMDVec_f_traits<float, 1> {
        typedef NullType<1>             HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint32_t, 1>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 1>   VEC_INT_TYPE;
        typedef int32_t                 SCALAR_INT_TYPE;
        typedef uint32_t                SCALAR_UINT_TYPE;
        typedef float*                  SCALAR_TYPE_PTR;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDSwizzle<1>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_LOWER_PRECISION;
        typedef double                  SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<2>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_LOWER_PRECISION;
        typedef double                  SCALAR_FLOAT_HIGHER_PRECISION;
    };

    template<>
    struct SIMDVec_f_traits<double, 1> {
        typedef NullType<1>             HALF_LEN_VEC_TYPE;
        typedef SIMDVec_u<uint64_t, 1>  VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 1>   VEC_INT_TYPE;
        typedef int64_t                 SCALAR_INT_TYPE;
        typedef uint64_t                SCALAR_UINT_TYPE;
        typedef double*                 SCALAR_TYPE_PTR;
        typedef SIMDVecMask<1>          MASK_TYPE;
        typedef SIMDSwizzle<1>          SWIZZLE_MASK_TYPE;
        typedef float                   SCALAR_FLOAT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<4>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_LOWER_PRECISION;
        typedef double                  SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<2>          SWIZZLE_MASK_TYPE;
        typedef float                   SCALAR_FLOAT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<8>          SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_LOWER_PRECISION;
        typedef double                  SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<4>          SWIZZLE_MASK_TYPE;
        typedef float                   SCALAR_FLOAT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<16>         SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_LOWER_PRECISION;
        typedef double                  SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<8>          SWIZZLE_MASK_TYPE;
        typedef float                   SCALAR_FLOAT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<32>         SWIZZLE_MASK_TYPE;
        typedef NullType<2>             SCALAR_FLOAT_LOWER_PRECISION;
        typedef NullType<3>             SCALAR_FLOAT_HIGHER_PRECISION;
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
        typedef SIMDSwizzle<16>         SWIZZLE_MASK_TYPE;
        typedef float                   SCALAR_FLOAT_LOWER_PRECISION;
        typedef NullType<2>             SCALAR_FLOAT_HIGHER_PRECISION;
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
    class SIMDVec_f :
        public SIMDVecFloatInterface<
            SIMDVec_f<SCALAR_FLOAT_TYPE, VEC_LEN>,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_UINT_TYPE,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::VEC_INT_TYPE,
            SCALAR_FLOAT_TYPE,
            VEC_LEN,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_UINT_TYPE,
            typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_INT_TYPE,
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

        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_FLOAT_LOWER_PRECISION  SCALAR_FLOAT_LOWER_PRECISION;
        typedef typename SIMDVec_f_traits<SCALAR_FLOAT_TYPE, VEC_LEN>::SCALAR_FLOAT_HIGHER_PRECISION SCALAR_FLOAT_HIGHER_PRECISION;

    private:
        VEC_EMU_REG mVec;

    public:
        constexpr static uint32_t alignment() {
            return VEC_LEN*sizeof(SCALAR_FLOAT_TYPE) > 16 ? 16 : VEC_LEN*sizeof(SCALAR_FLOAT_TYPE);
        }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(SCALAR_FLOAT_TYPE f) : mVec(f) {};
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, SCALAR_FLOAT_TYPE>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<SCALAR_FLOAT_TYPE>(i)) {}

        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); }

        UME_FORCE_INLINE SIMDVec_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec.insert(0, f0); mVec.insert(1, f1);
        }

        UME_FORCE_INLINE SIMDVec_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3) {
            mVec.insert(0, f0);  mVec.insert(1, f1);  mVec.insert(2, f2);  mVec.insert(3, f3);
        }

        UME_FORCE_INLINE SIMDVec_f(
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

        UME_FORCE_INLINE SIMDVec_f(
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

        UME_FORCE_INLINE SIMDVec_f(
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

        // EXTRACT
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec.insert(index, value);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, SCALAR_FLOAT_TYPE> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, SCALAR_FLOAT_TYPE>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return this->assign(b);
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & operator= (SCALAR_FLOAT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS

        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<SCALAR_FLOAT_LOWER_PRECISION, VEC_LEN>() const;
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<SCALAR_FLOAT_HIGHER_PRECISION, VEC_LEN>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>() const;
    };

    // SIMD NullTypes. These are used whenever a terminating
    // scalar type is used as a creator function for SIMD type.
    // These types cannot be instantiated, but are necessary for 
    // typeset to be consistent.
    template<>
    class SIMDVec_f<NullType<1>, 1>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 2>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 4>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 8>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 16>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 32>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 64>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<1>, 128>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 1>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 2>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 4>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 8>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 16>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 32>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 64>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<2>, 128>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 1>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 2>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 4>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 8>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 16>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 32>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 64>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };

    template<>
    class SIMDVec_f<NullType<3>, 128>
    {
    private:
        SIMDVec_f() {}
        ~SIMDVec_f() {}
    };
}
}

#endif
