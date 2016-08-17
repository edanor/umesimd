// The MIT License (MIT)
//
// Copyright (c) 2016 CERN
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


    public:
        constexpr static uint32_t alignment() { return VEC_LEN*sizeof(SCALAR_FLOAT_TYPE); }

    private:
        alignas(alignment()) SCALAR_FLOAT_TYPE mVec[VEC_LEN];

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(SCALAR_FLOAT_TYPE f) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for (int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = f;
            }
        }

        // LOAD-CONSTR
        inline explicit SIMDVec_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); }

        inline SIMDVec_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec[0] = f0; 
            mVec[1] = f1;
        }

        inline SIMDVec_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3) {
            mVec[0] = f0; 
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
        }

        inline SIMDVec_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3,
            SCALAR_FLOAT_TYPE f4, SCALAR_FLOAT_TYPE f5,
            SCALAR_FLOAT_TYPE f6, SCALAR_FLOAT_TYPE f7)
        {
            mVec[0] = f0; 
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
            mVec[4] = f4; 
            mVec[5] = f5;
            mVec[6] = f6;
            mVec[7] = f7;
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
            mVec[0] = f0; 
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
            mVec[4] = f4; 
            mVec[5] = f5;
            mVec[6] = f6;
            mVec[7] = f7;
            mVec[8] = f8; 
            mVec[9] = f9;
            mVec[10] = f10;
            mVec[11] = f11;
            mVec[12] = f12; 
            mVec[13] = f13;
            mVec[14] = f14;
            mVec[15] = f15;
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
            mVec[0] = f0; 
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
            mVec[4] = f4; 
            mVec[5] = f5;
            mVec[6] = f6;
            mVec[7] = f7;
            mVec[8] = f8; 
            mVec[9] = f9;
            mVec[10] = f10;
            mVec[11] = f11;
            mVec[12] = f12; 
            mVec[13] = f13;
            mVec[14] = f14;
            mVec[15] = f15;
            mVec[16] = f16; 
            mVec[17] = f17;
            mVec[18] = f18;
            mVec[19] = f19;
            mVec[20] = f20; 
            mVec[21] = f21;
            mVec[22] = f22;
            mVec[23] = f23;
            mVec[24] = f24; 
            mVec[25] = f25;
            mVec[26] = f26;
            mVec[27] = f27;
            mVec[28] = f28; 
            mVec[29] = f29;
            mVec[30] = f30;
            mVec[31] = f31;
        }

        // EXTRACT
        inline SCALAR_FLOAT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        inline SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec[index] = value;
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, SCALAR_FLOAT_TYPE> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, SCALAR_FLOAT_TYPE>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE> operator[] (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, SCALAR_FLOAT_TYPE, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & src) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_src_ptr = &src.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & src) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_src_ptr = &src.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b;
            }
            return *this;
        }
        inline SIMDVec_f & operator= (SCALAR_FLOAT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = b;
            }
            return *this;
        }
        
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        
        // LOAD
        inline SIMDVec_f & load(SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        
        // STORE
        inline SCALAR_FLOAT_TYPE* store(SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTORE
        inline SCALAR_FLOAT_TYPE* store(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // STOREA
        inline SCALAR_FLOAT_TYPE* storea(SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTOREA
        inline SCALAR_FLOAT_TYPE* storea(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        
        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE *retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_b_ptr = &b.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = local_b_ptr[i];
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *retval_ptr = &retval.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = b;
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDS
        inline SIMDVec_f add(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + b;
            }
            return retval;
        }
        inline SIMDVec_f operator+ (SCALAR_FLOAT_TYPE b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] += b;
            }
            return *this;
        }
        inline SIMDVec_f & operator+= (SCALAR_FLOAT_TYPE b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
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
        inline SIMDVec_f postinc() {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]++;
            }
            return retval;
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<VEC_LEN> const & mask) {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]++;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                ++local_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) ++local_ptr[i];
            }
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBS
        inline SIMDVec_f sub(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - b;
            }
            return retval;
        }
        inline SIMDVec_f operator- (SCALAR_FLOAT_TYPE b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] -= b;
            }
            return *this;
        }
        inline SIMDVec_f & operator-= (SCALAR_FLOAT_TYPE b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
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
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b - local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b - local_ptr[i];
                else local_ptr[i] = b;
            }
            return *this;
        }
        
        
        // POSTDEC
        inline SIMDVec_f postdec() {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]--;
            }
            return retval;
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<VEC_LEN> const & mask) {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]--;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                --local_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) --local_ptr[i];
            }
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULS
        inline SIMDVec_f mul(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * b;
            }
            return retval;
        }
        inline SIMDVec_f operator* (SCALAR_FLOAT_TYPE b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] *= b;
            }
            return *this;
        }
        inline SIMDVec_f & operator*= (SCALAR_FLOAT_TYPE b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= b;
            }
            return *this;
        }        
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVS
        inline SIMDVec_f div(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / b;
            }
            return retval;
        }
        inline SIMDVec_f operator/ (SCALAR_FLOAT_TYPE b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] /= b;
            }
            return *this;
        }
        inline SIMDVec_f & operator/= (SCALAR_FLOAT_TYPE b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= b;
            }
            return *this;
        }
        
        // RCP
        inline SIMDVec_f rcp() const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
            }
            return retval;
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPS
        inline SIMDVec_f rcp(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b / local_ptr[i];
            }
            return retval;
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }

        // CMPEQV
        inline SIMDVecMask<VEC_LEN> cmpeq(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<VEC_LEN> cmpeq(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == b;
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator== (SCALAR_FLOAT_TYPE b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<VEC_LEN> cmpne(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<VEC_LEN> cmpne(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != b;
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator!= (SCALAR_FLOAT_TYPE b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<VEC_LEN> cmpgt(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<VEC_LEN> cmpgt(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > b;
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator> (SCALAR_FLOAT_TYPE b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<VEC_LEN> cmplt(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<VEC_LEN> cmplt(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < b;
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator< (SCALAR_FLOAT_TYPE b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<VEC_LEN> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<VEC_LEN> cmpge(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= b;
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator>= (SCALAR_FLOAT_TYPE b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<VEC_LEN> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= local_b_ptr[i];
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<VEC_LEN> cmple(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= b;
            }
            return retval;
        }
        inline SIMDVecMask<VEC_LEN> operator<= (SCALAR_FLOAT_TYPE b) const {
            return cmple(b);
        }
        
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool local_mask_ptr[VEC_LEN];
            bool retval = true;
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_mask_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            #pragma omp simd reduction(&&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval && local_mask_ptr[i];
            }
            return retval;
        }
        // CMPES
        inline bool cmpe(SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool local_mask_ptr[VEC_LEN];
            bool retval = true;
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_mask_ptr[i] = local_ptr[i] == b;
            }
            #pragma omp simd reduction(&&:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval && local_mask_ptr[i];
            }
            return retval;
        }
        // UNIQUE
        // TODO
        
        
        // HADD
        inline SCALAR_FLOAT_TYPE hadd() const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(0.0f);
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADD
        inline SCALAR_FLOAT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(0.0f);
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HADDS
        inline SCALAR_FLOAT_TYPE hadd(SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADDS
        inline SCALAR_FLOAT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HMUL
        inline SCALAR_FLOAT_TYPE hmul() const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(1.0f);
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;        }
        // MHMUL
        inline SCALAR_FLOAT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(1.0f);
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        // HMULS
        inline SCALAR_FLOAT_TYPE hmul(SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;
        }
        // MHMULS
        inline SCALAR_FLOAT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
            }
            return retval;
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
            }
            return retval;
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        
        
        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXS
        inline SIMDVec_f max(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] > b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = b;
            }
            return *this;
        }
        
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < b;
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_retval_ptr[i] = b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < local_b_ptr[i];
                bool cond = local_mask_ptr[i] && !predicate;
                if(cond) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
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
        

        // GATHERS
        inline SIMDVec_f & gather(SCALAR_FLOAT_TYPE * baseAddr, SCALAR_UINT_TYPE* indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(SCALAR_FLOAT_TYPE * baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // SCATTERS
        inline SCALAR_FLOAT_TYPE* scatter(SCALAR_FLOAT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERS
        inline SCALAR_FLOAT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // SCATTERV
        inline SCALAR_FLOAT_TYPE* scatter(SCALAR_FLOAT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERV
        inline SCALAR_FLOAT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) const {
            for(int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        
        // NEG
        inline SIMDVec_f neg() const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = -local_ptr[i];
            }
            return retval;
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = -local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // NEGA
        inline SIMDVec_f & nega() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        
        // ABS
        inline SIMDVec_f abs() const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] >= 0 ) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = -local_ptr[i];
            }
            return retval;
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < 0;
                bool cond = local_mask_ptr[i] && predicate;
                
                if(cond) local_retval_ptr[i] = -local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ABSA
        inline SIMDVec_f & absa() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < 0 ) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                bool predicate = local_ptr[i] < 0;
                bool cond = local_mask_ptr[i] && predicate;
                
                if(cond) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        // MSQR
        // SQRA
        // MSQRA
        // SQRT
        // MSQRT
        // SQRTA
        // MSQRTA
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        // MROUND
        // TRUNC
        inline SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> trunc() const {
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (SCALAR_INT_TYPE) local_ptr[i];
            }
            return retval;
        }
        // MTRUNC
        inline SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> trunc(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd simdlen(VEC_LEN) safelen(VEC_LEN)
            for(int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i]) local_retval_ptr[i] = (SCALAR_INT_TYPE) local_ptr[i];
                else local_retval_ptr[i] = 0;
            }
            return retval;
        }
        // FLOOR
        // MFLOOR
        // CEIL
        // MCEIL
        // ISFIN
        // ISINF
        // ISAN
        // ISNAN
        // ISSUB
        // ISZERO
        // ISZEROSUB
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        // PACKLO
        // PCAKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI
        
        // DEGRADE
        inline operator SIMDVec_f<SCALAR_FLOAT_LOWER_PRECISION, VEC_LEN>() const;
        // PROMOTE
        inline operator SIMDVec_f<SCALAR_FLOAT_HIGHER_PRECISION, VEC_LEN>() const;

        // FTOU
        inline operator SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN>() const;
        // FTOI
        inline operator SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN>() const;
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
