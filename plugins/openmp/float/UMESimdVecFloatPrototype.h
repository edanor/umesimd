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
        UME_FORCE_INLINE SIMDVec_f() : mVec() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(SCALAR_FLOAT_TYPE x) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for (unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = x;
            }
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_floating_point<T>::value && 
                                    !std::is_same<T, SCALAR_FLOAT_TYPE>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<SCALAR_FLOAT_TYPE>(i)) {}

        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(SCALAR_FLOAT_TYPE const * p) { this->load(p); }

        UME_FORCE_INLINE SIMDVec_f(SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1) {
            mVec[0] = f0; 
            mVec[1] = f1;
        }

        UME_FORCE_INLINE SIMDVec_f(
            SCALAR_FLOAT_TYPE f0, SCALAR_FLOAT_TYPE f1,
            SCALAR_FLOAT_TYPE f2, SCALAR_FLOAT_TYPE f3) {
            mVec[0] = f0; 
            mVec[1] = f1;
            mVec[2] = f2;
            mVec[3] = f3;
        }

        UME_FORCE_INLINE SIMDVec_f(
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
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE extract(uint32_t index) const {
            return mVec[index];
        }
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, SCALAR_FLOAT_TYPE value) {
            mVec[index] = value;
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
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & src) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_src_ptr = &src.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & src) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_src_ptr = &src.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_src_ptr[i];
            }
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SCALAR_FLOAT_TYPE b) {
            return this->assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f & load(SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE const *p) {
            SCALAR_FLOAT_TYPE *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_ptr[i] = local_p_ptr[i];
            }
            return *this;
        }
        
        // STORE
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* store(SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* store(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // STOREA
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* storea(SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* storea(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* p) const {
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *local_p_ptr = &p[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if (local_mask_ptr[i] == true) local_p_ptr[i] = local_ptr[i];
            }
            return p;
        }
        
        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE *retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const *local_b_ptr = &b.mVec[0];
            bool const *local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) retval_ptr[i] = local_b_ptr[i];
                else retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE const *local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE *retval_ptr = &retval.mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] + b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SCALAR_FLOAT_TYPE b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] + b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] += local_b_ptr[i];
            }
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] += b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SCALAR_FLOAT_TYPE b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f postinc() {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]++;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<VEC_LEN> const & mask) {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]++;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                ++local_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) ++local_ptr[i];
            }
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] - b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SCALAR_FLOAT_TYPE b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] - b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] -= local_b_ptr[i];
            }
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] -= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SCALAR_FLOAT_TYPE b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b - local_ptr[i];
            }
            return retval;
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b - local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = local_b_ptr[i] - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = local_b_ptr[i] - local_ptr[i];
                else local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b - local_ptr[i];
            }
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b - local_ptr[i];
                else local_ptr[i] = b;
            }
            return *this;
        }
        
        
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i]--;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<VEC_LEN> const & mask) {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i]--;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                --local_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) --local_ptr[i];
            }
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SCALAR_FLOAT_TYPE b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= local_b_ptr[i];
            }
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] *= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SCALAR_FLOAT_TYPE b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] *= b;
            }
            return *this;
        }        
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / local_b_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] / b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SCALAR_FLOAT_TYPE b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] / b;
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= local_b_ptr[i];
            }
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] /= b;
            }
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SCALAR_FLOAT_TYPE b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] /= b;
            }
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
            }
            return retval;
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = b / local_ptr[i];
            }
            return retval;
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = b / local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
               local_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = SCALAR_FLOAT_TYPE(1.0f) / local_ptr[i];
            }
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = b / local_ptr[i];
            }
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpeq(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpeq(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] == b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator== (SCALAR_FLOAT_TYPE b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpne(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpne(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] != b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator!= (SCALAR_FLOAT_TYPE b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpgt(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpgt(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] > b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator> (SCALAR_FLOAT_TYPE b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmplt(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmplt(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] < b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator< (SCALAR_FLOAT_TYPE b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpge(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmpge(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] >= b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator>= (SCALAR_FLOAT_TYPE b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmple(SIMDVec_f const & b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= local_b_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> cmple(SCALAR_FLOAT_TYPE b) const {
            SIMDVecMask<VEC_LEN> retval;
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool * local_retval_ptr = &retval.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] <= b;
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVecMask<VEC_LEN> operator<= (SCALAR_FLOAT_TYPE b) const {
            return cmple(b);
        }
        
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
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
        UME_FORCE_INLINE bool cmpe(SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hadd() const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(0.0f);
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(0.0f);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hadd(SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + local_ptr[i];
            }
            return retval;
        }
        // MHADDS
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hadd(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(0.0f);
            }
            #pragma omp simd reduction(+:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval + masked_copy[i];
            }
            return retval;
        }
        // HMUL
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hmul() const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(1.0f);
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;        }
        // MHMUL
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = SCALAR_FLOAT_TYPE(1.0f);
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hmul(SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * local_ptr[i];
            }
            return retval;
        }
        // MHMULS
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE hmul(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE masked_copy[VEC_LEN];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE retval = b;
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) masked_copy[i] = local_ptr[i];
                else masked_copy[i] = SCALAR_FLOAT_TYPE(1.0f);
            }
            #pragma omp simd reduction(*:retval)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                retval = retval * masked_copy[i];
            }
            return retval;
        }        
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
            }
            return retval;
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] + local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
            }
            return retval;
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = local_ptr[i] * local_b_ptr[i] - local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] + local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
            }
            return retval;
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            SCALAR_FLOAT_TYPE const * local_c_ptr = &c.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = (local_ptr[i] - local_b_ptr[i]) * local_c_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f max(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f & maxa(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] <= b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < local_b_ptr[i]) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = local_b_ptr[i];
            }
            return retval;
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f min(SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < b) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = b;
            }
            return retval;
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > local_b_ptr[i]) local_ptr[i] = local_b_ptr[i];
            }
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<VEC_LEN> const & mask, SIMDVec_f const & b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            SCALAR_FLOAT_TYPE const * local_b_ptr = &b.mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f & mina(SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] > b) local_ptr[i] = b;
            }
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE b) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
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

        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SCALAR_FLOAT_TYPE const * baseAddr, SCALAR_UINT_TYPE const * indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE const * baseAddr, SCALAR_UINT_TYPE const * indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices[i]];
            }
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SCALAR_FLOAT_TYPE const * baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE const * baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i] == true) mVec[i] = baseAddr[indices.mVec[i]];
            }
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* scatter(SCALAR_FLOAT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* baseAddr, SCALAR_UINT_TYPE* indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices[i]] = mVec[i];
            }
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* scatter(SCALAR_FLOAT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE SCALAR_FLOAT_TYPE* scatter(SIMDVecMask<VEC_LEN> const & mask, SCALAR_FLOAT_TYPE* baseAddr, SIMDVec_u<SCALAR_UINT_TYPE, VEC_LEN> const & indices) const {
            for(unsigned int i = 0; i < VEC_LEN; i++)
            {
                if(mask.mMask[i]) baseAddr[indices.mVec[i]] = mVec[i];
            }
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = -local_ptr[i];
            }
            return retval;
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_retval_ptr[i] = -local_ptr[i];
                else local_retval_ptr[i] = local_ptr[i];
            }
            return retval;
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_mask_ptr[i] == true) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] >= 0 ) local_retval_ptr[i] = local_ptr[i];
                else local_retval_ptr[i] = -local_ptr[i];
            }
            return retval;
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_f retval;
            SCALAR_FLOAT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
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
        UME_FORCE_INLINE SIMDVec_f & absa() {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                if(local_ptr[i] < 0 ) local_ptr[i] = -local_ptr[i];
            }
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<VEC_LEN> const & mask) {
            SCALAR_FLOAT_TYPE * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
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
        UME_FORCE_INLINE SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> trunc() const {
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
                local_retval_ptr[i] = (SCALAR_INT_TYPE) local_ptr[i];
            }
            return retval;
        }
        // MTRUNC
        UME_FORCE_INLINE SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> trunc(SIMDVecMask<VEC_LEN> const & mask) const {
            SIMDVec_i<SCALAR_INT_TYPE, VEC_LEN> retval;
            SCALAR_INT_TYPE * local_retval_ptr = &retval.mVec[0];
            SCALAR_FLOAT_TYPE const * local_ptr = &mVec[0];
            bool const * local_mask_ptr = &mask.mMask[0];
            #pragma omp simd safelen(VEC_LEN)
            for(unsigned int i = 0; i < VEC_LEN; i++) {
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
