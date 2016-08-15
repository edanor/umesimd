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

#ifndef UME_SIMD_VEC_FLOAT32_32_H_
#define UME_SIMD_VEC_FLOAT32_32_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define BLEND(a_256, b_256, mask_256i) _mm256_blendv_ps(a_256, b_256, _mm256_castsi256_ps(mask_256i))

namespace UME {
namespace SIMD {

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
            SIMDSwizzle<32>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 32>,
            SIMDVec_f<float, 16>>
    {
    private:
        __m256 mVec[4];

        inline SIMDVec_f(__m256 const & x0,
            __m256 const & x1,
            __m256 const & x2,
            __m256 const & x3) {
            this->mVec[0] = x0;
            this->mVec[1] = x1;
            this->mVec[2] = x2;
            this->mVec[3] = x3;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        inline SIMDVec_f() {}

        // SET-CONSTR  - One element constructor
        inline SIMDVec_f(float f) {
            mVec[0] = _mm256_set1_ps(f);
            mVec[1] = _mm256_set1_ps(f);
            mVec[2] = _mm256_set1_ps(f);
            mVec[3] = _mm256_set1_ps(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        inline explicit SIMDVec_f(float const *p) { this->load(p); }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        inline SIMDVec_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11,
            float f12, float f13, float f14, float f15,
            float f16, float f17, float f18, float f19,
            float f20, float f21, float f22, float f23,
            float f24, float f25, float f26, float f27,
            float f28, float f29, float f30, float f31) {
            mVec[0] = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
            mVec[1] = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
            mVec[2] = _mm256_setr_ps(f16, f17, f18, f19, f20, f21, f22, f23);
            mVec[3] = _mm256_setr_ps(f24, f25, f26, f27, f28, f29, f30, f31);
        }
        // EXTRACT
        inline float extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVec[0]);
                return raw[index];
            }
            else if (index < 16) {
                _mm256_store_ps(raw, mVec[1]);
                return raw[index - 8];
            }
            else if (index < 24) {
                _mm256_store_ps(raw, mVec[2]);
                return raw[index - 16];
            }
            else {
                _mm256_store_ps(raw, mVec[3]);
                return raw[index - 24];
            }
        }
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_ps(raw);
            }
            else if (index < 16) {
                _mm256_store_ps(raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm256_load_ps(raw);
            }
            else if (index < 24) {
                _mm256_store_ps(raw, mVec[2]);
                raw[index - 16] = value;
                mVec[2] = _mm256_load_ps(raw);
            }
            else {
                _mm256_store_ps(raw, mVec[3]);
                raw[index - 24] = value;
                mVec[3] = _mm256_load_ps(raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<32>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            mVec[2] = b.mVec[2];
            mVec[3] = b.mVec[3];
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return this->assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            mVec[0] = BLEND(mVec[0], b.mVec[0], mask.mMask[0]);
            mVec[1] = BLEND(mVec[1], b.mVec[1], mask.mMask[1]);
            mVec[2] = BLEND(mVec[2], b.mVec[2], mask.mMask[2]);
            mVec[3] = BLEND(mVec[3], b.mVec[3], mask.mMask[3]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec[0] = _mm256_set1_ps(b);
            mVec[1] = _mm256_set1_ps(b);
            mVec[2] = _mm256_set1_ps(b);
            mVec[3] = _mm256_set1_ps(b);
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<32> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t0, _mm256_cvtepi32_ps(mask.mMask[1]));
            mVec[2] = _mm256_blendv_ps(mVec[2], t0, _mm256_cvtepi32_ps(mask.mMask[2]));
            mVec[3] = _mm256_blendv_ps(mVec[3], t0, _mm256_cvtepi32_ps(mask.mMask[3]));
            return *this;
        }

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
        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec[0] = _mm256_loadu_ps(p);
            mVec[1] = _mm256_loadu_ps(p + 8);
            mVec[2] = _mm256_loadu_ps(p + 16);
            mVec[3] = _mm256_loadu_ps(p + 24);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<32> const & mask, float const * p) {
            __m256 t0 = _mm256_loadu_ps(p);
            __m256 t1 = _mm256_loadu_ps(p + 8);
            __m256 t2 = _mm256_loadu_ps(p + 16);
            __m256 t3 = _mm256_loadu_ps(p + 24);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            mVec[2] = _mm256_blendv_ps(mVec[2], t2, _mm256_cvtepi32_ps(mask.mMask[2]));
            mVec[3] = _mm256_blendv_ps(mVec[3], t3, _mm256_cvtepi32_ps(mask.mMask[3]));
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec[0] = _mm256_load_ps(p);
            mVec[1] = _mm256_load_ps(p + 8);
            mVec[2] = _mm256_load_ps(p + 16);
            mVec[3] = _mm256_load_ps(p + 24);
            return *this;
        }

        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<32> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            __m256 t1 = _mm256_load_ps(p + 8);
            __m256 t2 = _mm256_load_ps(p + 16);
            __m256 t3 = _mm256_load_ps(p + 24);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            mVec[2] = _mm256_blendv_ps(mVec[2], t2, _mm256_cvtepi32_ps(mask.mMask[2]));
            mVec[3] = _mm256_blendv_ps(mVec[3], t3, _mm256_cvtepi32_ps(mask.mMask[3]));
            return *this;
        }
        // STORE
        inline float* store(float* p) const {
            _mm256_storeu_ps(p, mVec[0]);
            _mm256_storeu_ps(p + 8, mVec[1]);
            _mm256_storeu_ps(p + 16, mVec[2]);
            _mm256_storeu_ps(p + 24, mVec[3]);
            return p;
        }
        // MSTORE
        inline float* store(SIMDVecMask<32> const & mask, float* p) const {
            __m256 t0 = _mm256_loadu_ps(p);
            __m256 t1 = _mm256_loadu_ps(p + 8);
            __m256 t2 = _mm256_loadu_ps(p + 16);
            __m256 t3 = _mm256_loadu_ps(p + 24);
            __m256 t4 = _mm256_blendv_ps(t0, mVec[0], _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t5 = _mm256_blendv_ps(t1, mVec[1], _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t6 = _mm256_blendv_ps(t2, mVec[2], _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t7 = _mm256_blendv_ps(t3, mVec[3], _mm256_cvtepi32_ps(mask.mMask[3]));
            _mm256_storeu_ps(p, t4);
            _mm256_storeu_ps(p + 8, t5);
            _mm256_storeu_ps(p + 16, t6);
            _mm256_storeu_ps(p + 24, t7);
            return p;
        }
        // STOREA
        inline float* storea(float* p) const {
            _mm256_store_ps(p, mVec[0]);
            _mm256_store_ps(p + 8, mVec[1]);
            _mm256_store_ps(p + 16, mVec[2]);
            _mm256_store_ps(p + 24, mVec[3]);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<32> const & mask, float* p) const {
            __m256 t0 = _mm256_load_ps(p);
            __m256 t1 = _mm256_load_ps(p + 8);
            __m256 t2 = _mm256_load_ps(p + 16);
            __m256 t3 = _mm256_load_ps(p + 24);
            __m256 t4 = _mm256_blendv_ps(t0, mVec[0], _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t5 = _mm256_blendv_ps(t1, mVec[1], _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t6 = _mm256_blendv_ps(t2, mVec[2], _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t7 = _mm256_blendv_ps(t3, mVec[3], _mm256_cvtepi32_ps(mask.mMask[3]));
            _mm256_storeu_ps(p, t4);
            _mm256_storeu_ps(p + 8, t5);
            _mm256_storeu_ps(p + 16, t6);
            _mm256_storeu_ps(p + 24, t7);
            return p;
        }

        //(Addition operations)
        // ADDV     - Add with vector
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_add_ps(mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_add_ps(mVec[3], b.mVec[3]);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_add_ps(mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_add_ps(mVec[3], b.mVec[3]);
            __m256 t4 = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t5 = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t6 = _mm256_blendv_ps(mVec[2], t2, _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t7 = _mm256_blendv_ps(mVec[3], t3, _mm256_cvtepi32_ps(mask.mMask[3]));
            return SIMDVec_f(t4, t5, t6, t7);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec[0], t0);
            __m256 t2 = _mm256_add_ps(mVec[1], t0);
            __m256 t3 = _mm256_add_ps(mVec[2], t0);
            __m256 t4 = _mm256_add_ps(mVec[3], t0);
            return SIMDVec_f(t1, t2, t3, t4);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<32> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec[0], t0);
            __m256 t2 = _mm256_add_ps(mVec[1], t0);
            __m256 t3 = _mm256_add_ps(mVec[2], t0);
            __m256 t4 = _mm256_add_ps(mVec[3], t0);
            __m256 t5 = _mm256_blendv_ps(mVec[0], t1, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t6 = _mm256_blendv_ps(mVec[1], t2, _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t7 = _mm256_blendv_ps(mVec[2], t3, _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t8 = _mm256_blendv_ps(mVec[3], t4, _mm256_cvtepi32_ps(mask.mMask[3]));
            return SIMDVec_f(t5, t6, t7, t8);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            this->mVec[0] = _mm256_add_ps(mVec[0], b.mVec[0]);
            this->mVec[1] = _mm256_add_ps(mVec[1], b.mVec[1]);
            this->mVec[2] = _mm256_add_ps(mVec[2], b.mVec[2]);
            this->mVec[3] = _mm256_add_ps(mVec[3], b.mVec[3]);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_add_ps(mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_add_ps(mVec[3], b.mVec[3]);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            mVec[2] = _mm256_blendv_ps(mVec[2], t2, _mm256_cvtepi32_ps(mask.mMask[2]));
            mVec[3] = _mm256_blendv_ps(mVec[3], t3, _mm256_cvtepi32_ps(mask.mMask[3]));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            this->mVec[0] = _mm256_add_ps(mVec[0], t0);
            this->mVec[1] = _mm256_add_ps(mVec[1], t0);
            this->mVec[2] = _mm256_add_ps(mVec[2], t0);
            this->mVec[3] = _mm256_add_ps(mVec[3], t0);
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<32> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec[0], t0);
            __m256 t2 = _mm256_add_ps(mVec[1], t0);
            __m256 t3 = _mm256_add_ps(mVec[2], t0);
            __m256 t4 = _mm256_add_ps(mVec[3], t0);
            mVec[0] = _mm256_blendv_ps(mVec[0], t1, _mm256_cvtepi32_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t2, _mm256_cvtepi32_ps(mask.mMask[1]));
            mVec[2] = _mm256_blendv_ps(mVec[2], t3, _mm256_cvtepi32_ps(mask.mMask[2]));
            mVec[3] = _mm256_blendv_ps(mVec[3], t4, _mm256_cvtepi32_ps(mask.mMask[3]));
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
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_sub_ps(mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_sub_ps(mVec[3], b.mVec[3]);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], b.mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(mVec[1], b.mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            __m256 t4 = _mm256_sub_ps(mVec[2], b.mVec[2]);
            __m256 t5 = BLEND(mVec[2], t4, mask.mMask[2]);
            __m256 t6 = _mm256_sub_ps(mVec[3], b.mVec[3]);
            __m256 t7 = BLEND(mVec[3], t6, mask.mMask[3]);
            return SIMDVec_f(t1, t3, t5, t7);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = _mm256_sub_ps(mVec[1], _mm256_set1_ps(b));
            __m256 t2 = _mm256_sub_ps(mVec[2], _mm256_set1_ps(b));
            __m256 t3 = _mm256_sub_ps(mVec[3], _mm256_set1_ps(b));
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<32> const & mask, float b) const {
            __m256 t0 = _mm256_sub_ps(mVec[0], _mm256_set1_ps(b));
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sub_ps(mVec[1], _mm256_set1_ps(b));
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            __m256 t4 = _mm256_sub_ps(mVec[2], _mm256_set1_ps(b));
            __m256 t5 = BLEND(mVec[2], t4, mask.mMask[2]);
            __m256 t6 = _mm256_sub_ps(mVec[3], _mm256_set1_ps(b));
            __m256 t7 = BLEND(mVec[3], t6, mask.mMask[3]);
            return SIMDVec_f(t1, t3, t5, t7);
        }
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
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_mul_ps(mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_mul_ps(mVec[3], b.mVec[3]);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_mul_ps(mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_mul_ps(mVec[3], b.mVec[3]);
            __m256 t4 = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t5 = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t6 = _mm256_blendv_ps(mVec[2], t2, _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t7 = _mm256_blendv_ps(mVec[3], t3, _mm256_cvtepi32_ps(mask.mMask[3]));
            return SIMDVec_f(t4, t5, t6, t7);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(this->mVec[0], t0);
            __m256 t2 = _mm256_mul_ps(this->mVec[1], t0);
            __m256 t3 = _mm256_mul_ps(this->mVec[2], t0);
            __m256 t4 = _mm256_mul_ps(this->mVec[3], t0);
            return SIMDVec_f(t1, t2, t3, t4);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<32> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVec[0], t0);
            __m256 t2 = _mm256_mul_ps(mVec[1], t0);
            __m256 t3 = _mm256_mul_ps(mVec[2], t0);
            __m256 t4 = _mm256_mul_ps(mVec[3], t0);
            __m256 t5 = _mm256_blendv_ps(mVec[0], t1, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t6 = _mm256_blendv_ps(mVec[1], t2, _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t7 = _mm256_blendv_ps(mVec[2], t3, _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t8 = _mm256_blendv_ps(mVec[3], t4, _mm256_cvtepi32_ps(mask.mMask[3]));
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
        // CMPLTV
        inline SIMDVecMask<32> cmplt(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec[0], b.mVec[0], _CMP_LT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], b.mVec[1], _CMP_LT_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            __m256 m4 = _mm256_cmp_ps(mVec[2], b.mVec[2], _CMP_LT_OS);
            __m256i m5 = _mm256_castps_si256(m4);
            __m256 m6 = _mm256_cmp_ps(mVec[3], b.mVec[3], _CMP_LT_OS);
            __m256i m7 = _mm256_castps_si256(m6);
            return SIMDVecMask<32>(m1, m3, m5, m7);
        }
        inline SIMDVecMask<32> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<32> cmplt(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec[0], t0, _CMP_LT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            __m256 m2 = _mm256_cmp_ps(mVec[1], t0, _CMP_LT_OS);
            __m256i m3 = _mm256_castps_si256(m2);
            __m256 m4 = _mm256_cmp_ps(mVec[2], t0, _CMP_LT_OS);
            __m256i m5 = _mm256_castps_si256(m4);
            __m256 m6 = _mm256_cmp_ps(mVec[3], t0, _CMP_LT_OS);
            __m256i m7 = _mm256_castps_si256(m6);
            return SIMDVecMask<32>(m1, m3, m5, m7);
        }
        inline SIMDVecMask<32> operator< (float b) const {
            return cmplt(b);
        }
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
        inline SIMDVec_f fmuladd(SIMDVec_f const & a, SIMDVec_f const & b) const {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVecLo, a.mVecLo, b.mVecLo);
            __m256 t1 = _mm256_fmadd_ps(this->mVecHi, a.mVecHi, b.mVecHi);
            return SIMDVec_f(t0, t1);
#else
            __m256 t0 = _mm256_add_ps(b.mVec[0], _mm256_mul_ps(this->mVec[0], a.mVec[0]));
            __m256 t1 = _mm256_add_ps(b.mVec[1], _mm256_mul_ps(this->mVec[1], a.mVec[1]));
            __m256 t2 = _mm256_add_ps(b.mVec[2], _mm256_mul_ps(this->mVec[2], a.mVec[2]));
            __m256 t3 = _mm256_add_ps(b.mVec[3], _mm256_mul_ps(this->mVec[3], a.mVec[3]));
            return SIMDVec_f(t0, t1, t2, t3);
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        inline SIMDVec_f fmuladd(SIMDVecMask<32> const & mask, SIMDVec_f const & a, SIMDVec_f const & b) const {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(mVec[0], a.mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_fmadd_ps(mVec[1], a.mVec[1], b.mVec[1]);
            __m256 t2 = _mm256_fmadd_ps(mVec[2], a.mVec[2], b.mVec[2]);
            __m256 t3 = _mm256_fmadd_ps(mVec[3], a.mVec[3], b.mVec[3]);
#else
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVec[0], a.mVec[0]), b.mVec[0]);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVec[1], a.mVec[1]), b.mVec[1]);
            __m256 t2 = _mm256_add_ps(_mm256_mul_ps(mVec[2], a.mVec[2]), b.mVec[2]);
            __m256 t3 = _mm256_add_ps(_mm256_mul_ps(mVec[3], a.mVec[3]), b.mVec[3]);
#endif
            __m256 t4 = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t5 = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            __m256 t6 = _mm256_blendv_ps(mVec[2], t2, _mm256_cvtepi32_ps(mask.mMask[2]));
            __m256 t7 = _mm256_blendv_ps(mVec[3], t3, _mm256_cvtepi32_ps(mask.mMask[3]));

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

        // 3) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign

        // (Mathematical functions)
        // ABS   - Absolute value
        // MABS  - Masked absolute value
        // ABSA  - Absolute value and assign
        // MABSA - Masked absolute value and assign

        // 4) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR       - Square of vector values
        // MSQR      - Masked square of vector values
        // SQRA      - Square of vector values and assign
        // MSQRA     - Masked square of vector values and assign
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            __m256 t1 = _mm256_sqrt_ps(mVec[1]);
            __m256 t2 = _mm256_sqrt_ps(mVec[2]);
            __m256 t3 = _mm256_sqrt_ps(mVec[3]);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<32> const & mask) const {
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_sqrt_ps(mVec[1]);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            __m256 t4 = _mm256_sqrt_ps(mVec[2]);
            __m256 t5 = BLEND(mVec[2], t4, mask.mMask[2]);
            __m256 t6 = _mm256_sqrt_ps(mVec[3]);
            __m256 t7 = BLEND(mVec[3], t6, mask.mMask[3]);
            return SIMDVec_f(t1, t3, t5, t7);
        }// SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND
        inline SIMDVec_f round() const {
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t2 = _mm256_round_ps(mVec[2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t3 = _mm256_round_ps(mVec[3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<32> const & mask) const {
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = BLEND(mVec[0], t0, mask.mMask[0]);
            __m256 t2 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t3 = BLEND(mVec[1], t2, mask.mMask[1]);
            __m256 t4 = _mm256_round_ps(mVec[2], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t5 = BLEND(mVec[2], t4, mask.mMask[2]);
            __m256 t6 = _mm256_round_ps(mVec[3], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t7 = BLEND(mVec[3], t6, mask.mMask[2]);
            return SIMDVec_f(t1, t3, t5, t7);
        }
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

        // PROMOTE
        // -
        // DEGRADE
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 32>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 32>() const;
    };
}
}

#undef BLEND

#endif
