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

#ifndef UME_SIMD_VEC_FLOAT32_16_H_
#define UME_SIMD_VEC_FLOAT32_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    //class SIMDVec_f<double, 16>;

    template<>
    class SIMDVec_f<float, 16> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_u<uint32_t, 16>,
            SIMDVec_i<int32_t, 16>,
            float,
            16,
            uint32_t,
            int32_t,
            SIMDVecMask<16>,
            SIMDSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 16>,
            SIMDVec_f<float, 8>>
    {
    private:
        __m256 mVec[2];

        UME_FORCE_INLINE SIMDVec_f(__m256 const & lo, __m256 const & hi) {
            this->mVec[0] = lo;
            this->mVec[1] = hi;
        }

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec[0] = _mm256_set1_ps(f);
            mVec[1] = _mm256_set1_ps(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_f(float const * p) {
            mVec[0] = _mm256_loadu_ps(p);
            mVec[1] = _mm256_loadu_ps(p + 8);
        }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3,
            float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11,
            float f12, float f13, float f14, float f15) {
            mVec[0] = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
            mVec[1] = _mm256_setr_ps(f8, f9, f10, f11, f12, f13, f14, f15);
        }

        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVec[0]);
                return raw[index];
            }
            else {
                _mm256_store_ps(raw, mVec[1]);
                return raw[index - 8];
            }
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            alignas(32) float raw[8];
            if (index < 8) {
                _mm256_store_ps(raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_ps(raw);
            }
            else {
                _mm256_store_ps(raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm256_load_ps(raw);
            }
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm256_blendv_ps(mVec[0], b.mVec[0], _mm256_castsi256_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], b.mVec[1], _mm256_castsi256_ps(mask.mMask[1]));
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec[0] = _mm256_set1_ps(b);
            mVec[1] = mVec[0];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t0, _mm256_castsi256_ps(mask.mMask[1]));
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
            mVec[0] = _mm256_loadu_ps(p);
            mVec[1] = _mm256_loadu_ps(p + 8);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<16> const & mask, float const * p) {
            __m256 t0 = _mm256_loadu_ps(p);
            __m256 t1 = _mm256_loadu_ps(p + 8);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_castsi256_ps(mask.mMask[1]));
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec[0] = _mm256_load_ps(p);
            mVec[1] = _mm256_load_ps(p + 8);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<16> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t1 = _mm256_load_ps(p + 8);
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_castsi256_ps(mask.mMask[1]));
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float* p) const {
            _mm256_storeu_ps(p, mVec[0]);
            _mm256_storeu_ps(p + 8, mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<16> const & mask, float * p) const {
            __m256 t0 = _mm256_loadu_ps(p);
            __m256 t1 = _mm256_loadu_ps(p + 8);
            __m256 t2 = _mm256_blendv_ps(t0, mVec[0], _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t3 = _mm256_blendv_ps(t1, mVec[1], _mm256_castsi256_ps(mask.mMask[1]));
            _mm256_storeu_ps(p, t2);
            _mm256_storeu_ps(p + 8, t3);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float* p) const {
            _mm256_store_ps(p, mVec[0]);
            _mm256_store_ps(p + 8, mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<16> const & mask, float* p) const {
            _mm256_maskstore_ps(p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_ps(p + 8, mask.mMask[1], mVec[1]);
            return p;
        }

        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(this->mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_add_ps(this->mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV    - Masked add with vector
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(this->mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t2 = _mm256_add_ps(this->mVec[1], b.mVec[1]);
            __m256 t3 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t1, t3);
        }
        // ADDS     - Add with scalar
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(this->mVec[0], t0);
            __m256 t2 = _mm256_add_ps(this->mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS    - Masked add with scalar
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec[0], t0);
            __m256 t2 = _mm256_add_ps(mVec[1], t0);
            __m256 t3 = _mm256_blendv_ps(mVec[0], t1, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t4 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t3, t4);
        }
        // ADDVA    - Add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm256_add_ps(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_ps(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return this->adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t1 = _mm256_add_ps(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_ps(mVec[1], t1, _mm256_castsi256_ps(mask.mMask[1]));
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec[0] = _mm256_add_ps(mVec[0], t0);
            mVec[1] = _mm256_add_ps(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return this->adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec[0], t0);
            __m256 t2 = _mm256_add_ps(mVec[1], t0);
            mVec[0] = _mm256_blendv_ps(mVec[0], t1, _mm256_castsi256_ps(mask.mMask[0]));
            mVec[1] = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
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
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(this->mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_sub_ps(this->mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(this->mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t2 = _mm256_sub_ps(this->mVec[1], b.mVec[1]);
            __m256 t3 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t1, t3);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_sub_ps(this->mVec[0], t0);
            __m256 t2 = _mm256_sub_ps(this->mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_sub_ps(mVec[0], t0);
            __m256 t2 = _mm256_sub_ps(mVec[1], t0);
            __m256 t3 = _mm256_blendv_ps(mVec[0], t1, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t4 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t3, t4);
        }
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
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(this->mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_mul_ps(this->mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV  - Masked multiplication with vector
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(this->mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t2 = _mm256_mul_ps(this->mVec[1], b.mVec[1]);
            __m256 t3 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t1, t3);
        }
        // MULS   - Multiplication with scalar
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(this->mVec[0], t0);
            __m256 t2 = _mm256_mul_ps(this->mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS  - Masked multiplication with scalar
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_mul_ps(mVec[0], t0);
            __m256 t2 = _mm256_mul_ps(mVec[1], t0);
            __m256 t3 = _mm256_blendv_ps(mVec[0], t1, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t4 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t3, t4);
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
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m256 t0 = _mm256_rcp_ps(this->mVec[0]);
            __m256 t1 = _mm256_rcp_ps(this->mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP   - Masked reciprocal
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_rcp_ps(this->mVec[0]);
            __m256 t1 = _mm256_blendv_ps(mVec[0], t0, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t2 = _mm256_rcp_ps(this->mVec[1]);
            __m256 t3 = _mm256_blendv_ps(mVec[1], t2, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t1, t3);
        }
        // RCPS   - Reciprocal with scalar numerator
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(t0, this->mVec[0]);
            __m256 t2 = _mm256_div_ps(t0, this->mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // MRCPS  - Masked reciprocal with scalar
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_div_ps(t0, mVec[0]);
            __m256 t2 = _mm256_blendv_ps(mVec[0], t1, _mm256_castsi256_ps(mask.mMask[0]));
            __m256 t3 = _mm256_div_ps(t0, mVec[1]);
            __m256 t4 = _mm256_blendv_ps(mVec[1], t3, _mm256_castsi256_ps(mask.mMask[1]));
            return SIMDVec_f(t2, t4);
        }
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
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_cmp_ps(mVec[0], b.mVec[0], 17);
            __m256 t1 = _mm256_cmp_ps(mVec[1], b.mVec[1], 17);
            __m256i m0 = _mm256_castps_si256(t0);
            __m256i m1 = _mm256_castps_si256(t1);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS - Element-wise 'less than' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(float b) const {
            __m256 t0 = _mm256_cmp_ps(mVec[0], _mm256_set1_ps(b), 17);
            __m256 t1 = _mm256_cmp_ps(mVec[1], _mm256_set1_ps(b), 17);
            __m256i m0 = _mm256_castps_si256(t0);
            __m256i m1 = _mm256_castps_si256(t1);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        // CMPGES - Element-wise 'greater than or equal' with scalar
        // CMPLEV - Element-wise 'less than or equal' with vector
        // CMPLES - Element-wise 'less than or equal' with scalar
        // CMPEX  - Check if vectors are exact (returns scalar 'bool')

        //(Bitwise operations)
        // ANDV   - AND with vector
        // MANDV  - Masked AND with vector
        // ANDS   - AND with scalar
        // MANDS  - Masked AND with scalar
        // ANDVA  - AND with vector and assign
        // MANDVA - Masked AND with vector and assign
        // ANDSA  - AND with scalar and assign
        // MANDSA - Masked AND with scalar and assign
        // ORV    - OR with vector
        // MORV   - Masked OR with vector
        // ORS    - OR with scalar
        // MORS   - Masked OR with scalar
        // ORVA   - OR with vector and assign
        // MORVA  - Masked OR with vector and assign
        // ORSA   - OR with scalar and assign
        // MORSA  - Masked OR with scalar and assign
        // XORV   - XOR with vector
        // MXORV  - Masked XOR with vector
        // XORS   - XOR with scalar
        // MXORS  - Masked XOR with scalar
        // XORVA  - XOR with vector and assign
        // MXORVA - Masked XOR with vector and assign
        // XORSA  - XOR with scalar and assign
        // MXORSA - Masked XOR with scalar and assign
        // NOT    - Negation of bits
        // MNOT   - Masked negation of bits
        // NOTA   - Negation of bits and assign
        // MNOTA  - Masked negation of bits and assign

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
        // assign
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        // MHADD - Masked add elements of a vector (horizontal add)
        // HMUL  - Multiply elements of a vector (horizontal mul)
        // MHMUL - Masked multiply elements of a vector (horizontal mul)
        // HAND  - AND of elements of a vector (horizontal AND)
        // MHAND - Masked AND of elements of a vector (horizontal AND)
        // HOR   - OR of elements of a vector (horizontal OR)
        // MHOR  - Masked OR of elements of a vector (horizontal OR)
        // HXOR  - XOR of elements of a vector (horizontal XOR)
        // MHXOR - Masked XOR of elements of a vector (horizontal XOR)

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & a, SIMDVec_f const & b) const {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVec[0], a.mVec[0], b.mVec[0]);
            __m256 t1 = _mm256_fmadd_ps(this->mVec[1], a.mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
#else
            __m256 t0 = _mm256_add_ps(b.mVec[0], _mm256_mul_ps(this->mVec[0], a.mVec[0]));
            __m256 t1 = _mm256_add_ps(b.mVec[1], _mm256_mul_ps(this->mVec[1], a.mVec[1]));
            return SIMDVec_f(t0, t1);
#endif
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & a, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVec[0], a.mVec[0]), b.mVec[0]);
            __m256 t1 = _mm256_add_ps(_mm256_mul_ps(mVec[1], a.mVec[1]), b.mVec[1]);
            __m256 t2 = _mm256_blendv_ps(mVec[0], t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t3 = _mm256_blendv_ps(mVec[1], t1, _mm256_cvtepi32_ps(mask.mMask[1]));
            return SIMDVec_f(t2, t3);
        }
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

        // (Binary shift operations)
        // LSHV   - Element-wise logical shift bits left (shift values in vector)
        // MLSHV  - Masked element-wise logical shift bits left (shift values in
        //          vector) 
        // LSHS   - Element-wise logical shift bits left (shift value in scalar)
        // MLSHS  - Masked element-wise logical shift bits left (shift value in
        //          scalar)
        // LSHVA  - Element-wise logical shift bits left (shift values in vector)
        //          and assign
        // MLSHVA - Masked element-wise logical shift bits left (shift values
        //          in vector) and assign
        // LSHSA  - Element-wise logical shift bits left (shift value in scalar)
        //          and assign
        // MLSHSA - Masked element-wise logical shift bits left (shift value in
        //          scalar) and assign
        // RSHV   - Logical shift bits right (shift values in vector)
        // MRSHV  - Masked logical shift bits right (shift values in vector)
        // RSHS   - Logical shift bits right (shift value in scalar)
        // MRSHV  - Masked logical shift bits right (shift value in scalar)
        // RSHVA  - Logical shift bits right (shift values in vector) and assign
        // MRSHVA - Masked logical shift bits right (shift values in vector) and
        //          assign
        // RSHSA  - Logical shift bits right (shift value in scalar) and assign
        // MRSHSA - Masked logical shift bits right (shift value in scalar) and
        //          assign

        // (Binary rotation operations)
        // ROLV   - Rotate bits left (shift values in vector)
        // MROLV  - Masked rotate bits left (shift values in vector)
        // ROLS   - Rotate bits right (shift value in scalar)
        // MROLS  - Masked rotate bits left (shift value in scalar)
        // ROLVA  - Rotate bits left (shift values in vector) and assign
        // MROLVA - Masked rotate bits left (shift values in vector) and assign
        // ROLSA  - Rotate bits left (shift value in scalar) and assign
        // MROLSA - Masked rotate bits left (shift value in scalar) and assign
        // RORV   - Rotate bits right (shift values in vector)
        // MRORV  - Masked rotate bits right (shift values in vector) 
        // RORS   - Rotate bits right (shift values in scalar)
        // MRORS  - Masked rotate bits right (shift values in scalar) 
        // RORVA  - Rotate bits right (shift values in vector) and assign 
        // MRORVA - Masked rotate bits right (shift values in vector) and assign
        // RORSA  - Rotate bits right (shift values in scalar) and assign
        // MRORSA - Masked rotate bits right (shift values in scalar) and assign

        // 3) Operations available for Signed integer and Unsigned integer 
        // data types:

        //(Signed/Unsigned cast)
        // UTOI - Cast unsigned vector to signed vector
        // ITOU - Cast signed vector to unsigned vector

        // 4) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG   - Negate signed values
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return this->neg();
        }
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
        SIMDVec_f sqrt() const {
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            __m256 t1 = _mm256_sqrt_ps(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT     - Masked square root of vector values
        SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m256 m0 = _mm256_castsi256_ps(mask.mMask[0]);
            __m256 m1 = _mm256_castsi256_ps(mask.mMask[1]);
            __m256 t0 = _mm256_sqrt_ps(mVec[0]);
            __m256 t1 = _mm256_sqrt_ps(mVec[1]);
            __m256 t2 = _mm256_blendv_ps(mVec[0], t0, m0);
            __m256 t3 = _mm256_blendv_ps(mVec[1], t1, m1);
            return SIMDVec_f(t2, t3);
        }
        // SQRTA     - Square root of vector values and assign
        // MSQRTA    - Masked square root of vector values and assign
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0, t1);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<16> const & mask) const {
            __m256 m0 = _mm256_castsi256_ps(mask.mMask[0]);
            __m256 m1 = _mm256_castsi256_ps(mask.mMask[1]);
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t2 = _mm256_blendv_ps(mVec[0], t0, m0);
            __m256 t3 = _mm256_blendv_ps(mVec[1], t1, m1);
            return SIMDVec_f(t2, t3);
        }
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        UME_FORCE_INLINE SIMDVec_i<int32_t, 16> trunc() const {
            __m256i t0 = _mm256_cvtps_epi32(_mm256_round_ps(mVec[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
            __m256i t1 = _mm256_cvtps_epi32(_mm256_round_ps(mVec[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));

            return SIMDVec_i<int32_t, 16>(t0, t1);
        }
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        UME_FORCE_INLINE SIMDVec_i<int32_t, 16> trunc(SIMDVecMask<16> const & mask) const {
            __m256 t0 = _mm256_round_ps(mVec[0], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            __m256 t1 = _mm256_round_ps(mVec[1], _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            __m256 t2 = _mm256_setzero_ps();
            __m256 t3 = _mm256_blendv_ps(t2, t0, _mm256_cvtepi32_ps(mask.mMask[0]));
            __m256 t4 = _mm256_blendv_ps(t2, t1, _mm256_cvtepi32_ps(mask.mMask[1]));

            __m256i t5 = _mm256_cvtps_epi32(t3);
            __m256i t6 = _mm256_cvtps_epi32(t4);
            return SIMDVec_i<int32_t, 16>(t5, t6);
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
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 16>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<16> const & mask) const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 16>, SIMDVecMask<16>>(mask, *this);
        }
        // LOG
        // MLOG
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 16>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 16>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 16>() const;
    };
}
}

#endif
