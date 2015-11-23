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

#ifndef UME_SIMD_VEC_FLOAT32_4_H_
#define UME_SIMD_VEC_FLOAT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

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
            SIMDVecSwizzle<4 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2 >>
    {
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_i<int32_t, 4>;


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
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV    - Masked assignment with another vector
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS     - Assignment with scalar
        inline SIMDVec_f & assign(float b) {
            mVec = _mm_set1_ps(b);
            return *this;
        }
        // MASSIGNS    - Masked assign with scalar
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
            mVec = _mm_mask_mov_ps(mVec, mask.mMask, _mm_set1_ps(b));
            return *this;
        }

        //(Memory access)
        // LOAD    - Load from memory (either aligned or unaligned) to vector 
        inline SIMDVec_f & load(float const * p) {
            mVec = _mm_loadu_ps(p);
            return *this;
        }
        // MLOAD   - Masked load from memory (either aligned or unaligned) to
        //        vector
        inline SIMDVec_f & load(SIMDVecMask<4> const & mask, float const * p) {
            mVec = _mm_mask_loadu_ps(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA   - Load from aligned memory to vector
        inline SIMDVec_f & loada(float const * p) {
            mVec = _mm_load_ps(p);
            return *this;
        }
        // MLOADA  - Masked load from aligned memory to vector
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const * p) {
            mVec = _mm_mask_loadu_ps(mVec, mask.mMask, p);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        inline float* store(float * p) const {
            _mm_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //        unaligned)
        inline float * store(SIMDVecMask<4> const & mask, float * p) const {
            _mm_mask_storeu_ps(p, mask.mMask, mVec);
            return p;
        }
        // STOREA  - Store vector content into aligned memory
        inline float* storea(float * p) const {
            _mm_store_ps(p, mVec);
            return p;
        }
        // MSTOREA - Masked store vector content into aligned memory
        inline float* storea(SIMDVecMask<4> const & mask, float * p) const {
            _mm_mask_store_ps(p, mask.mMask, mVec);
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
            __m128 t0 = _mm_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // ADDS     - Add with scalar
        inline SIMDVec_f add(float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MADDS    - Masked add with scalar
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // ADDVA    - Add with vector and assign
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm_add_ps(this->mVec, b.mVec);
            return *this;
        }
        // MADDVA   - Masked add with vector and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA    - Add with scalar and assign
        inline SIMDVec_f & adda(float b) {
            mVec = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return *this;
        }
        // MADDSA   - Masked add with scalar and assign
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(b));
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
        inline SIMDVec_f postinc() {
            __m128 t0 = mVec;
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            __m128 t0 = mVec;
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTINC - Masked postfix increment
        inline SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // PREFINC  - Prefix increment
        inline SIMDVec_f & prefinc() {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC - Masked prefix increment
        inline SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_add_ps(mVec, mask.mMask, mVec, _mm_set1_ps(1.0f));
            return *this;
        }

        //(Subtraction operations)
        // SUBV       - Sub with vector
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MSUBV      - Masked sub with vector
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // SUBS       - Sub with scalar
        inline SIMDVec_f sub(float b) const {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MSUBS      - Masked subtraction with scalar
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // SUBVA      - Sub with vector and assign
        inline SIMDVec_f & sub(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(mVec, b.mVec);
            return *this;
        }
        // MSUBVA     - Masked sub with vector and assign
        inline SIMDVec_f & sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA      - Sub with scalar and assign
        inline SIMDVec_f & sub(float b) {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return *this;
        }
        // MSUBSA     - Masked sub with scalar and assign
        inline SIMDVec_f & sub(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t0);
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
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV  - Masked sub from vector
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMS   - Sub from scalar (promoted to vector)
        inline SIMDVec_f subfrom(float b) const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return SIMDVec_f(t1);
        }
        // SUBFROMVA  - Sub from vector and assign
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA - Masked sub from vector and assign
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        inline SIMDVec_f & subfroma(float b) {
            mVec = _mm_sub_ps(_mm_set1_ps(b), mVec);
            return *this;
        }
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC    - Postfix decrement
        inline SIMDVec_f postdec() {
            __m128 t0 = mVec;
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            __m128 t0 = mVec;
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTDEC   - Masked postfix decrement
        inline SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
            __m128 t1 = _mm_set1_ps(1.0f);
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t1);
            return SIMDVec_f(t0);
        }
        // PREFDEC    - Prefix decrement
        inline SIMDVec_f & prefdec() {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        // MPREFDEC   - Masked prefix decrement
        inline SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_set1_ps(1.0f);
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        //(Multiplication operations)
        // MULV   - Multiplication with vector
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMULV  - Masked multiplication with vector
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MULS   - Multiplication with scalar
        inline SIMDVec_f mul(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMULS  - Masked multiplication with scalar
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MULVA  - Multiplication with vector and assign
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm_mul_ps(mVec, b.mVec);
            return *this;
        }
        // MMULVA - Masked multiplication with vector and assign
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA  - Multiplication with scalar and assign
        inline SIMDVec_f & mula(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(mVec, t0);
            return *this;
        }
        // MMULSA - Masked multiplication with scalar and assign
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        //(Division operations)
        // DIVV   - Division with vector
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MDIVV  - Masked division with vector
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // DIVS   - Division with scalar
        inline SIMDVec_f div(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MDIVS  - Masked division with scalar
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // DIVVA  - Division with vector and assign
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm_div_ps(mVec, b.mVec);
            return *this;
        }
        // MDIVVA - Masked division with vector and assign
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA  - Division with scalar and assign
        inline SIMDVec_f & diva(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_div_ps(mVec, t0);
            return *this;
        }
        // MDIVSA - Masked division with scalar and assign
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RCP    - Reciprocal
        inline SIMDVec_f rcp() const {
            __m128 t0 = _mm_rcp14_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MRCP   - Masked reciprocal
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // RCPS   - Reciprocal with scalar numerator
        inline SIMDVec_f rcp(float b) const {
            __m128 t0 = _mm_rcp14_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS  - Masked reciprocal with scalar
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return SIMDVec_f(t2);
        }
        // RCPA   - Reciprocal and assign
        inline SIMDVec_f & rcpa() {
            mVec = _mm_rcp14_ps(mVec);
            return *this;
        }
        // MRCPA  - Masked reciprocal and assign
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RCPSA  - Reciprocal with scalar and assign
        inline SIMDVec_f & rcpa(float b) {
            __m128 t0 = _mm_rcp14_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA - Masked reciprocal with scalar and assign
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m128 t1 = _mm_set1_ps(b);
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return *this;
        }

        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        inline SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 0);
            return SIMDVecMask<4>(t0);
        }
        // CMPEQS - Element-wise 'equal' with scalar
        inline SIMDVecMask<4> cmpeq(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 0);
            return SIMDVecMask<4>(t0);
        }
        // CMPNEV - Element-wise 'not equal' with vector
        inline SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 12);
            return SIMDVecMask<4>(t0);
        }
        // CMPNES - Element-wise 'not equal' with scalar
        inline SIMDVecMask<4> cmpne(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 12);
            return SIMDVecMask<4>(t0);
        }
        // CMPGTV - Element-wise 'greater than' with vector
        inline SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 30);
            return SIMDVecMask<4>(t0);
        }
        // CMPGTS - Element-wise 'greater than' with scalar
        inline SIMDVecMask<4> cmpgt(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 30);
            return SIMDVecMask<4>(t0);
        }
        // CMPLTV - Element-wise 'less than' with vector
        inline SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 17);
            return SIMDVecMask<4>(t0);
        }
        // CMPLTS - Element-wise 'less than' with scalar
        inline SIMDVecMask<4> cmplt(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 17);
            return SIMDVecMask<4>(t0);
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        inline SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 29);
            return SIMDVecMask<4>(t0);
        }
        // CMPGES - Element-wise 'greater than or equal' with scalar
        inline SIMDVecMask<4> cmpge(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 29);
            return SIMDVecMask<4>(t0);
        }
        // CMPLEV - Element-wise 'less than or equal' with vector
        inline SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 18);
            return SIMDVecMask<4>(t0);
        }
        // CMPLES - Element-wise 'less than or equal' with scalar
        inline SIMDVecMask<4> cmple(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 18);
            return SIMDVecMask<4>(t0);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, b.mVec, 0);
            return (t0 == 0x0F);
        }
        // CMPES
        inline bool cmpe(float b) const {
            __mmask8 t0 = _mm_cmp_ps_mask(mVec, _mm_set1_ps(b), 0);
            return (t0 == 0x0F);
        }
        //(Blend/Swizzle operations)
        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // SWIZZLE  - Swizzle (reorder/permute) vector elements
        // SWIZZLEA - Swizzle (reorder/permute) vector elements and assign

        //(Reduction to scalar operations)
        // HADD  - Add elements of a vector (horizontal add)
        inline float hadd() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD - Masked add elements of a vector (horizontal add)
        inline float hadd(SIMDVecMask<4> const mask) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
        }
        // HADDS
        inline float hadd(float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADDS
        inline float hadd(SIMDVecMask<4> const mask, float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return b + t0;
        }

        // HMUL
        inline float hmul() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMUL
        inline float hmul(SIMDVecMask<4> const mask) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
        }
        // HMULS
        inline float hmul(float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMULS
        inline float hmul(SIMDVecMask<4> const mask, float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return b * t0;
        }

        //(Fused arithmetics)
        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_mask_fmadd_ps(mVec, 0xF, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_mask_fmsub_ps(mVec, 0xF, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_add_ps(mVec, b.mVec);
            __m128 t1 = _mm_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            __m128 t1 = _mm_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            __m128 t1 = _mm_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) {
            __m128 t0 = _mm_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            __m128 t1 = _mm_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS  - Masked max with scalar
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_max_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_max_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MINS
        inline SIMDVec_f min(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mask_min_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = _mm_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mask_min_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX   - Max of elements of a vector (horizontal max)
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        // IMAX   - Index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        // IMIN   - Index of min element of a vector
        // MIMIN  - Masked index of min element of a vector

        // (Gather/Scatter operations)
        // GATHERS
        /*inline SIMDVec_f & gather(float* baseAddr, uint64_t* indices) {
            alignas(16) float raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_load_ps(raw);
            return *this;
        }*/
        // MGATHERS
        /*inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, float* baseAddr, uint64_t* indices) {
            alignas(16) float raw[4] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_mask_load_ps(mVec, mask.mMask, raw);
            return *this;
        }*/
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV

        // 3) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG
        inline SIMDVec_f neg() const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_mask_sub_ps(mVec, mask.mMask, _mm_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_sub_ps(mVec, mask.mMask, _mm_set1_ps(0.0f), mVec);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_abs_ps(t0);
            __m128 t2 = _mm512_castps512_ps128(t1);
            return SIMDVec_f(t2);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            __m512 t2 = _mm512_mask_abs_ps(t0, t1, t0);
            __m128 t3 = _mm512_castps512_ps128(t2);
            return SIMDVec_f(t3);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __m512 t1 = _mm512_abs_ps(t0);
            mVec = _mm512_castps512_ps128(t1);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __m512 t0 = _mm512_castps128_ps512(mVec);
            __mmask16 t1 = (__mmask16)mask.mMask;
            __m512 t2 = _mm512_mask_abs_ps(t0, t1, t0);
            mVec = _mm512_castps512_ps128(t2);
            return *this;
        }

        // 4) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR
        inline SIMDVec_f sqr() const {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec = _mm_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m128 t0 = _mm_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = _mm_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
            mVec = _mm_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND
        inline SIMDVec_f round() const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 4> trunc() {
            __m128i t0 = _mm_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 4> trunc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_mask_cvttps_epi32(_mm_setzero_si128(), mask.mMask, mVec);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // FLOOR
        inline SIMDVec_f floor() const {
            __m128 t0 = _mm_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_floor_ps(mVec);
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // CEIL
        inline SIMDVec_f ceil() const {
            __m128 t0 = _mm_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        inline SIMDVec_f ceil(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_ceil_ps(mVec);
            __m128 t1 = _mm_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // ISFIN
        inline SIMDVecMask<4> isfin() const {
            __mmask8 t0 = _mm_fpclass_ps_mask(mVec, 0x08);
            __mmask8 t1 = _mm_fpclass_ps_mask(mVec, 0x10);
            __mmask8 t2 = 0xF & ((~t0) & (~t1));
            return SIMDVecMask<4>(t2);
        }
        // ISINF
        inline SIMDVecMask<4> isinf() const {
            __mmask8 t0 = _mm_fpclass_ps_mask(mVec, 0x08);
            __mmask8 t1 = _mm_fpclass_ps_mask(mVec, 0x10);
            __mmask8 t2 = 0xF & (t0 | t1);
            return SIMDVecMask<4>(t2);
        }
        // ISAN
        inline SIMDVecMask<4> isan() const {
            __mmask8 t0 = _mm_fpclass_ps_mask(mVec, 0x01);
            __mmask8 t1 = _mm_fpclass_ps_mask(mVec, 0x80);
            __mmask8 t2 = 0xF & ((~t0) & (~t1));
            return SIMDVecMask<4>(t2);
        }
        // ISNAN
        inline SIMDVecMask<4> isnan() const {
            __mmask8 t0 = _mm_fpclass_ps_mask(mVec, 0x01);
            __mmask8 t1 = _mm_fpclass_ps_mask(mVec, 0x80);
            __mmask8 t2 = 0xF & (t0 | t1);
            return SIMDVecMask<4>(t2);
        }
        // ISNORM
        inline SIMDVecMask<4> isnorm() const {
            __mmask8 t0 = ~_mm_fpclass_ps_mask(mVec, 0x01);
            __mmask8 t1 = ~_mm_fpclass_ps_mask(mVec, 0x02);
            __mmask8 t2 = ~_mm_fpclass_ps_mask(mVec, 0x04);
            __mmask8 t3 = ~_mm_fpclass_ps_mask(mVec, 0x08);
            __mmask8 t4 = ~_mm_fpclass_ps_mask(mVec, 0x10);
            __mmask8 t5 = ~_mm_fpclass_ps_mask(mVec, 0x20);
            __mmask8 t6 = ~_mm_fpclass_ps_mask(mVec, 0x80);
            __mmask8 t7 = 0xF & t0 & t1 & t2 & t3 & t4 & t5 & t6;
            return SIMDVecMask<4>(t7);
        }
        // ISSUB
        inline SIMDVecMask<4> issub() const {
            __mmask8 t0 = 0xF & _mm_fpclass_ps_mask(mVec, 0x20);
            return SIMDVecMask<4>(t0);
        }
        // ISZERO    - Is zero
        inline SIMDVecMask<4> iszero() const {
            __mmask8 t0 = _mm_fpclass_ps_mask(mVec, 0x02);
            __mmask8 t1 = _mm_fpclass_ps_mask(mVec, 0x04);
            __mmask8 t2 = 0xF & (t0 | t1);
            return SIMDVecMask<4>(t2);
        }
        // ISZEROSUB - Is zero or subnormal
        inline SIMDVecMask<4> iszerosub() const {
            __mmask8 t0 = _mm_fpclass_ps_mask(mVec, 0x02);
            __mmask8 t1 = _mm_fpclass_ps_mask(mVec, 0x04);
            __mmask8 t2 = _mm_fpclass_ps_mask(mVec, 0x20);
            __mmask8 t3 = 0xF & (t0 | t1 | t2);
            return SIMDVecMask<4>(t3);
        }
        // SIN       - Sine
        // MSIN      - Masked sine
        // COS       - Cosine
        // MCOS      - Masked cosine
        // TAN       - Tangent
        // MTAN      - Masked tangent
        // CTAN      - Cotangent
        // MCTAN     - Masked cotangent

        // (Pack/Unpack operations - not available for SIMD1)
        // PACK
        inline SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_ps(raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], 0.0f, 0.0f };
            mVec = _mm_mask_load_ps(mVec, 0x3, raw);
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4] = { 0.0f, 0.0f, b.mVec[0], b.mVec[1] };
            mVec = _mm_mask_load_ps(mVec, 0xC, raw);
            return *this;
        }
        // UNPACK   - Unpack lower and upper halfs to half-length vectors.
        inline void unpack(SIMDVec_f<float, 2> & a, SIMDVec_f<float, 2> & b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        inline SIMDVec_f<float, 2> unpacklo() const {
            alignas(16) float raw[4];
            _mm_mask_store_ps(raw, 0x3, mVec);
            return SIMDVec_f<float, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 2> unpackhi() const {
            alignas(16) float raw[4];
            _mm_mask_store_ps(raw, 0xC, mVec);
            return SIMDVec_f<float, 2>(raw[2], raw[3]);
        }

        // FTOU
        inline operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 4>() const;
    };
}
}

#endif
