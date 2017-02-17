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
    public:
        typedef typename SIMDVec_f_traits<float, 16>::VEC_UINT_TYPE  VEC_UINT_TYPE;
        typedef typename SIMDVec_f_traits<float, 16>::VEC_INT_TYPE   VEC_INT_TYPE;
        typedef typename SIMDVec_f_traits<float, 16>::MASK_TYPE      MASK_TYPE;

    private:
        __m512 mVec;

        UME_FORCE_INLINE SIMDVec_f(__m512 x) {
            mVec = x;
        }

    public:
        // ZERO-CONSTR - Zero element constructor 
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec = _mm512_set1_ps(f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}

        // LOAD-CONSTR - Construct by loading from memory
        UME_FORCE_INLINE explicit SIMDVec_f(float const * p) { load(p); }

        // FULL-CONSTR - constructor with VEC_LEN scalar element 
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7,
            float f8, float f9, float f10, float f11, float f12, float f13, float f14, float f15) {
            mVec = _mm512_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15);
        }

        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_ps(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, MASK_TYPE> operator() (MASK_TYPE const & mask) {
            return IntermediateMask<SIMDVec_f, float, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, MASK_TYPE> operator[] (MASK_TYPE & mask) {
            return IntermediateMask<SIMDVec_f, float, MASK_TYPE>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        //(Initialization)
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec = _mm512_set1_ps(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<16> const & mask, float b) {
            mVec = _mm512_mask_mov_ps(mVec, mask.mMask, _mm512_set1_ps(b));
            return *this;
        }

        //(Memory access)
        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
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
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<16> const & mask, float const * p) {
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
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec = _mm512_load_ps(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<16> const & mask, float const * p) {
            mVec = _mm512_mask_load_ps(mVec, mask.mMask, p);
            return *this;
        }
        // STORE   - Store vector content into memory (either aligned or unaligned)
        UME_FORCE_INLINE float * store(float * p)
        {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_store_ps(p, mVec);
            }
            else {
                alignas(64) float raw[16];
                _mm512_store_ps(raw, mVec);
                memcpy(p, raw, 16 * sizeof(float));
            }
            return p;
        }
        // MSTORE  - Masked store vector content into memory (either aligned or
        //           unaligned)
        UME_FORCE_INLINE float * store(SIMDVecMask<16> const & mask, float *p) {
            if ((uint64_t(p) % 64) == 0) {
                _mm512_mask_store_ps(p, mask.mMask, mVec);
            }
            else {
                alignas(64) float raw[8];
                //_mm512_mask_store_ps(p, mask.mMask, mVec);
            }
            return p;
        }

        // STOREA  - Store vector content into aligned memory
        UME_FORCE_INLINE float* storea(float* p) {
            _mm512_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<16> const & mask, float * p) const {
            _mm512_mask_store_ps(p, mask.mMask, mVec);
            return p;
        }
        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __m512 t0 = _mm512_add_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm512_add_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA   - Masked add with vector and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA   - Masked add with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
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
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_add_ps(t0, t1);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<16> const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_add_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<16> const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            return SIMDVec_f(_mm512_sub_ps(mVec, b.mVec));
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // SUBS       - Sub with scalar
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            return SIMDVec_f(_mm512_sub_ps(mVec, _mm512_set1_ps(b)));
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm512_sub_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
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
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            return SIMDVec_f(_mm512_sub_ps(a.mVec, mVec));
        }
        // MSUBFROMV  - Masked sub from vector
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<16> const & mask, SIMDVec_f const & a) const {
            __m512 t0 = _mm512_mask_sub_ps(a.mVec, mask.mMask, a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMS   - Sub from scalar (promoted to vector)
        UME_FORCE_INLINE SIMDVec_f subfrom(float a) const {
            return SIMDVec_f(_mm512_sub_ps(_mm512_set1_ps(a), mVec));
        }
        // MSUBFROMS  - Masked sub from scalar (promoted to vector)
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<16> const & mask, float a) const {
            __m512 t0 = _mm512_set1_ps(a);
            __m512 t1 = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return SIMDVec_f(t1);
        }
        // SUBFROMVA  - Sub from vector and assign
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = _mm512_sub_ps(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA - Masked sub from vector and assign
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, SIMDVec_f const & a) {
            mVec = _mm512_mask_sub_ps(a.mVec, mask.mMask, a.mVec, mVec);
            return *this;
        }
        // SUBFROMSA  - Sub from scalar (promoted to vector) and assign
        UME_FORCE_INLINE SIMDVec_f subfroma(float a) {
            mVec = _mm512_sub_ps(_mm512_set1_ps(a), mVec);
            return *this;
        }
        // MSUBFROMSA - Masked sub from scalar (promoted to vector) and assign
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, float a) {
            __m512 t0 = _mm512_set1_ps(a);
            mVec = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_sub_ps(mVec, t0);
            return t1;
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<16> const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            __m512 t1 = mVec;
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return t1;
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_sub_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<16> const & mask) {
            __m512 t0 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV  - Masked multiplication with vector
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __m512 t0 = _mm512_mul_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm512_mul_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA - Masked multiplication with vector and assign
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mul_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA - Masked multiplication with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV  - Masked division with vector
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __m512 t0 = _mm512_div_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS  - Masked division with scalar
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // DIVVA  - Division with vector and assign
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm512_div_ps(mVec, b.mVec);
            return *this;
        }
        // MDIVVA - Masked division with vector and assign
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA  - Division with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            mVec = _mm512_div_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MDIVSA - Masked division with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RCP    - Reciprocal
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m512 t0 = _mm512_rcp23_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MRCP   - Masked reciprocal
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_rcp23_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // RCPS   - Reciprocal with scalar numerator
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            __m512 t2 = _mm512_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS  - Masked reciprocal with scalar
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            __m512 t2 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return SIMDVec_f(t2);
        }
        // RCPA   - Reciprocal and assign
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = _mm512_rcp23_ps(mVec);
            return *this;
        }
        // MRCPA  - Masked reciprocal and assign
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_rcp23_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RCPSA  - Reciprocal with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            mVec = _mm512_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA - Masked reciprocal with scalar and assign
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_rcp23_ps(mVec);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return *this;
        }

        //(Comparison operations)
        // CMPEQV - Element-wise 'equal' with vector
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPEQS - Element-wise 'equal' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(float b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, _mm512_set1_ps(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPNEV - Element-wise 'not equal' with vector
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmpneq_ps_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPNES - Element-wise 'not equal' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(float b) const {
            __mmask16 m0 = _mm512_cmpneq_ps_mask(mVec, _mm512_set1_ps(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPGTV - Element-wise 'greater than' with vector
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(SIMDVec_f const & b) const {
            //__mmask16 m0 = _mm512_cmpgt_ps_mask(mVec, b.mVec);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, b.mVec, 14);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPGTS - Element-wise 'greater than' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(float b) const {
            //__mmask16 m0 = _mm512_cmpgt_ps_mask(mVec, _mm512_set1_ps(b));
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 14);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPLTV - Element-wise 'less than' with vector
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmplt_ps_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPLTS - Element-wise 'less than' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(float b) const {
            __mmask16 m0 = _mm512_cmplt_ps_mask(mVec, _mm512_set1_ps(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPGEV - Element-wise 'greater than or equal' with vector
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(SIMDVec_f const & b) const {
            //__mmask16 m0 = _mm512_cmpge_ps_mask(mVec, b.mVec);
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, b.mVec, 13);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPGES - Element-wise 'greater than or equal' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(float b) const {
            //__mmask16 m0 = _mm512_cmpge_ps_mask(mVec, _mm512_set1_ps(b));
            __mmask16 m0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 13);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPLEV - Element-wise 'less than or equal' with vector
        UME_FORCE_INLINE SIMDVecMask<16> cmple(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmple_ps_mask(mVec, b.mVec);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPLES - Element-wise 'less than or equal' with scalar
        UME_FORCE_INLINE SIMDVecMask<16> cmple(float b) const {
            __mmask16 m0 = _mm512_cmple_ps_mask(mVec, _mm512_set1_ps(b));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // CMPEV  - Check if vectors are exact (returns scalar 'bool')
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, b.mVec);
            return m0 == 0xFF;
        }
        // CMPES - Check if all vector elements are equal to scalar value
        UME_FORCE_INLINE bool cmpe(float b) const {
            __mmask16 m0 = _mm512_cmpeq_ps_mask(mVec, _mm512_set1_ps(b));
            return m0 == 0xFF;
        }

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
        // HADD
        UME_FORCE_INLINE float hadd() const {
            alignas(64) float raw[16];
            float    retval = 0.0f;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                retval += raw[i];
            }
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<16> const & mask) const {
            alignas(64) float raw[16];
            float retval = 0.0f;
            _mm512_store_ps(raw, mVec);
            for (unsigned int i = 0; i < 16; i++) {
                if (mask.mMask & (1 << i)) retval += raw[i];
            }
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
            alignas(64) float raw[16];
            float    retval = b;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                retval += raw[i];
            }
            return retval;
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<16> const & mask, float b) const {
            alignas(64) float raw[16];
            float retval = b;
            _mm512_store_ps(raw, mVec);
            for (unsigned int i = 0; i < 16; i++) {
                if (mask.mMask & (1 << i)) retval += raw[i];
            }
            return retval;
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
            alignas(64) float raw[16];
            float retval = 1.0f;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                retval *= raw[i];
            }
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<16> const & mask) const {
            alignas(64) float raw[16];
            float retval = 1.0f;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                if (mask.mMask & (1 << i)) retval *= raw[i];
            }
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
            alignas(64) float raw[16];
            float retval = b;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                retval *= raw[i];
            }
            return retval;
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<16> const & mask, float b) const {
            alignas(64) float raw[16];
            float retval = b;
            _mm512_store_ps(raw, mVec);
            for (int i = 0; i < 16; i++) {
                if (mask.mMask & (1 << i)) retval *= raw[i];
            }
            return retval;
        }

        //(Fused arithmetics)
        // FMULADDV  - Fused multiply and add (A*B + C) with vectors
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmadd_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV - Masked fused multiply and add (A*B + C) with vectors
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FMULSUBV  - Fused multiply and sub (A*B - C) with vectors
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmsub_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV - Masked fused multiply and sub (A*B - C) with vectors
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FADDMULV  - Fused add and multiply ((A + B)*C) with vectors
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV - Masked fused add and multiply ((A + B)*C) with vectors
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // FSUBMULV  - Fused sub and multiply ((A - B)*C) with vectors
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_sub_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV - Masked fused sub and multiply ((A - B)*C) with vectors
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }

        // (Mathematical operations)
        // MAXV   - Max with vector
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_gmax_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV  - Masked max with vector
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MAXS   - Max with scalar
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            __m512 t0 = _mm512_gmax_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MMAXS  - Masked max with scalar
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MAXVA  - Max with vector and assign
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm512_gmax_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA - Masked max with vector and assign
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA  - Max with scalar (promoted to vector) and assign
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            mVec = _mm512_gmax_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MMAXSA - Masked max with scalar (promoted to vector) and assign
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<16> const & mask, float b) {
            mVec = _mm512_mask_gmax_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MINV   - Min with vector
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_gmin_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV  - Masked min with vector
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MINS   - Min with scalar (promoted to vector)
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            __m512 t0 = _mm512_gmin_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MMINS  - Masked min with scalar (promoted to vector)
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MINVA  - Min with vector and assign
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm512_gmin_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA - Masked min with vector and assign
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA  - Min with scalar (promoted to vector) and assign
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            mVec = _mm512_gmin_ps(mVec, _mm512_set1_ps(b));
            return *this;
        }
        // MMINSA - Masked min with scalar (promoted to vector) and assign
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<16> const & mask, float b) {
            mVec = _mm512_mask_gmin_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return *this;
        }
        // HMAX   - Max of elements of a vector (horizontal max)
        UME_FORCE_INLINE float hmax() const {
            return _mm512_reduce_gmax_ps(mVec);
        }
        // MHMAX  - Masked max of elements of a vector (horizontal max)
        UME_FORCE_INLINE float hmax(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_gmax_ps(mask.mMask, mVec);
        }
        // IMAX   - Index of max element of a vector
        // MIMAX  - Masked index of max element of a vector
        // HMIN   - Min of elements of a vector (horizontal min)
        UME_FORCE_INLINE float hmin() const {
            return _mm512_reduce_gmin_ps(mVec);
        }
        // MHMIN  - Masked min of elements of a vector (horizontal min)
        UME_FORCE_INLINE float hmin(SIMDVecMask<16> const & mask) const {
            return _mm512_mask_reduce_gmin_ps(mask.mMask, mVec);
        }
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

        // 3) Operations available for Signed integer and floating point SIMD types:

        // (Sign modification)
        // NEG   - Negate signed values
        UME_FORCE_INLINE SIMDVec_f operator-  () const {
            return neg();
        }
        // MNEG  - Masked negate signed values
        // NEGA  - Negate signed values and assign
        // MNEGA - Masked negate signed values and assign

        // (Mathematical functions)
        // ABS   - Absolute value
        UME_FORCE_INLINE SIMDVec_f abs() const {
            return SIMDVec_f(_mm512_abs_ps(mVec));
        }
        // MABS  - Masked absolute value
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // ABSA  - Absolute value and assign
        UME_FORCE_INLINE SIMDVec_f & abs() {
            mVec = _mm512_abs_ps(mVec);
            return *this;
        }
        // MABSA - Masked absolute value and assign
        UME_FORCE_INLINE SIMDVec_f & abx(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return *this;
        }

        // 4) Operations available for floating point SIMD types:

        // (Comparison operations)
        // CMPEQRV - Compare 'Equal within range' with margins from vector
        // CMPEQRS - Compare 'Equal within range' with scalar margin

        // (Mathematical functions)
        // SQR       - Square of vector values
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m512 t0 = _mm512_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR      - Masked square of vector values
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SQRA      - Square of vector values and assign
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm512_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA     - Masked square of vector values and assign
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT      - Square root of vector values
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            return SIMDVec_f(_mm512_sqrt_ps(mVec));
        }
        // MSQRT     - Masked square root of vector values 
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // SQRTA     - Square root of vector values and assign
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm512_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA    - Masked square root of vector values and assign
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RSQRT     - Reciprocal square root
        UME_FORCE_INLINE SIMDVec_f rsqr() const {
            return SIMDVec_f(_mm512_rsqrt23_ps(mVec));
        }
        // MRSQRT    - Masked reciprocal square root
        UME_FORCE_INLINE SIMDVec_f rsqrt(SIMDVecMask<16> const & mask) const {
            return SIMDVec_f(_mm512_mask_rsqrt23_ps(mVec, mask.mMask, mVec));
        }
        // RSQRTA    - Reciprocal square root and assign
        UME_FORCE_INLINE SIMDVec_f & rsqrta() {
            mVec = _mm512_rsqrt23_ps(mVec);
            return *this;
        }
        // MRSQRTA   - Masked reciprocal square root and assign
        UME_FORCE_INLINE SIMDVec_f & rsqrta(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_rsqrt23_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV      - Power (exponents in vector)
        // MPOWV     - Masked power (exponents in vector)
        // POWS      - Power (exponent in scalar)
        // MPOWS     - Masked power (exponent in scalar) 
        // ROUND     - Round to nearest integer
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m512 t0 = _mm512_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // MROUND    - Masked round to nearest integer
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_round_ps(mVec, mask.mMask, mVec, _MM_FROUND_TO_NEAREST_INT, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // TRUNC     - Truncate to integer (returns Signed integer vector)
        // MTRUNC    - Masked truncate to integer (returns Signed integer vector)
        // FLOOR     - Floor
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m512 t0 = _mm512_round_ps(mVec, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // MFLOOR    - Masked floor
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_round_ps(mVec, mask.mMask, mVec, _MM_FROUND_TO_NEG_INF, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // CEIL      - Ceil
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m512 t0 = _mm512_round_ps(mVec, _MM_FROUND_TO_POS_INF, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // MCEIL     - Masked ceil
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_round_ps(mVec, mask.mMask, mVec, _MM_FROUND_TO_POS_INF, _MM_EXPADJ_NONE);
            return SIMDVec_f(t0);
        }
        // ISFIN     - Is finite
        // ISINF     - Is infinite (INF)
        UME_FORCE_INLINE SIMDVecMask<16> isinf() const {
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_slli_epi32(t0, 1);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(t1, _mm512_set1_epi32(0xFF000000));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        // ISAN      - Is a number
        // ISNAN     - Is 'Not a Number (NaN)'
        UME_FORCE_INLINE SIMDVecMask<16> isnan() const {
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_slli_epi32(t0, 1);
            __m512i t2 = _mm512_set1_epi32(0xFF000000);
            __m512i t3 = _mm512_and_epi32(t1, t2);
            __m512i t4 = _mm512_andnot_epi32(t1, t2);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(t3, t2);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(t4, _mm512_set1_epi32(0));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m0 && m1;
            return ret_mask;
        }
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
