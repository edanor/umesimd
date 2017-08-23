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

    template<> class SIMDVec_f<double, 16>;

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
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_i<int32_t, 16>;

        friend class SIMDVec_f<float, 32>;
    private:
        __m512 mVec;

        UME_FORCE_INLINE SIMDVec_f(__m512 const & x) {
            this->mVec = x;
        }

    public:
        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
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
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const *p) { this->load(p); }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f0,  float f1,  float f2,  float f3,
                         float f4,  float f5,  float f6,  float f7, 
                         float f8,  float f9,  float f10, float f11,
                         float f12, float f13, float f14, float f15) {
            mVec = _mm512_setr_ps(f0, f1, f2,  f3,  f4,  f5,  f6,  f7,
                                  f8, f9, f10, f11, f12, f13, f14, f15);
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
            mVec = _mm512_loadu_ps(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<16> const & mask, float const * p) {
            mVec = _mm512_mask_loadu_ps(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec = _mm512_load_ps(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<16> const & mask, float const * p) {
            mVec = _mm512_mask_loadu_ps(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float * p) const {
            _mm512_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float * store(SIMDVecMask<16> const & mask, float * p) const {
            _mm512_mask_storeu_ps(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float * p) const {
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
            __m512 t0 = _mm512_add_ps(this->mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm512_add_ps(this->mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            mVec = _mm512_add_ps(this->mVec, _mm512_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, float b) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(b));
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            return add(b);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            return add(mask, b);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(float b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<16> const & mask, float b) const {
            return add(mask, b);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            return adda(b);
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            return adda(mask, b);
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(float b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<16> const & mask, float b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __m512 t0 = mVec;
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            __m512 t0 = mVec;
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<16> const & mask) {
            __m512 t0 = mVec;
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            mVec = _mm512_add_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_add_ps(mVec, mask.mMask, mVec, _mm512_set1_ps(1.0f));
            return *this;
        }

        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __m512 t0 = _mm512_sub_ps(mVec, _mm512_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
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
        UME_FORCE_INLINE SIMDVec_f & operator-=(SIMDVec_f const & b) {
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
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            return sub(mask, b);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(float b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<16> const & mask, float b) const {
            return sub(mask, b);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            return suba(mask, b);
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(float b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<16> const & mask, float b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            __m512 t0 = _mm512_sub_ps(_mm512_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return SIMDVec_f(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm512_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_sub_ps(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            mVec = _mm512_sub_ps(_mm512_set1_ps(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_sub_ps(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m512 t0 = mVec;
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            __m512 t0 = mVec;
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<16> const & mask) {
            __m512 t0 = mVec;
            __m512 t1 = _mm512_set1_ps(1.0f);
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, t1);
            return SIMDVec_f(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            mVec = _mm512_sub_ps(mVec, _mm512_set1_ps(1.0f));
            return *this;
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
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
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
        // MMULVA
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
        // MMULSA
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
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm512_div_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_div_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_div_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m512 t0 = _mm512_rcp14_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            __m512 t0 = _mm512_rcp14_ps(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m512 t1 = _mm512_set1_ps(b);
            __m512 t2 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return SIMDVec_f(t2);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = _mm512_rcp14_ps(mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            __m512 t0 = _mm512_rcp14_ps(mVec);
            __m512 t1 = _mm512_set1_ps(b);
            mVec = _mm512_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_mask_rcp14_ps(mVec, mask.mMask, mVec);
            __m512 t1 = _mm512_set1_ps(b);
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, t0, t1);
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 0);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 0);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 12);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 12);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 30);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 30);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 17);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 17);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 29);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 29);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<16> cmple(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 18);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<16> cmple(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 18);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, b.mVec, 0);
            return (t0 == 0xFFFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            __mmask16 t0 = _mm512_cmp_ps_mask(mVec, _mm512_set1_ps(b), 0);
            return (t0 == 0xFFFF);
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_mov_ps(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // HADD
        UME_FORCE_INLINE float hadd() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                   raw[8] + raw[9] + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            float retval = _mm512_reduce_add_ps(mVec);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 = 0;
            if (mask.mMask & 0x0001) t0 += raw[0];
            if (mask.mMask & 0x0002) t0 += raw[1];
            if (mask.mMask & 0x0004) t0 += raw[2];
            if (mask.mMask & 0x0008) t0 += raw[3];
            if (mask.mMask & 0x0010) t0 += raw[4];
            if (mask.mMask & 0x0020) t0 += raw[5];
            if (mask.mMask & 0x0040) t0 += raw[6];
            if (mask.mMask & 0x0080) t0 += raw[7];
            if (mask.mMask & 0x0100) t0 += raw[8];
            if (mask.mMask & 0x0200) t0 += raw[9];
            if (mask.mMask & 0x0400) t0 += raw[10];
            if (mask.mMask & 0x0800) t0 += raw[11];
            if (mask.mMask & 0x1000) t0 += raw[12];
            if (mask.mMask & 0x2000) t0 += raw[13];
            if (mask.mMask & 0x4000) t0 += raw[14];
            if (mask.mMask & 0x8000) t0 += raw[15];
            return t0;
#else
            float retval = _mm512_mask_reduce_add_ps(mask.mMask, mVec);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return b + raw[0] + raw[1] + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                       raw[8] + raw[9] + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            float retval = _mm512_reduce_add_ps(mVec);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<16> const & mask, float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 = b;
            if (mask.mMask & 0x0001) t0 += raw[0];
            if (mask.mMask & 0x0002) t0 += raw[1];
            if (mask.mMask & 0x0004) t0 += raw[2];
            if (mask.mMask & 0x0008) t0 += raw[3];
            if (mask.mMask & 0x0010) t0 += raw[4];
            if (mask.mMask & 0x0020) t0 += raw[5];
            if (mask.mMask & 0x0040) t0 += raw[6];
            if (mask.mMask & 0x0080) t0 += raw[7];
            if (mask.mMask & 0x0100) t0 += raw[8];
            if (mask.mMask & 0x0200) t0 += raw[9];
            if (mask.mMask & 0x0400) t0 += raw[10];
            if (mask.mMask & 0x0800) t0 += raw[11];
            if (mask.mMask & 0x1000) t0 += raw[12];
            if (mask.mMask & 0x2000) t0 += raw[13];
            if (mask.mMask & 0x4000) t0 += raw[14];
            if (mask.mMask & 0x8000) t0 += raw[15];
            return t0;
#else
            float retval = _mm512_mask_reduce_add_ps(mask.mMask, mVec);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[9] * raw[9] * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            float retval = _mm512_reduce_mul_ps(mVec);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 = 1;
            if (mask.mMask & 0x0001) t0 *= raw[0];
            if (mask.mMask & 0x0002) t0 *= raw[1];
            if (mask.mMask & 0x0004) t0 *= raw[2];
            if (mask.mMask & 0x0008) t0 *= raw[3];
            if (mask.mMask & 0x0010) t0 *= raw[4];
            if (mask.mMask & 0x0020) t0 *= raw[5];
            if (mask.mMask & 0x0040) t0 *= raw[6];
            if (mask.mMask & 0x0080) t0 *= raw[7];
            if (mask.mMask & 0x0100) t0 *= raw[8];
            if (mask.mMask & 0x0200) t0 *= raw[9];
            if (mask.mMask & 0x0400) t0 *= raw[10];
            if (mask.mMask & 0x0800) t0 *= raw[11];
            if (mask.mMask & 0x1000) t0 *= raw[12];
            if (mask.mMask & 0x2000) t0 *= raw[13];
            if (mask.mMask & 0x4000) t0 *= raw[14];
            if (mask.mMask & 0x8000) t0 *= raw[15];
            return t0;
#else
            float retval = _mm512_mask_reduce_mul_ps(mask.mMask, mVec);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            return b * raw[0] * raw[1] * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[9] * raw[9] * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            float retval = b;
            retval *= _mm512_reduce_mul_ps(mVec);
            return retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<16> const & mask, float b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 = b;
            if (mask.mMask & 0x0001) t0 *= raw[0];
            if (mask.mMask & 0x0002) t0 *= raw[1];
            if (mask.mMask & 0x0004) t0 *= raw[2];
            if (mask.mMask & 0x0008) t0 *= raw[3];
            if (mask.mMask & 0x0010) t0 *= raw[4];
            if (mask.mMask & 0x0020) t0 *= raw[5];
            if (mask.mMask & 0x0040) t0 *= raw[6];
            if (mask.mMask & 0x0080) t0 *= raw[7];
            if (mask.mMask & 0x0100) t0 *= raw[8];
            if (mask.mMask & 0x0200) t0 *= raw[9];
            if (mask.mMask & 0x0400) t0 *= raw[10];
            if (mask.mMask & 0x0800) t0 *= raw[11];
            if (mask.mMask & 0x1000) t0 *= raw[12];
            if (mask.mMask & 0x2000) t0 *= raw[13];
            if (mask.mMask & 0x4000) t0 *= raw[14];
            if (mask.mMask & 0x8000) t0 *= raw[15];
            return t0;
#else
            float retval = b;
            retval *= _mm512_mask_reduce_mul_ps(mask.mMask, mVec);
            return retval;
#endif
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmadd_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_fmadd_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_fmsub_ps(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_fmsub_ps(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_add_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_add_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_sub_ps(mVec, b.mVec);
            __m512 t1 = _mm512_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, mVec, b.mVec);
            __m512 t1 = _mm512_mask_mul_ps(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_max_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm512_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_max_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_max_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m512 t0 = _mm512_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512 t0 = _mm512_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<16> const & mask, float b) const {
            __m512 t0 = _mm512_set1_ps(b);
            __m512 t1 = _mm512_mask_min_ps(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm512_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_min_ps(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<16> const & mask, float b) {
            __m512 t0 = _mm512_set1_ps(b);
            mVec = _mm512_mask_min_ps(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 = raw[0] > raw[1] ? raw[0] : raw[1];
            float t1 = raw[2] > raw[3] ? raw[2] : raw[3];
            float t2 = raw[4] > raw[5] ? raw[4] : raw[5];
            float t3 = raw[6] > raw[7] ? raw[6] : raw[7];
            float t4 = raw[8] > raw[9] ? raw[8] : raw[9];
            float t5 = raw[10] > raw[11] ? raw[10] : raw[11];
            float t6 = raw[12] > raw[13] ? raw[12] : raw[13];
            float t7 = raw[14] > raw[15] ? raw[14] : raw[15];
            float t8 = t0 > t1 ? t0 : t1;
            float t9 = t2 > t3 ? t2 : t3;
            float t10 = t4 > t5 ? t4 : t5;
            float t11 = t6 > t7 ? t6 : t7;
            float t12 = t8 > t9 ? t8 : t9;
            float t13 = t10 > t11 ? t10 : t11;
            return t12 > t13 ? t12 : t13;
#else
            float retval = _mm512_reduce_max_ps(mVec);
            return retval;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 =  ((mask.mMask & 0x0001) != 0) ? raw[0] : std::numeric_limits<float>::lowest();
            float t1 = (((mask.mMask & 0x0002) != 0) && raw[1] > t0) ? raw[1] : t0;
            float t2 = (((mask.mMask & 0x0004) != 0) && raw[2] > t1) ? raw[2] : t1;
            float t3 = (((mask.mMask & 0x0008) != 0) && raw[3] > t2) ? raw[3] : t2;
            float t4 = (((mask.mMask & 0x0010) != 0) && raw[4] > t3) ? raw[4] : t3;
            float t5 = (((mask.mMask & 0x0020) != 0) && raw[5] > t4) ? raw[5] : t4;
            float t6 = (((mask.mMask & 0x0040) != 0) && raw[6] > t5) ? raw[6] : t5;
            float t7 = (((mask.mMask & 0x0080) != 0) && raw[7] > t6) ? raw[7] : t6;
            float t8 = (((mask.mMask & 0x0100) != 0) && raw[8] > t7) ? raw[8] : t7;
            float t9 = (((mask.mMask & 0x0200) != 0) && raw[9] > t8) ? raw[9] : t8;
            float t10 = (((mask.mMask & 0x0400) != 0) && raw[10] > t9) ? raw[10] : t9;
            float t11 = (((mask.mMask & 0x0800) != 0) && raw[11] > t10) ? raw[11] : t10;
            float t12 = (((mask.mMask & 0x1000) != 0) && raw[12] > t11) ? raw[12] : t11;
            float t13 = (((mask.mMask & 0x2000) != 0) && raw[13] > t12) ? raw[13] : t12;
            float t14 = (((mask.mMask & 0x4000) != 0) && raw[14] > t13) ? raw[14] : t13;
            float t15 = (((mask.mMask & 0x8000) != 0) && raw[15] > t14) ? raw[15] : t14;
            return t15;
#else
            float retval = _mm512_mask_reduce_max_ps(mask.mMask, mVec);
            return retval;
#endif
        }
        // IMAX
        // HMIN
        UME_FORCE_INLINE float hmin() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 = raw[0] < raw[1] ? raw[0] : raw[1];
            float t1 = raw[2] < raw[3] ? raw[2] : raw[3];
            float t2 = raw[4] < raw[5] ? raw[4] : raw[5];
            float t3 = raw[6] < raw[7] ? raw[6] : raw[7];
            float t4 = raw[8] < raw[9] ? raw[8] : raw[9];
            float t5 = raw[10] < raw[11] ? raw[10] : raw[11];
            float t6 = raw[12] < raw[13] ? raw[12] : raw[13];
            float t7 = raw[14] < raw[15] ? raw[14] : raw[15];
            float t8 = t0 < t1 ? t0 : t1;
            float t9 = t2 < t3 ? t2 : t3;
            float t10 = t4 < t5 ? t4 : t5;
            float t11 = t6 < t7 ? t6 : t7;
            float t12 = t8 < t9 ? t8 : t9;
            float t13 = t10 < t11 ? t10 : t11;
            return t12 < t13 ? t12 : t13;
#else
            float retval = _mm512_reduce_min_ps(mVec);
            return retval;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            float t0 =  ((mask.mMask & 0x0001) != 0) ? raw[0] : std::numeric_limits<float>::max();
            float t1 = (((mask.mMask & 0x0002) != 0) && raw[1] < t0) ? raw[1] : t0;
            float t2 = (((mask.mMask & 0x0004) != 0) && raw[2] < t1) ? raw[2] : t1;
            float t3 = (((mask.mMask & 0x0008) != 0) && raw[3] < t2) ? raw[3] : t2;
            float t4 = (((mask.mMask & 0x0010) != 0) && raw[4] < t3) ? raw[4] : t3;
            float t5 = (((mask.mMask & 0x0020) != 0) && raw[5] < t4) ? raw[5] : t4;
            float t6 = (((mask.mMask & 0x0040) != 0) && raw[6] < t5) ? raw[6] : t5;
            float t7 = (((mask.mMask & 0x0080) != 0) && raw[7] < t6) ? raw[7] : t6;
            float t8 = (((mask.mMask & 0x0100) != 0) && raw[8] < t7) ? raw[8] : t7;
            float t9 = (((mask.mMask & 0x0200) != 0) && raw[9] < t8) ? raw[9] : t8;
            float t10 = (((mask.mMask & 0x0400) != 0) && raw[10] < t9) ? raw[10] : t9;
            float t11 = (((mask.mMask & 0x0800) != 0) && raw[11] < t10) ? raw[11] : t10;
            float t12 = (((mask.mMask & 0x1000) != 0) && raw[12] < t11) ? raw[12] : t11;
            float t13 = (((mask.mMask & 0x2000) != 0) && raw[13] < t12) ? raw[13] : t12;
            float t14 = (((mask.mMask & 0x4000) != 0) && raw[14] < t13) ? raw[14] : t13;
            float t15 = (((mask.mMask & 0x8000) != 0) && raw[15] < t14) ? raw[15] : t14;
            return t15;
#else
            float retval = _mm512_mask_reduce_min_ps(mask.mMask, mVec);
            return retval;
#endif
        }
        // IMIN
        // MIMIN
        // GATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(float const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            mVec = _mm512_i32gather_ps(t2, baseAddr, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(SIMDVecMask<16> const & mask, float const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            mVec = _mm512_mask_i32gather_ps(mVec, mask.mMask, t2, baseAddr, 4);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_i32gather_ps(t0, baseAddr, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<16> const & mask, float const * baseAddr, uint32_t const * indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_mask_i32gather_ps(mVec, m0, t0, baseAddr, 4);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 16> const & indices) {
            mVec = _mm512_i32gather_ps(indices.mVec, baseAddr, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<16> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 16> const & indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            mVec = _mm512_mask_i32gather_ps(mVec, m0, indices.mVec, baseAddr, 4);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE float* scatteru(float* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            _mm512_i32scatter_ps(baseAddr, t2, mVec, 4);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE float*  scatteru(SIMDVecMask<16> const & mask, float* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_mullo_epi32(t0, t1);
            _mm512_mask_i32scatter_ps(baseAddr, mask.mMask, t2, mVec, 4);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE float* scatter(float* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            _mm512_i32scatter_ps(baseAddr, t0, mVec, 4);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE float* scatter(SIMDVecMask<16> const & mask, float* baseAddr, uint32_t* indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __m512i t0 = _mm512_loadu_si512(indices);
            _mm512_mask_i32scatter_ps(baseAddr, m0, t0, mVec, 4);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 16> const & indices) {
            _mm512_i32scatter_ps(baseAddr, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE float* scatter(SIMDVecMask<16> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 16> const & indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            _mm512_mask_i32scatter_ps(baseAddr, m0, indices.mVec, mVec, 4);
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m512 t0 = _mm512_sub_ps(_mm512_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_sub_ps(mVec, mask.mMask, _mm512_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = _mm512_sub_ps(_mm512_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sub_ps(mVec, mask.mMask, _mm512_set1_ps(0.0f), mVec);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __m512 t3 = _mm512_castsi512_ps(t2);
            return SIMDVec_f(t3);
#else
            __m512 t0 = _mm512_abs_ps(mVec);
            return SIMDVec_f(t0);
#endif
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __m512 t3 = _mm512_castsi512_ps(t2);
            __m512 t4 = _mm512_mask_mov_ps(mVec, mask.mMask, t3);
            return SIMDVec_f(t4);
#else
            __m512 t0 = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
#endif
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            mVec = _mm512_castsi512_ps(t2);
            return *this;
#else
            mVec = _mm512_abs_ps(mVec);
            return *this;
#endif
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<16> const & mask) {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __m512 t3 = _mm512_castsi512_ps(t2);
            mVec    = _mm512_mask_mov_ps(mVec, mask.mMask, t3);
            return *this;
#else
            mVec = _mm512_mask_abs_ps(mVec, mask.mMask, mVec);
            return *this;
#endif
        }
        // CMPEQRV
        // CMPEQRS
        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m512 t0 = _mm512_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm512_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_mul_ps(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m512 t0 = _mm512_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm512_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<16> const & mask) {
            mVec = _mm512_mask_sqrt_ps(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m512 t0 = _mm512_roundscale_ps(mVec, 0);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_roundscale_ps(mVec, 0);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 16> trunc() const {
            __m512i t0 = _mm512_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 16>(t0);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 16> trunc(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_mask_cvttps_epi32(_mm512_setzero_epi32(), mask.mMask, mVec);
            return SIMDVec_i<int32_t, 16>(t0);
        }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m512 t0 = _mm512_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_floor_ps(mVec);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m512 t0 = _mm512_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<16> const & mask) const {
            __m512 t0 = _mm512_ceil_ps(mVec);
            __m512 t1 = _mm512_mask_mov_ps(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // ISFIN
        UME_FORCE_INLINE SIMDVecMask<16> isfin() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x08);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x10);
            __mmask16 t2 = (~t0) & (~t1);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t2;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpneq_epi32_mask(t2, t1);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t3;
            return ret_mask;
#endif
        }
        // ISINF
        UME_FORCE_INLINE SIMDVecMask<16> isinf() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x08);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x10);
            __mmask16 t2 = t0 | t1;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t2;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpeq_epi32_mask(t2, _mm512_set1_epi32(0x7F800000));
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t3;
            return ret_mask;
#endif
        }
        // ISAN
        UME_FORCE_INLINE SIMDVecMask<16> isan() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x01);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x80);
            __mmask16 t2 = (~t0) & (~t1);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t2;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpneq_epi32_mask(t2, t1);   // is finite

            __m512i t4 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t5 = _mm512_and_epi32(t4, t0);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t2, t1);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t9 = ~(t7 & t8);                         // is not NaN

            __mmask16 t10 = t3 & t9;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t10;
            return ret_mask;
#endif
        }
        // ISNAN
        UME_FORCE_INLINE SIMDVecMask<16> isnan() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x01);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x80);
            __mmask16 t2 = t0 | t1;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t2;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __m512i t3 = _mm512_set1_epi32(0xFF800000);
            __m512i t4 = _mm512_andnot_epi32(t3, t0);
            __m512i t5 = _mm512_setzero_epi32();
            __mmask16 t6 = _mm512_cmpeq_epi32_mask(t2, t1);
            __mmask16 t7 = _mm512_cmpneq_epi32_mask(t4, t5);
            __mmask16 t8 = t6 & t7;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t8;
            return ret_mask;
#endif
        }
        // ISNORM
        UME_FORCE_INLINE SIMDVecMask<16> isnorm() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = ~_mm512_fpclass_ps_mask(mVec, 0x01);
            __mmask16 t1 = ~_mm512_fpclass_ps_mask(mVec, 0x02);
            __mmask16 t2 = ~_mm512_fpclass_ps_mask(mVec, 0x04);
            __mmask16 t3 = ~_mm512_fpclass_ps_mask(mVec, 0x08);
            __mmask16 t4 = ~_mm512_fpclass_ps_mask(mVec, 0x10);
            __mmask16 t5 = ~_mm512_fpclass_ps_mask(mVec, 0x20);
            __mmask16 t6 = ~_mm512_fpclass_ps_mask(mVec, 0x80);
            __mmask16 t7 = t0 & t1 & t2 & t3 & t4 & t5 & t6;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t7;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpneq_epi32_mask(t2, t1);   // is not finite

            __m512i t4 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t5 = _mm512_and_epi32(t4, t0);
            __m512i t6 = _mm512_setzero_epi32();
            __mmask16 t7 = _mm512_cmpeq_epi32_mask(t2, t1);
            __mmask16 t8 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t9 = ~(t7 & t8);                         // is not NaN

            __mmask16 t10 = _mm512_cmpeq_epi32_mask(t2, t6);
            __mmask16 t11 = _mm512_cmpneq_epi32_mask(t5, t6);
            __mmask16 t12 = ~(t10 & t11);                      // is not subnormal

            __m512i t14 = _mm512_or_epi32(t2, t5);
            __mmask16 t15 = _mm512_cmpneq_epi32_mask(t6, t14);      // is not zero

            __mmask16 t16 = (t3 & t9 & t12 & t15);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t16;
            return ret_mask;
#endif
        }
        // ISSUB
        UME_FORCE_INLINE SIMDVecMask<16> issub() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x20);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t0;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __m512i t3 = _mm512_setzero_epi32();
            __mmask16 t4 = _mm512_cmpeq_epi32_mask(t2, t3);
            __m512i t5 = _mm512_set1_epi32(0x007FFFFF);
            __m512i t6 = _mm512_and_epi32(t0, t5);
            __mmask16 t7 = _mm512_cmpneq_epi32_mask(t6, t3);
            __mmask16 t8 = t4 & t7;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t8;
            return ret_mask;
#endif
        }
        // ISZERO
        UME_FORCE_INLINE SIMDVecMask<16> iszero() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x02);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x04);
            __mmask16 t2 = t0 | t1;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t2;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7FFFFFFF);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __mmask16 t3 = _mm512_cmpeq_epi32_mask(t2, _mm512_setzero_epi32());
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t3;
            return ret_mask;
#endif
        }
        // ISZEROSUB
        UME_FORCE_INLINE SIMDVecMask<16> iszerosub() const {
#if defined(__AVX512DQ__)
            __mmask16 t0 = _mm512_fpclass_ps_mask(mVec, 0x02);
            __mmask16 t1 = _mm512_fpclass_ps_mask(mVec, 0x04);
            __mmask16 t2 = _mm512_fpclass_ps_mask(mVec, 0x20);
            __mmask16 t3 = t0 | t1 | t2;
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t3;
            return ret_mask;
#else
            __m512i t0 = _mm512_castps_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(0x7F800000);
            __m512i t2 = _mm512_and_epi32(t0, t1);
            __m512i t3 = _mm512_setzero_epi32();
            __mmask16 t4 = _mm512_cmpeq_epi32_mask(t2, t3);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = t4;
            return ret_mask;
#endif
        }
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_exp_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 16>>(*this);
        #endif
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_mask_exp_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_log_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::logf<SIMDVec_f, SIMDVec_u<uint32_t, 16>>(*this);
        #endif
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_mask_log_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::logf<SIMDVec_f, SIMDVec_u<uint32_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_sin_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(*this);
        #endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_mask_sin_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_cos_ps(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(*this);
        #endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512 t0 = _mm512_mask_cos_ps(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) float raw_cos[16];
            sinvec.mVec = _mm512_sincos_ps((__m512*)raw_cos, mVec);
            cosvec.mVec = _mm512_load_ps(raw_cos);
        #else
            VECTOR_EMULATION::sincosf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(*this, sinvec, cosvec);
        #endif
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<16> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) float raw_cos[16]; // 64B aligned data for 512b vector operation is needed.
            sinvec.mVec = _mm512_mask_sincos_ps((__m512*)raw_cos, mVec, mVec, mask.mMask, mVec);
            cosvec.mVec = _mm512_load_ps(raw_cos);
        #else
            sinvec = VECTOR_EMULATION::sinf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(mask, *this);
            cosvec = VECTOR_EMULATION::cosf<SIMDVec_f, SIMDVec_i<int32_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN
        // PACK
        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 8> const & a, SIMDVec_f<float, 8> const & b) {
#if defined(__AVX512DQ__)
            mVec = _mm512_insertf32x8(mVec, a.mVec, 0);
            mVec = _mm512_insertf32x8(mVec, b.mVec, 1);
#else
            alignas(64) float raw[16];
            _mm256_store_ps(&raw[0], a.mVec);
            _mm256_store_ps(&raw[8], b.mVec);
            mVec = _mm512_load_ps(&raw[0]);
#endif
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 8> const & a) {
#if defined(__AVX512DQ__)
            mVec = _mm512_insertf32x8(mVec, a.mVec, 0);
#else
            alignas(64) float raw[16];
            _mm512_store_ps(&raw[0], mVec);
            _mm256_store_ps(&raw[0], a.mVec);
            mVec = _mm512_load_ps(&raw[0]);
#endif
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 8> const & b) {
#if defined(__AVX512VL__)
            mVec = _mm512_insertf32x8(mVec, b.mVec, 1);
#else
            alignas(64) float raw[16];
            _mm512_store_ps(&raw[0], mVec);
            _mm256_store_ps(&raw[8], b.mVec);
            mVec = _mm512_load_ps(&raw[0]);
#endif
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 8> & a, SIMDVec_f<float, 8> & b) const {
#if defined(__AVX512DQ__)
            a.mVec = _mm512_extractf32x8_ps(mVec, 0);
            b.mVec = _mm512_extractf32x8_ps(mVec, 1);
#else
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            a.mVec = _mm256_load_ps(&raw[0]);
            b.mVec = _mm256_load_ps(&raw[8]);
#endif
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 8> unpacklo() const {
#if defined(__AVX512DQ__)
            __m256 t0 = _mm512_extractf32x8_ps(mVec, 0);
#else
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            __m256 t0 = _mm256_load_ps(raw);
#endif
            return SIMDVec_f<float, 8>(t0);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 8> unpackhi() const {
#if defined(__AVX512DQ__)
            __m256 t0 = _mm512_extractf32x8_ps(mVec, 1);
#else
            alignas(64) float raw[16];
            _mm512_store_ps(raw, mVec);
            __m256 t0 = _mm256_load_ps(&raw[8]);
#endif
            return SIMDVec_f<float, 8>(t0);
        }

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
