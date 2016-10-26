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

#if defined UME_USE_MASK_64B
#define BLEND(a_128, b_128, mask_256i) \
    _mm_blendv_ps( \
        a_128, \
        b_128, \
        _mm_castsi128_ps( \
            _mm256_castsi256_si128( \
                _mm256_permutevar8x32_epi32( \
                    mask_256i, \
                    _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)))))

    #define MASK_STORE(f32_addr, mask_256i, a_128) \
        _mm_maskstore_ps( \
            f32_addr, \
            _mm256_extractf128_si256( \
                _mm256_permutevar8x32_epi32( \
                    mask_256i, \
                    _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0)), \
                0), \
            a_128 \
            )
#else
    #define BLEND(a_128, b_128, mask_128i) _mm_blendv_ps(a_128, b_128, _mm_castsi128_ps(mask_128i))
    #define MASK_STORE(f32_addr, mask_128i, a_128) _mm_maskstore_ps(f32_addr, mask_128i, a_128)
#endif

namespace UME {
namespace SIMD {

    template<> class SIMDVec_f<double, 4>;

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
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2 >>
    {
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_i<int32_t, 4>;

        friend class SIMDVec_f<float, 8>;
    private:
        __m128 mVec;

        UME_FORCE_INLINE SIMDVec_f(__m128 const & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec = _mm_set1_ps(f);
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
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const * p) {
            mVec = _mm_loadu_ps(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3) {
            mVec = _mm_setr_ps(f0, f1, f2, f3);
        }
        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm_load_ps(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec = _mm_set1_ps(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
            mVec = _mm_loadu_ps(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<4> const & mask, float const * p) {
            __m128 t0 = _mm_loadu_ps(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec = _mm_load_ps(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const * p) {
            __m128 t0 = _mm_load_ps(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float* p) const {
            _mm_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<4> const & mask, float * p) const {
            MASK_STORE(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float * p) const {
            _mm_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<4> const & mask, float * p) const {
            MASK_STORE(p, mask.mMask, mVec);
            return p;
        }

        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm_add_ps(this->mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            mVec = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            mVec = BLEND(mVec, t0, mask.mMask);
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
            __m128 t0 = mVec;
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
            __m128 t1 = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-=(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            mVec = BLEND(mVec, t0, mask.mMask);
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
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            __m128 t1 = BLEND(b.mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_sub_ps(t0, mVec);
            __m128 t2 = BLEND(t0, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            mVec = BLEND(b.mVec, t0, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_sub_ps(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_sub_ps(t0, mVec);
            mVec = BLEND(t0, t1, mask.mMask);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m128 t0 = mVec;
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
            __m128 t1 = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            __m128 t2 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm_mul_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm_div_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_div_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m128 t0 = _mm_rcp_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            __m128 t3 = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t3);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = _mm_rcp_ps(mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_rcp_ps(mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            mVec = BLEND(mVec, t2, mask.mMask);
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            __m128i m0 = _mm_castps_si128(_mm_cmpeq_ps(mVec, b.mVec));
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128i m0 = _mm_castps_si128(_mm_cmpeq_ps(mVec, t0));
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            __m128i m0 = _mm_castps_si128(_mm_cmpneq_ps(mVec, b.mVec));
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128i m0 = _mm_castps_si128(_mm_cmpneq_ps(mVec, t0));
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            __m128i m0 = _mm_castps_si128(_mm_cmpgt_ps(mVec, b.mVec));
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128i m0 = _mm_castps_si128(_mm_cmpgt_ps(mVec, t0));
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            __m128 t0 = _mm_cmplt_ps(mVec, b.mVec);
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(float b) const {
            __m128 t0 = _mm_cmplt_ps(mVec, _mm_set1_ps(b));
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            __m128 t0 = _mm_cmpge_ps(mVec, b.mVec);
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(float b) const {
            __m128 t0 = _mm_cmpge_ps(mVec, _mm_set1_ps(b));
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            __m128 t0 = _mm_cmple_ps(mVec, b.mVec);
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(float b) const {
            __m128 t0 = _mm_cmple_ps(mVec, _mm_set1_ps(b));
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            alignas(16) uint32_t raw[4];
            __m128 m0 = _mm_cmpeq_ps(mVec, b.mVec);
            _mm_store_si128((__m128i*)raw, _mm_castps_si128(m0));
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            alignas(16) uint32_t raw[4];
            __m128 m0 = _mm_cmpeq_ps(mVec, _mm_set1_ps(b));
            _mm_store_si128((__m128i*)raw, _mm_castps_si128(m0));
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0);
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = BLEND(mVec, b.mVec, mask.mMask);
            return SIMDVec_f(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = BLEND(mVec, _mm_set1_ps(b), mask.mMask);
            return SIMDVec_f(t0);
        }
        // HADD
        UME_FORCE_INLINE float hadd() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<4> const & mask) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(_mm_set1_ps(0.0f), mVec, mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<4> const & mask, float b) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(_mm_set1_ps(0.0f), mVec, mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<4> const & mask) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(_mm_set1_ps(1.0f), mVec, mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3] * b;
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<4> const & mask, float b) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(_mm_set1_ps(1.0f), mVec, mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3] * b;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m128 t0 = _mm_fmadd_ps(mVec, b.mVec, c.mVec);
#else
            __m128 t0 = _mm_add_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m128 t0 = _mm_fmadd_ps(mVec, b.mVec, c.mVec);
#else
            __m128 t0 = _mm_add_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
#endif
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_sub_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_sub_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_add_ps(mVec, b.mVec), c.mVec);
            return SIMDVec_f(t0);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_add_ps(mVec, b.mVec), c.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_sub_ps(mVec, b.mVec), c.mVec);
            return SIMDVec_f(t0);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_sub_ps(mVec, b.mVec), c.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_min_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_min_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            float t1 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<4> const & mask) const {
            alignas(16) float raw[4];
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::min());
            __m128 t1 = BLEND(t0, mVec, mask.mMask);
            _mm_store_ps(raw, t1);
            float t2 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            float t3 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t2 > t3 ? t2 : t3;
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE float hmin() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            float t1 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<4> const & mask) const {
            alignas(16) float raw[4];
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::max());
            __m128 t1 = BLEND(t0, mVec, mask.mMask);
            _mm_store_ps(raw, t1);
            float t2 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            float t3 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t2 < t3 ? t2 : t3;
        }
        // IMIN
        // MIMIN

        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
            __m128i t0 = _mm_load_si128((__m128i*)indices);
            mVec = _mm_i32gather_ps((const float *)baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, uint32_t const * indices) {
            __m128i t0 = _mm_load_si128((__m128i*)indices);
            __m128 t1 = _mm_i32gather_ps((const float *)baseAddr, t0, 4);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            mVec = _mm_i32gather_ps((const float *)baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            __m128 t0 = _mm_i32gather_ps((const float *)baseAddr, indices.mVec, 4);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE float* scatter(float* baseAddr, uint32_t* indices) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            for (int i = 0; i < 4; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t rawMask[4];
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
#else
            alignas(16) uint32_t rawMask[4];
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
#endif
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            for (int i = 0; i < 4; i++) { if (rawMask[i] == SIMDVecMask<4>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            alignas(16) float raw[4];
            alignas(16) uint32_t rawIndices[4];
            _mm_store_ps(raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            for (int i = 0; i < 4; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t rawMask[4];
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
#else
            alignas(16) uint32_t rawMask[4];
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
#endif
            alignas(16) float raw[4];
            alignas(16) uint32_t rawIndices[4];
            _mm_store_ps(raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            for (int i = 0; i < 4; i++) {
                if (rawMask[i] == SIMDVecMask<4>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            __m128 t2 = _mm_and_ps(t1, mVec);
            return SIMDVec_f(t2);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            __m128 t2 = _mm_and_ps(t1, mVec);
            __m128 t3 = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t3);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            mVec = _mm_and_ps(t1, mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            __m128 t2 = _mm_and_ps(t1, mVec);
            mVec = BLEND(mVec, t2, mask.mMask);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m128 t0 = _mm_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_sqrt_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_sqrt_ps(mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // TRUNC
        SIMDVec_i<int32_t, 4> trunc() const {
            __m128i t0 = _mm_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // MTRUNC
        SIMDVec_i<int32_t, 4> trunc(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_setzero_ps();
            __m128i t1 = _mm_cvttps_epi32(BLEND(t0, mVec, mask.mMask));
            return SIMDVec_i<int32_t, 4>(t1);
        }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m128 t0 = _mm_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_floor_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m128 t0 = _mm_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_ceil_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ISFIN
        // ISINF
        // ISAN
        // ISNAN
        // ISSUB
        // ISZERO
        // ISZEROSUB
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 4>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<4> const & mask) const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 4>, SIMDVecMask<4>>(mask, *this);
        }
        // LOG
        // MLOG
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_ps(raw);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t mask_raw[4] = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0 };
            __m256i t0 = _mm256_load_si256((__m256i*)mask_raw);
#else
            alignas(16) uint32_t mask_raw[4] = { 0xFFFFFFFF, 0xFFFFFFFF, 0, 0 };
            __m128i t0 = _mm_load_si128((__m128i*)mask_raw);
#endif
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], 0.0f, 0.0f };
            __m128 t1 = _mm_load_ps(raw);
            mVec = BLEND(mVec, t1, t0);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t mask_raw[4] = { 0, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF };
            __m256i t0 = _mm256_load_si256((__m256i*)mask_raw);
#else
            alignas(16) uint32_t mask_raw[4] = { 0, 0, 0xFFFFFFFF, 0xFFFFFFFF };
            __m128i t0 = _mm_load_si128((__m128i*)mask_raw);
#endif
            alignas(16) float raw[4] = { 0.0f, 0.0f, b.mVec[0], b.mVec[1] };
            __m128 t1 = _mm_load_ps(raw);
            mVec = BLEND(mVec, t1, t0);
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 2> & a, SIMDVec_f<float, 2> & b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 2> unpacklo() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return SIMDVec_f<float, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 2> unpackhi() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return SIMDVec_f<float, 2>(raw[2], raw[3]);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 4>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 4>() const;
    };
}
}

#undef BLEND
#undef MASK_STORE

#endif
