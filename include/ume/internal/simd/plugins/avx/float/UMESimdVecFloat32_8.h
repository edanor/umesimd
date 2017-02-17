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

#ifndef UME_SIMD_VEC_FLOAT32_8_H_
#define UME_SIMD_VEC_FLOAT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define BLEND(a_256, b_256, mask_256i) _mm256_blendv_ps(a_256, b_256, _mm256_castsi256_ps(mask_256i))

namespace UME {
namespace SIMD {

    template<> class SIMDVec_f<double, 8>;

    template<>
    class SIMDVec_f<float, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_u<uint32_t, 8>,
            SIMDVec_i<int32_t, 8>,
            float,
            8,
            uint32_t,
            int32_t,
            SIMDVecMask<8>,
            SIMDSwizzle<8 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 8>,
            SIMDVec_f<float, 4 >>
    {
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_u<uint32_t, 8>;

        friend class SIMDVec_f<float, 16>;
    private:
        __m256 mVec;

        UME_FORCE_INLINE SIMDVec_f(__m256 const & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            mVec = _mm256_set1_ps(f);
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
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3, float f4, float f5, float f6, float f7) {
            mVec = _mm256_setr_ps(f0, f1, f2, f3, f4, f5, f6, f7);
        }
        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            alignas(32) float raw[8];
            _mm256_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_ps(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
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
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            mVec = _mm256_set1_ps(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const * p) {
            mVec = _mm256_loadu_ps(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<8> const & mask, float const * p) {
            __m256 t0 = _mm256_loadu_ps(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const * p) {
            mVec = _mm256_load_ps(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<8> const & mask, float const * p) {
            __m256 t0 = _mm256_load_ps(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float* p) const {
            _mm256_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<8> const & mask, float * p) const {
            _mm256_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float* p) const {
            _mm256_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<8> const & mask, float* p) const {
            _mm256_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }

        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec, t0);
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_add_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_add_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
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
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec;
            mVec = _mm256_add_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec;
            __m256 t2 = _mm256_add_ps(mVec, t0);
            mVec = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            __m256 t0 = _mm256_set1_ps(1);
            mVec = _mm256_add_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = _mm256_add_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __m256 t0 = _mm256_sub_ps(mVec, _mm256_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_sub_ps(mVec, _mm256_set1_ps(b));
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm256_sub_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            mVec = _mm256_sub_ps(mVec, _mm256_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_sub_ps(mVec, _mm256_set1_ps(b));
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
            __m256 t0 = _mm256_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_sub_ps(b.mVec, mVec);
            __m256 t1 = BLEND(b.mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_sub_ps(t0, mVec);
            __m256 t2 = BLEND(t0, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm256_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_sub_ps(b.mVec, mVec);
            mVec = BLEND(b.mVec, t0, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            mVec = _mm256_sub_ps(_mm256_set1_ps(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f subfroma(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_sub_ps(t0, mVec);
            mVec = BLEND(t0, t1, mask.mMask);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec;
            mVec = _mm256_sub_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = mVec;
            __m256 t2 = _mm256_sub_ps(mVec, t0);
            mVec = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            __m256 t0 = _mm256_set1_ps(1);
            mVec = _mm256_sub_ps(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_set1_ps(1);
            __m256 t1 = _mm256_sub_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            return SIMDVec_f(_mm256_mul_ps(mVec, b.mVec));
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            return SIMDVec_f(BLEND(mVec, t0, mask.mMask));
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            return SIMDVec_f(_mm256_mul_ps(mVec, _mm256_set1_ps(b)));
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_mul_ps(mVec, _mm256_set1_ps(b));
            return SIMDVec_f(BLEND(mVec, t0, mask.mMask));
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm256_mul_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            mVec = _mm256_mul_ps(mVec, _mm256_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_mul_ps(mVec, _mm256_set1_ps(b));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_div_ps(mVec, b.mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __m256 t0 = _mm256_div_ps(mVec, _mm256_set1_ps(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_div_ps(mVec, _mm256_set1_ps(b));
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm256_div_ps(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_div_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            mVec = _mm256_div_ps(mVec, _mm256_set1_ps(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_div_ps(mVec, _mm256_set1_ps(b));
            mVec  = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            return SIMDVec_f(_mm256_rcp_ps(mVec));
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_rcp_ps(mVec);
            return SIMDVec_f(BLEND(mVec, t0, mask.mMask));
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec), _mm256_set1_ps(b));
            return SIMDVec_f(t0);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec), _mm256_set1_ps(b));
            return SIMDVec_f(BLEND(mVec, t0, mask.mMask));
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = _mm256_rcp_ps(mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_rcp_ps(mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
            mVec = _mm256_mul_ps(_mm256_rcp_ps(mVec), _mm256_set1_ps(b));
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_mul_ps(_mm256_rcp_ps(mVec), _mm256_set1_ps(b));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, 0);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator==(SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_EQ_OQ);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, _CMP_NEQ_UQ);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_NEQ_UQ);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(SIMDVec_f const & b) const {;
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, _CMP_GT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_GT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, _CMP_LT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_LT_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(SIMDVec_f const & b) const {
            ;
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, _CMP_GE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_GE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple(SIMDVec_f const & b) const {
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, _CMP_LE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_LE_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            return SIMDVecMask<8>(m1);
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            alignas(32) int32_t raw[8];
            __m256 m0 = _mm256_cmp_ps(mVec, b.mVec, _CMP_EQ_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            _mm256_store_si256((__m256i*)raw, m1);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            alignas(32) int32_t raw[8];
            __m256 t0 = _mm256_set1_ps(b);
            __m256 m0 = _mm256_cmp_ps(mVec, t0, _CMP_EQ_OS);
            __m256i m1 = _mm256_castps_si256(m0);
            _mm256_store_si256((__m256i*)raw, m1);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            /* alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            for (unsigned int i = 0; i < 7; i++) {
            for (unsigned int j = i + 1; j < 8; j++) {
            if (raw[i] == raw[j]) {
            return false;
            }
            }
            }*/
            return true;
        }
        // HADD
        UME_FORCE_INLINE float hadd() const {
            __m256 t0 = _mm256_hadd_ps(mVec, mVec);
            __m256 t1 = _mm256_hadd_ps(t0, t0);
            __m128 t2 = _mm256_extractf128_ps(t1, 1);
            __m128 t3 = _mm256_castps256_ps128(t1);
            __m128 t4 = _mm_add_ps(t2, t3);
            float retval = _mm_cvtss_f32(t4);
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE float hadd(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = BLEND(t0, mVec, mask.mMask);
            __m256 t2 = _mm256_hadd_ps(t1, t1);
            __m256 t3 = _mm256_hadd_ps(t2, t2);
            __m128 t4 = _mm256_extractf128_ps(t3, 1);
            __m128 t5 = _mm256_castps256_ps128(t3);
            __m128 t6 = _mm_add_ps(t4, t5);
            float retval = _mm_cvtss_f32(t6);
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE float hadd(float b) const {
            __m256 t0 = _mm256_hadd_ps(mVec, mVec);
            __m256 t1 = _mm256_hadd_ps(t0, t0);
            __m128 t2 = _mm256_extractf128_ps(t1, 1);
            __m128 t3 = _mm256_castps256_ps128(t1);
            __m128 t4 = _mm_add_ps(t2, t3);
            float retval = _mm_cvtss_f32(t4);
            return retval + b;
        }
        // MHADDS
        UME_FORCE_INLINE float hadd(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = BLEND(t0, mVec, mask.mMask);
            __m256 t2 = _mm256_hadd_ps(t1, t1);
            __m256 t3 = _mm256_hadd_ps(t2, t2);
            __m128 t4 = _mm256_extractf128_ps(t3, 1);
            __m128 t5 = _mm256_castps256_ps128(t3);
            __m128 t6 = _mm_add_ps(t4, t5);
            float retval = _mm_cvtss_f32(t6);
            return retval + b;
        }
        // HMUL
        UME_FORCE_INLINE float hmul() const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m128 t1 = _mm256_castps256_ps128(mVec);
            __m128 t2 = _mm256_extractf128_ps(mVec, 1);
            __m128 t3 = _mm_mul_ps(t1, t2);
            __m128 t4 = _mm_shuffle_ps(t3, t0, 0xE);
            __m128 t5 = _mm_mul_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0x1);
            __m128 t7 = _mm_mul_ps(t5, t6);
            float retval = _mm_cvtss_f32(t7);
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE float hmul(SIMDVecMask<8> const & mask) const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m256 t1 = _mm256_set1_ps(1.0f);
            __m256 t2 = BLEND(t1, mVec, mask.mMask);
            __m128 t3 = _mm256_castps256_ps128(t2);
            __m128 t4 = _mm256_extractf128_ps(t2, 1);
            __m128 t5 = _mm_mul_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0xE);
            __m128 t7 = _mm_mul_ps(t5, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0x1);
            __m128 t9 = _mm_mul_ps(t7, t8);
            float retval = _mm_cvtss_f32(t9);
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE float hmul(float b) const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m128 t1 = _mm256_castps256_ps128(mVec);
            __m128 t2 = _mm256_extractf128_ps(mVec, 1);
            __m128 t3 = _mm_mul_ps(t1, t2);
            __m128 t4 = _mm_shuffle_ps(t3, t0, 0xE);
            __m128 t5 = _mm_mul_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0x1);
            __m128 t7 = _mm_mul_ps(t5, t6);
            float retval = _mm_cvtss_f32(t7);
            return retval + b;
        }
        // MHMULS
        UME_FORCE_INLINE float hmul(SIMDVecMask<8> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(1.0f);
            __m256 t1 = _mm256_set1_ps(1.0f);
            __m256 t2 = BLEND(t1, mVec, mask.mMask);
            __m128 t3 = _mm256_castps256_ps128(t2);
            __m128 t4 = _mm256_extractf128_ps(t2, 1);
            __m128 t5 = _mm_mul_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0xE);
            __m128 t7 = _mm_mul_ps(t5, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0x1);
            __m128 t9 = _mm_mul_ps(t7, t8);
            float retval = _mm_cvtss_f32(t9);
            return retval + b;
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            return _mm256_fmadd_ps(this->mVec, a.mVec, b.mVec);
#else
            return _mm256_add_ps(_mm256_mul_ps(this->mVec, b.mVec), c.mVec);
#endif
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m256 t0 = _mm256_fmadd_ps(this->mVec, b.mVec, c.mVec);
            return _mm256_blendv_ps(this->mVec, t0, _mm256_cvtepi32_ps(mask.mMask));
#else
            __m256 t0 = _mm256_add_ps(_mm256_mul_ps(mVec, b.mVec), c.mVec);
            return BLEND(mVec, t0, mask.mMask);
#endif
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            __m256 t1 = _mm256_sub_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_mul_ps(mVec, b.mVec);
            __m256 t1 = _mm256_sub_ps(t0, c.mVec);
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_add_ps(mVec, b.mVec);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec);
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256 t0 = _mm256_sub_ps(mVec, b.mVec);
            __m256 t1 = _mm256_mul_ps(t0, c.mVec);
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_max_ps(mVec, b.mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec, t0);
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm256_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_max_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_max_ps(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_max_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m256 t0 = _mm256_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m256 t0 = _mm256_min_ps(mVec, b.mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<8> const & mask, float b) const {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec, t0);
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm256_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            __m256 t0 = _mm256_min_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            __m256 t0 = _mm256_set1_ps(b);
            mVec = _mm256_min_ps(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<8> const & mask, float b) {
            __m256 t0 = _mm256_set1_ps(b);
            __m256 t1 = _mm256_min_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE float hmax() const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::lowest());
            __m128 t1 = _mm256_castps256_ps128(mVec);
            __m128 t2 = _mm256_extractf128_ps(mVec, 1);
            __m128 t3 = _mm_max_ps(t1, t2);
            __m128 t4 = _mm_shuffle_ps(t3, t0, 0xE);
            __m128 t5 = _mm_max_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0x1);
            __m128 t7 = _mm_max_ps(t5, t6);
            float retval = _mm_cvtss_f32(t7);
            return retval;
        }
        // MHMAX
        UME_FORCE_INLINE float hmax(SIMDVecMask<8> const & mask) const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::lowest());
            __m256 t1 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            __m128 t3 = _mm256_castps256_ps128(t2);
            __m128 t4 = _mm256_extractf128_ps(t2, 1);
            __m128 t5 = _mm_max_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0xE);
            __m128 t7 = _mm_max_ps(t5, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0x1);
            __m128 t9 = _mm_max_ps(t7, t8);
            float retval = _mm_cvtss_f32(t9);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE float hmin() const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::max());
            __m128 t1 = _mm256_castps256_ps128(mVec);
            __m128 t2 = _mm256_extractf128_ps(mVec, 1);
            __m128 t3 = _mm_min_ps(t1, t2);
            __m128 t4 = _mm_shuffle_ps(t3, t0, 0xE);
            __m128 t5 = _mm_min_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0x1);
            __m128 t7 = _mm_min_ps(t5, t6);
            float retval = _mm_cvtss_f32(t7);
            return retval;
        }
        // MHMIN
        UME_FORCE_INLINE float hmin(SIMDVecMask<8> const & mask) const {
            __m128 t0 = _mm_set1_ps(std::numeric_limits<float>::max());
            __m256 t1 = _mm256_set1_ps(std::numeric_limits<float>::max());
            __m256 t2 = BLEND(mVec, t1, mask.mMask);
            __m128 t3 = _mm256_castps256_ps128(t2);
            __m128 t4 = _mm256_extractf128_ps(t2, 1);
            __m128 t5 = _mm_min_ps(t3, t4);
            __m128 t6 = _mm_shuffle_ps(t5, t0, 0xE);
            __m128 t7 = _mm_min_ps(t5, t6);
            __m128 t8 = _mm_shuffle_ps(t7, t0, 0x1);
            __m128 t9 = _mm_min_ps(t7, t8);
            float retval = _mm_cvtss_f32(t9);
            return retval;
        }
        // IMIN
        // MIMIN

        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_sub_ps(_mm256_set1_ps(0.0f), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec, t0, _CMP_LT_OS);
            __m256 t2 = _mm256_sub_ps(t0, mVec);
            __m256 t3 = _mm256_blendv_ps(mVec, t2, t1);
            return SIMDVec_f(t3);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec, t0, _CMP_LT_OS);
            __m256 t2 = _mm256_and_ps(t1, _mm256_castsi256_ps(mask.mMask));
            __m256 t3 = _mm256_sub_ps(t0, mVec);
            __m256 t4 = _mm256_blendv_ps(mVec, t3, t2);
            return SIMDVec_f(t4);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec, t0, _CMP_LT_OS);
            __m256 t2 = _mm256_sub_ps(t0, mVec);
            mVec = _mm256_blendv_ps(mVec, t2, t1);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_set1_ps(0.0f);
            __m256 t1 = _mm256_cmp_ps(mVec, t0, _CMP_LT_OS);
            __m256 t2 = _mm256_and_ps(t1, _mm256_castsi256_ps(mask.mMask));
            __m256 t3 = _mm256_sub_ps(t0, mVec);
            mVec = _mm256_blendv_ps(mVec, t3, t2);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m256 t0 = _mm256_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_mul_ps(mVec, mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm256_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<8> const & mask) {
            __m256 t0 = _mm256_mul_ps(mVec, mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            return SIMDVec_f(_mm256_sqrt_ps(mVec));
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<8> const & mask) const {
            __m256 ret = _mm256_sqrt_ps(mVec);
            return SIMDVec_f(BLEND(mVec, ret, mask.mMask));
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm256_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<8> const & mask) {
            __m256 ret = _mm256_sqrt_ps(mVec);
            mVec = BLEND(mVec, ret, mask.mMask);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m256 t0 = _mm256_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // TRUNC
        UME_FORCE_INLINE SIMDVec_i<int32_t, 8> trunc() const {
            __m256i t0 = _mm256_cvttps_epi32(mVec);
            return SIMDVec_i<int32_t, 8>(t0);
        }
        // MTRUNC
        UME_FORCE_INLINE SIMDVec_i<int32_t, 8> trunc(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_setzero_ps();
            __m256 t1 = BLEND(t0, mVec, mask.mMask);
            __m256i t2 = _mm256_cvttps_epi32(t1);
            return SIMDVec_i<int32_t, 8>(t2);
        }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m256 t0 = _mm256_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_floor_ps(mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m256 t0 = _mm256_ceil_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<8> const & mask) const {
            __m256 t0 = _mm256_ceil_ps(mVec);
            __m256 t1 = BLEND(mVec, t0, mask.mMask);
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
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 8>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<8> const & mask) const {
            return VECTOR_EMULATION::expf<SIMDVec_f, SIMDVec_u<uint32_t, 8>, SIMDVecMask<8>>(mask, *this);
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
        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 4> const & a, SIMDVec_f<float, 4> const & b) {
            mVec = _mm256_insertf128_ps(mVec, a.mVec, 0);
            mVec = _mm256_insertf128_ps(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 4> const & a) {
            mVec = _mm256_insertf128_ps(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 4> const & b) {
            mVec = _mm256_insertf128_ps(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_f<float, 4> & a, SIMDVec_f<float, 4> & b) const {
            a.mVec = _mm256_extractf128_ps(mVec, 0);
            b.mVec = _mm256_extractf128_ps(mVec, 1);
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_f<float, 4> unpacklo() const {
            __m128 t0 = _mm256_extractf128_ps(mVec, 0);
            return SIMDVec_f<float, 4>(t0);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_f<float, 4> unpackhi() const {
            __m128 t0 = _mm256_extractf128_ps(mVec, 1);
            return SIMDVec_f<float, 4>(t0);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 8>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 8>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 8>() const;
    };
}
}

#undef BLEND

#endif
