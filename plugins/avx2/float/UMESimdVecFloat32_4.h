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

#define BLEND(a, b, mask) _mm_blendv_ps(a, b, _mm_castsi128_ps(mask))

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

        inline SIMDVec_f(__m128 const & x) {
            this->mVec = x;
        }

    public:
        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline SIMDVec_f(float f) {
            mVec = _mm_set1_ps(f);
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
        inline float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_f & insert(uint32_t index, float value) {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            raw[index] = value;
            mVec = _mm_load_ps(raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            mVec = BLEND(mVec, b.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(float b) {
            mVec = _mm_set1_ps(b);
            return *this;
        }
        inline SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(float const * p) {
            mVec = _mm_loadu_ps(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<4> const & mask, float const * p) {
            __m128 t0 = _mm_loadu_ps(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(float const * p) {
            mVec = _mm_load_ps(p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const * p) {
            __m128 t0 = _mm_load_ps(p);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        inline float* store(float* p) const {
            _mm_storeu_ps(p, mVec);
            return p;
        }
        // MSTORE
        inline float* store(SIMDVecMask<4> const & mask, float * p) const {
            _mm_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline float* storea(float * p) const {
            _mm_store_ps(p, mVec);
            return p;
        }
        // MSTOREA
        inline float* storea(SIMDVecMask<4> const & mask, float * p) const {
            _mm_maskstore_ps(p, mask.mMask, mVec);
            return p;
        }

        // BLENDV
        // BLENDS
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ADDS
        inline SIMDVec_f add(float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm_add_ps(this->mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_add_ps(this->mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(float b) {
            mVec = _mm_add_ps(this->mVec, _mm_set1_ps(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
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
        inline SIMDVec_f postinc() {
            __m128 t0 = mVec;
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
            __m128 t1 = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t0);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            mVec = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_add_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBS
        inline SIMDVec_f sub(float b) const {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(b));
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator-=(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_sub_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(float b) {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(b));
            return *this;
        }
        inline SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<4> const & mask, float b) {
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
        inline SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            __m128 t1 = BLEND(b.mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(float b) const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(b), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_sub_ps(t0, mVec);
            __m128 t2 = BLEND(t0, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = _mm_sub_ps(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_sub_ps(b.mVec, mVec);
            mVec = BLEND(b.mVec, t0, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_sub_ps(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_sub_ps(t0, mVec);
            mVec = BLEND(t0, t1, mask.mMask);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m128 t0 = mVec;
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = mVec;
            __m128 t1 = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t0);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_sub_ps(mVec, _mm_set1_ps(1.0f));
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            __m128 t2 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MULS
        inline SIMDVec_f mul(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            __m128 t2 = _mm_castsi128_ps(mask.mMask);
            __m128 t3 = _mm_blendv_ps(mVec, t1, t2);
            return SIMDVec_f(t3);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm_mul_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_mul_ps(mVec, b.mVec);
            mVec = _mm_blendv_ps(mVec, t0, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(mVec, t0);
            return *this;
        }
        inline SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_mul_ps(mVec, t0);
            mVec = _mm_blendv_ps(mVec, t1, _mm_castsi128_ps(mask.mMask));
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // DIVS
        inline SIMDVec_f div(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        inline SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm_div_ps(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_div_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(float b) {
            __m128 t0 = _mm_set1_ps(b);
            mVec = _mm_div_ps(mVec, t0);
            return *this;
        }
        inline SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_div_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m128 t0 = _mm_rcp_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // RCPS
        inline SIMDVec_f rcp(float b) const {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            return SIMDVec_f(t2);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            __m128 t3 = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t3);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec = _mm_rcp_ps(mVec);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_rcp_ps(mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(float b) {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            mVec = _mm_mul_ps(t0, t1);
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
            __m128 t0 = _mm_rcp_ps(mVec);
            __m128 t1 = _mm_set1_ps(b);
            __m128 t2 = _mm_mul_ps(t0, t1);
            mVec = BLEND(mVec, t2, mask.mMask);
            return *this;
        }
        // CMPEQV
        inline SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            __m128i m0 = _mm_castps_si128(_mm_cmpeq_ps(mVec, b.mVec));
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128i m0 = _mm_castps_si128(_mm_cmpeq_ps(mVec, t0));
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            __m128i m0 = _mm_castps_si128(_mm_cmpneq_ps(mVec, b.mVec));
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128i m0 = _mm_castps_si128(_mm_cmpneq_ps(mVec, t0));
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            __m128i m0 = _mm_castps_si128(_mm_cmpgt_ps(mVec, b.mVec));
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128i m0 = _mm_castps_si128(_mm_cmpgt_ps(mVec, t0));
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            __m128 t0 = _mm_cmplt_ps(mVec, b.mVec);
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(float b) const {
            __m128 t0 = _mm_cmplt_ps(mVec, _mm_set1_ps(b));
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            __m128 t0 = _mm_cmpge_ps(mVec, b.mVec);
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(float b) const {
            __m128 t0 = _mm_cmpge_ps(mVec, _mm_set1_ps(b));
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            __m128 t0 = _mm_cmple_ps(mVec, b.mVec);
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(float b) const {
            __m128 t0 = _mm_cmple_ps(mVec, _mm_set1_ps(b));
            __m128i m0 = _mm_castps_si128(t0);
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            alignas(16) uint32_t raw[4];
            __m128 m0 = _mm_cmpeq_ps(mVec, b.mVec);
            _mm_store_si128((__m128i*)raw, _mm_castps_si128(m0));
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0);
        }
        // CMPES
        inline bool cmpe(float b) const {
            alignas(16) uint32_t raw[4];
            __m128 m0 = _mm_cmpeq_ps(mVec, _mm_set1_ps(b));
            _mm_store_si128((__m128i*)raw, _mm_castps_si128(m0));
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0);
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = BLEND(mVec, b.mVec, mask.mMask);
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = BLEND(mVec, _mm_set1_ps(b), mask.mMask);
            return SIMDVec_f(t0);
        }
        // HADD
        inline float hadd() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD
        inline float hadd(SIMDVecMask<4> const & mask) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(mVec, _mm_set1_ps(0.0f), mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // HADDS
        inline float hadd(float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // MHADDS
        inline float hadd(SIMDVecMask<4> const & mask, float b) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(mVec, _mm_set1_ps(0.0f), mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // HMUL
        inline float hmul() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // MHMUL
        inline float hmul(SIMDVecMask<4> const & mask) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(mVec, _mm_set1_ps(0.0f), mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3];
        }
        // HMULS
        inline float hmul(float b) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3] * b;
        }
        // MHMULS
        inline float hmul(SIMDVecMask<4> const & mask, float b) const {
            alignas(16) float raw[4];
            __m128 t0 = BLEND(mVec, _mm_set1_ps(0.0f), mask.mMask);
            _mm_store_ps(raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3] * b;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m128 t0 = _mm_fmadd_ps(mVec, b.mVec, c.mVec);
#else
            __m128 t0 = _mm_add_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#ifdef FMA
            __m128 t0 = _mm_fmadd_ps(mVec, b.mVec, c.mVec);
#else
            __m128 t0 = _mm_add_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
#endif
            __m128 t1 = _mm_blendv_ps(mVec, t0, _mm_cvtepi32_ps(mask.mMask));
            return SIMDVec_f(t1);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_sub_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_sub_ps(_mm_mul_ps(mVec, b.mVec), c.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_add_ps(mVec, b.mVec), c.mVec);
            return SIMDVec_f(t0);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_add_ps(mVec, b.mVec), c.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_sub_ps(mVec, b.mVec), c.mVec);
            return SIMDVec_f(t0);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m128 t0 = _mm_mul_ps(_mm_sub_ps(mVec, b.mVec), c.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXS
        inline SIMDVec_f max(float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
            __m128 t0 = _mm_set1_ps(b);
            __m128 t1 = _mm_max_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm_max_ps(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_max_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
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
            __m128 t1 = _mm_max_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
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
            __m128 t1 = _mm_min_ps(mVec, t0);
            __m128 t2 = BLEND(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm_min_ps(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __m128 t0 = _mm_min_ps(mVec, b.mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
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
            __m128 t1 = _mm_min_ps(mVec, t0);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // HMAX
        inline float hmax() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            float t1 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        inline float hmax(SIMDVecMask<4> const & mask) const {
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
        inline float hmin() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            float t0 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            float t1 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        inline float hmin(SIMDVecMask<4> const & mask) const {
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
        inline SIMDVec_f & gather(float* baseAddr, uint32_t* indices) {
            __m128i t0 = _mm_load_si128((__m128i*)indices);
            mVec = _mm_i32gather_ps((const float *)baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) {
            __m128i t0 = _mm_load_si128((__m128i*)indices);
            __m128 t1 = _mm_i32gather_ps((const float *)baseAddr, t0, 4);
            mVec = BLEND(mVec, t1, mask.mMask);
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            mVec = _mm_i32gather_ps((const float *)baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
            __m128 t0 = _mm_i32gather_ps((const float *)baseAddr, indices.mVec, 4);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SCATTERS
        inline float* scatter(float* baseAddr, uint32_t* indices) const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            for (int i = 0; i < 4; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) const {
            alignas(16) float raw[4];
            alignas(16) uint32_t rawMask[4];
            _mm_store_ps(raw, mVec);
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
            for (int i = 0; i < 4; i++) { if (rawMask[i] == SIMDVecMask<4>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            alignas(16) float raw[4];
            alignas(16) uint32_t rawIndices[4];
            _mm_store_ps(raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            for (int i = 0; i < 4; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
            alignas(16) float raw[8];
            alignas(16) uint32_t rawIndices[8];
            alignas(16) uint32_t rawMask[8];
            _mm_store_ps(raw, mVec);
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            _mm_store_si128((__m128i*) rawMask, mask.mMask);
            for (int i = 0; i < 4; i++) {
                if (rawMask[i] == SIMDVecMask<4>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
            return baseAddr;
        }
        // NEG
        inline SIMDVec_f neg() const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_sub_ps(_mm_set1_ps(0.0f), mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            __m128 t2 = _mm_and_ps(t1, mVec);
            return SIMDVec_f(t2);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            __m128 t2 = _mm_and_ps(t1, mVec);
            __m128 t3 = BLEND(mVec, t2, mask.mMask);
            return SIMDVec_f(t3);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            mVec = _mm_and_ps(t1, mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(0x7FFFFFFF);
            __m128 t1 = _mm_castsi128_ps(t0);
            __m128 t2 = _mm_and_ps(t1, mVec);
            mVec = BLEND(mVec, t2, mask.mMask);
            return *this;
        }
        // CMPEQRV
        // CMPEQRS

        // SQR
        inline SIMDVec_f sqr() const {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec = _mm_mul_ps(mVec, mVec);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_mul_ps(mVec, mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m128 t0 = _mm_sqrt_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_sqrt_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = _mm_sqrt_ps(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
            __m128 t0 = _mm_sqrt_ps(mVec);
            mVec = BLEND(mVec, t0, mask.mMask);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m128 t0 = _mm_round_ps(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<4> const & mask) const {
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
            __m128 t0 = _mm_castsi128_ps(mask.mMask);
            __m128 t1 = _mm_setzero_ps();
            __m128i t2 = _mm_cvttps_epi32(_mm_blendv_ps(t1, mVec, t0));
            return SIMDVec_i<int32_t, 4>(t2);
        }
        // FLOOR
        inline SIMDVec_f floor() const {
            __m128 t0 = _mm_floor_ps(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __m128 t0 = _mm_floor_ps(mVec);
            __m128 t1 = BLEND(mVec, t0, mask.mMask);
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
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        inline SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_ps(raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
            alignas(16) float raw[4] = { a.mVec[0], a.mVec[1], 0.0f, 0.0f };
            alignas(16) uint32_t mask_raw[4] = { 0xFFFFFFFF, 0xFFFFFFFF, 0, 0 };
            __m128i t0 = _mm_load_si128((__m128i*)mask_raw);
            __m128 t1 = _mm_load_ps(raw);
            mVec = BLEND(mVec, t1, t0);
            return *this;
        }
        // PACKHI
        inline SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
            alignas(16) float raw[4] = { 0.0f, 0.0f, b.mVec[0], b.mVec[1] };
            alignas(16) uint32_t mask_raw[4] = { 0, 0, 0xFFFFFFFF, 0xFFFFFFFF };
            __m128i t0 = _mm_load_si128((__m128i*)mask_raw);
            __m128 t1 = _mm_load_ps(raw);
            mVec = BLEND(mVec, t1, t0);
            return *this;
        }
        // UNPACK
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
            _mm_store_ps(raw, mVec);
            return SIMDVec_f<float, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        inline SIMDVec_f<float, 2> unpackhi() const {
            alignas(16) float raw[4];
            _mm_store_ps(raw, mVec);
            return SIMDVec_f<float, 2>(raw[2], raw[3]);
        }

        // PROMOTE
        inline operator SIMDVec_f<double, 4>() const;
        // DEGRADE
        // -

        // FTOU
        inline operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        inline operator SIMDVec_i<int32_t, 4>() const;
    };
}
}

#undef BLEND

#endif
