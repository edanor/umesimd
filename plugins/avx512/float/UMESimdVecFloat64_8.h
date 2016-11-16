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

#ifndef UME_SIMD_VEC_FLOAT64_8_H_
#define UME_SIMD_VEC_FLOAT64_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 8> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 8>,
            SIMDVec_u<uint64_t, 8>,
            SIMDVec_i<int64_t, 8>,
            double,
            8,
            uint64_t,
            SIMDVecMask<8>,
            SIMDSwizzle<8>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 8>,
            SIMDVec_f<double, 4>>
    {
    private:
        __m512d mVec;

        typedef SIMDVec_u<uint64_t, 8>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 8>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 4>      HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<double, 16>;

        UME_FORCE_INLINE SIMDVec_f(__m512d const & x) {
            mVec = x;
        }

    public:
        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double d) {
            mVec = _mm512_set1_pd(d);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, double>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<double>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(double const *p) {
            mVec = _mm512_loadu_pd(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double d0, double d1, double d2, double d3, 
                         double d4, double d5, double d6, double d7) {
            mVec = _mm512_set_pd(d7, d6, d5, d4, d3, d2, d1, d0);
        }
        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_pd(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

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
            mVec = _mm512_mask_mov_pd(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec = _mm512_set1_pd(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_mov_pd(mVec, mask.mMask, _mm512_set1_pd(b));
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec = _mm512_loadu_pd(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<8> const & mask, double const * p) {
            mVec = _mm512_mask_loadu_pd(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec = _mm512_load_pd(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<8> const & mask, double const * p) {
            mVec = _mm512_mask_load_pd(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double * p) const {
            _mm512_storeu_pd(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<8> const & mask, double * p) const {
            _mm512_mask_storeu_pd(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double * p) const {
            _mm512_store_pd(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<8> const & mask, double * p) const {
             _mm512_mask_store_pd(p, mask.mMask, mVec);
            return p;
        }
        
        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec, mask.mMask, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __m512d t0 = _mm512_add_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm512_add_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            mVec = _mm512_add_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            return add(b);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            return add(mask, b);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(double b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<8> const & mask, double b) const {
            return add(mask, b);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            return adda(b);
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            return adda(mask, b);
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(double b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<8> const & mask, double b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __m512d t0 = mVec;
            mVec = _mm512_add_pd(mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<8> const & mask) {
            __m512d t0 = mVec;
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec = _mm512_add_pd(mVec, _mm512_set1_pd(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_sub_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __m512d t0 = _mm512_sub_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm512_sub_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const double b) {
            mVec = _mm512_sub_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<8> const & mask, const double b) {
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            return sub(mask, b);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(double b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<8> const & mask, double b) const {
            return sub(mask, b);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            return suba(mask, b);
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(double b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<8> const & mask, double b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            __m512d t0 = _mm512_sub_pd(a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & a) const {
            __m512d t0 = _mm512_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double a) const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(a), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<8> const & mask, double a) const {
            __m512d t0 = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask, _mm512_set1_pd(a), mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = _mm512_sub_pd(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, SIMDVec_f const & a) {
            mVec = _mm512_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(double a) {
            mVec = _mm512_sub_pd(_mm512_set1_pd(a), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, double a) {
            mVec = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask, _mm512_set1_pd(a), mVec);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m512d t0 = mVec;
            mVec = _mm512_sub_pd(mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<8> const & mask) {
            __m512d t0 = mVec;
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec = _mm512_sub_pd(mVec, _mm512_set1_pd(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __m512d t0 = _mm512_mul_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm512_mul_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            mVec = _mm512_mul_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_div_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            __m512d t0 = _mm512_div_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm512_div_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(double b) {
            mVec = _mm512_div_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m512d t0 = _mm512_div_pd(_mm512_set1_pd(1.0), mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(1.0), mVec);
            return SIMDVec_f(t0);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            __m512d t0 = _mm512_div_pd(_mm512_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec = _mm512_div_pd(_mm512_set1_pd(1.0), mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(1.0), mVec);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(double b) {
            mVec = _mm512_div_pd(_mm512_set1_pd(b), mVec);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(b), mVec);
            return *this;
        }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 0);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 0);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 12);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 12);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (double b) const {
            return cmpne(b);
        }

        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 30);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 30);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 17);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 17);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 29);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 29);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 18);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 18);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 0);
            return (m0 == 0x03);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 0);
            return (m0 == 0x03);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            alignas(16) double raw[2];
            _mm512_store_pd(raw, mVec);
            return raw[0] != raw[1];
        }
        // HADD
        UME_FORCE_INLINE double hadd() const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3] + raw[4] + raw[5] + raw[6] + raw[7];
#else
            double retval = _mm512_reduce_add_pd(mVec);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE double hadd(SIMDVecMask<8> const & mask) const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 0;
            double t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 0;
            double t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 0;
            double t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 0;
            double t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 0;
            double t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 0;
            double t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 0;
            double t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 0;
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
#else
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, mVec);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE double hadd(double b) const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3] + raw[4] + raw[5] + raw[6] + raw[7];
#else
            double retval = _mm512_reduce_add_pd(mVec);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE double hadd(SIMDVecMask<8> const & mask, double b) const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 0;
            double t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 0;
            double t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 0;
            double t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 0;
            double t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 0;
            double t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 0;
            double t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 0;
            double t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 0;
            return b + t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
#else
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, mVec);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE double hmul() const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3] * raw[4] * raw[5] * raw[6] * raw[7];
#else
            double retval = _mm512_reduce_mul_pd(mVec);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE double hmul(SIMDVecMask<8> const & mask) const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 1.0;
            double t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 1.0;
            double t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 1.0;
            double t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 1.0;
            double t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 1.0;
            double t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 1.0;
            double t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 1.0;
            double t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 1.0;
            return t0 * t1 * t2 * t3 * t4 * t5 * t6 * t7;
#else
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, mVec);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE double hmul(double b) const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3] * raw[4] * raw[5] * raw[6] * raw[7];
#else
            double retval = _mm512_reduce_mul_pd(mVec);
            return b * retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE double hmul(SIMDVecMask<8> const & mask, double b) const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 1.0;
            double t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 1.0;
            double t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 1.0;
            double t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 1.0;
            double t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 1.0;
            double t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 1.0;
            double t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 1.0;
            double t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 1.0;
            return b * t0 * t1 * t2 * t3 * t4 * t5 * t6 * t7;
#else
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, mVec);
            return b * retval;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmadd_pd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmadd_pd(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmsub_pd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmsub_pd(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_add_pd(mVec, b.mVec);
            __m512d t1 = _mm512_mul_pd(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            __m512d t1 = _mm512_mask_mul_pd(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_sub_pd(mVec, b.mVec);
            __m512d t1 = _mm512_mul_pd(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            __m512d t1 = _mm512_mask_mul_pd(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_max_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_max_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_max_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_max_pd(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm512_max_pd(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_max_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_max_pd(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<8> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_mask_max_pd(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_min_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_min_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_min_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_min_pd(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm512_min_pd(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_min_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_min_pd(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<8> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_mask_min_pd(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE double hmax() const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = raw[0] > raw[1] ? raw[0] : raw[1];
            double t1 = raw[2] > raw[3] ? raw[2] : raw[3];
            double t2 = raw[4] > raw[5] ? raw[4] : raw[5];
            double t3 = raw[6] > raw[7] ? raw[6] : raw[7];
            double t4 = t0 > t1 ? t0 : t1;
            double t5 = t2 > t3 ? t2 : t3;
            return t4 > t5 ? t4 : t5;
#else
            double retval = _mm512_reduce_max_pd(mVec);
            return retval;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE double hmax(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : std::numeric_limits<double>::lowest();
            double t1 = (((mask.mMask & 0x02) != 0) && raw[1] > t0) ? raw[1] : t0;
            double t2 = (((mask.mMask & 0x04) != 0) && raw[2] > t1) ? raw[2] : t1;
            double t3 = (((mask.mMask & 0x08) != 0) && raw[3] > t2) ? raw[3] : t2;
            double t4 = (((mask.mMask & 0x10) != 0) && raw[4] > t3) ? raw[4] : t3;
            double t5 = (((mask.mMask & 0x20) != 0) && raw[5] > t4) ? raw[5] : t4;
            double t6 = (((mask.mMask & 0x40) != 0) && raw[6] > t5) ? raw[6] : t5;
            double t7 = (((mask.mMask & 0x80) != 0) && raw[7] > t6) ? raw[7] : t6;
            return t7;
#else
            double retval = _mm512_mask_reduce_max_pd(mask.mMask, mVec);
            return retval;
#endif
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE double hmin() const {
#if defined(__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = raw[0] < raw[1] ? raw[0] : raw[1];
            double t1 = raw[2] < raw[3] ? raw[2] : raw[3];
            double t2 = raw[4] < raw[5] ? raw[4] : raw[5];
            double t3 = raw[6] < raw[7] ? raw[6] : raw[7];
            double t4 = t0 < t1 ? t0 : t1;
            double t5 = t2 < t3 ? t2 : t3;
            return t4 < t5 ? t4 : t5;
#else
            double retval = _mm512_reduce_min_pd(mVec);
            return retval;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE double hmin(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            double t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : std::numeric_limits<double>::lowest();
            double t1 = (((mask.mMask & 0x02) != 0) && raw[1] < t0) ? raw[1] : t0;
            double t2 = (((mask.mMask & 0x04) != 0) && raw[2] < t1) ? raw[2] : t1;
            double t3 = (((mask.mMask & 0x08) != 0) && raw[3] < t2) ? raw[3] : t2;
            double t4 = (((mask.mMask & 0x10) != 0) && raw[4] < t3) ? raw[4] : t3;
            double t5 = (((mask.mMask & 0x20) != 0) && raw[5] < t4) ? raw[5] : t4;
            double t6 = (((mask.mMask & 0x40) != 0) && raw[6] < t5) ? raw[6] : t5;
            double t7 = (((mask.mMask & 0x80) != 0) && raw[7] < t6) ? raw[7] : t6;
            return t7;
#else
            double retval = _mm512_mask_reduce_min_pd(mask.mMask, mVec);
            return retval;
#endif
        }
        // IMIN
        // MIMIN

        // GATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(double const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            mVec = _mm512_i64gather_pd(t2, baseAddr, 8);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(SIMDVecMask<8> const & mask, double const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            __mmask8 m0 = mask.mMask & 0xFF;
            mVec = _mm512_mask_i64gather_pd(mVec, m0, t2, baseAddr, 8);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, uint64_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_i64gather_pd(t0, baseAddr, 8);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<8> const & mask, double const * baseAddr, uint64_t const * indices) {
            __mmask8 m0 = mask.mMask & 0xFF;
            __m512i t0 = _mm512_loadu_si512(indices);
            mVec = _mm512_mask_i64gather_pd(mVec, m0, t0, baseAddr, 8);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, SIMDVec_u<uint64_t, 8> const & indices) {
            mVec = _mm512_i64gather_pd(indices.mVec, baseAddr, 8);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<8> const & mask, double const * baseAddr, SIMDVec_u<uint64_t, 8> const & indices) {
            __mmask8 m0 = mask.mMask & 0xFF;
            mVec = _mm512_mask_i64gather_pd(mVec, m0, indices.mVec, baseAddr, 8);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE double* scatteru(double* baseAddr, uint64_t stride) const {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            _mm512_i64scatter_pd(baseAddr, t2, mVec, 8);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE double* scatteru(SIMDVecMask<8> const & mask, double* baseAddr, uint64_t stride) const {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            _mm512_mask_i64scatter_pd(baseAddr, mask.mMask, t2, mVec, 8);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE double* scatter(double* baseAddr, uint64_t* indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            _mm512_i64scatter_pd(baseAddr, t0, mVec, 8);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE double* scatter(SIMDVecMask<8> const & mask, double* baseAddr, uint64_t* indices) {
            __mmask8 m0 = mask.mMask & 0xFF;
            __m512i t0 = _mm512_loadu_si512(indices);
            _mm512_mask_i64scatter_pd(baseAddr, m0, t0, mVec, 8);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE double* scatter(double* baseAddr, SIMDVec_u<uint64_t, 8> const & indices) {
            _mm512_i64scatter_pd(baseAddr, indices.mVec, mVec, 8);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE double* scatter(SIMDVecMask<8> const & mask, double* baseAddr, SIMDVec_u<uint64_t, 8> const & indices) {
            __mmask8 m0 = mask.mMask & 0xFF;
            _mm512_mask_i64scatter_pd(baseAddr, m0, indices.mVec, mVec, 8);
            return baseAddr;
        }

        // NEG
        
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_setzero_pd();
            __m512d t1 = _mm512_mask_sub_pd(mVec, mask.mMask, t0, mVec);
            return SIMDVec_f(t1);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<8> const & mask) {
            __m512d t0 = _mm512_setzero_pd();
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, t0, mVec);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
#if defined (__GNUG__)
            __m512i t0 = _mm512_castpd_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t2 = _mm512_and_epi64(t0, t1);
            __m512d t3 = _mm512_castsi512_pd(t2);
            return SIMDVec_f(t3);
#else
            __m512d t0 = _mm512_abs_pd(mVec);
            return SIMDVec_f(t0);
#endif
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            __m512i t0 = _mm512_castpd_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t2 = _mm512_and_epi64(t0, t1);
            __m512d t3 = _mm512_castsi512_pd(t2);
            __m512d t4 = _mm512_mask_mov_pd(mVec, mask.mMask, t3);
            return SIMDVec_f(t4);
#else
            __m512d t0 = _mm512_mask_abs_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
#endif
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
#if defined (__GNUG__)
            __m512i t0 = _mm512_castpd_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t2 = _mm512_and_epi64(t0, t1);
            mVec = _mm512_castsi512_pd(t2);
            return *this;
#else
            mVec = _mm512_abs_pd(mVec);
            return *this;
#endif
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
#if defined (__GNUG__)
            __m512i t0 = _mm512_castpd_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t2 = _mm512_and_epi64(t0, t1);
            __m512d t3 = _mm512_castsi512_pd(t2);
            mVec = _mm512_mask_mov_pd(mVec, mask.mMask, t3);
            return *this;
#else
            mVec = _mm512_mask_abs_pd(mVec, mask.mMask, mVec);
            return *this;
#endif
        }

        // COPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVec_f const & b) const {
#if defined(__AVX512DQ__)
            __m512d t0 = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
            __m512d t1 = _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000));
            __m512d t2 = _mm512_and_pd(mVec, t0);
            __m512d t3 = _mm512_and_pd(b.mVec, t1);
            __m512d t4 = _mm512_or_pd(t2, t3);
            return SIMDVec_f(t4);
#else
            __m512i t0 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t1 = _mm512_set1_epi64(0x8000000000000000);
            __m512i t2 = _mm512_castpd_si512(mVec);
            __m512i t3 = _mm512_castpd_si512(b.mVec);
            __m512i t4 = _mm512_and_epi64(t2, t0);
            __m512i t5 = _mm512_and_epi64(t3, t1);
            __m512i t6 = _mm512_or_epi64(t4, t5);
            __m512d t7 = _mm512_castsi512_pd(t6);
            return SIMDVec_f(t7);
#endif
        }
        // MCOPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512DQ__)
            __m512d t0 = _mm512_castsi512_pd(_mm512_set1_epi64(0x7FFFFFFFFFFFFFFF));
            __m512d t1 = _mm512_castsi512_pd(_mm512_set1_epi64(0x8000000000000000));
            __m512d t2 = _mm512_and_pd(mVec, t0);
            __m512d t3 = _mm512_and_pd(b.mVec, t1);
            __m512d t4 = _mm512_mask_or_pd(mVec, mask.mMask, t2, t3);
            return SIMDVec_f(t4);
#else
            __m512i t0 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t1 = _mm512_set1_epi64(0x8000000000000000);
            __m512i t2 = _mm512_castpd_si512(mVec);
            __m512i t3 = _mm512_castpd_si512(b.mVec);
            __m512i t4 = _mm512_and_epi64(t2, t0);
            __m512i t5 = _mm512_and_epi64(t3, t1);
            __m512i t6 = _mm512_mask_or_epi64(t2, mask.mMask, t4, t5);
            __m512d t7 = _mm512_castsi512_pd(t6);
            return SIMDVec_f(t7);
#endif
        }
        
        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m512d t0 = _mm512_mul_pd(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm512_mul_pd(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m512d t0 = _mm512_sqrt_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_sqrt_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm512_sqrt_pd(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_sqrt_pd(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec, 0);
            __m256d t1 = _mm512_extractf64x4_pd(mVec, 1);

            __m256d t2 = _mm256_round_pd(t0, _MM_FROUND_TO_NEAREST_INT);
            __m256d t3 = _mm256_round_pd(t1, _MM_FROUND_TO_NEAREST_INT);
            __m512d t4 = _mm512_castpd256_pd512(t2);
            __m512d t5 = _mm512_insertf64x4(t4, t3, 1);
            return SIMDVec_f(t5);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<8> const & mask) const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec, 0);
            __m256d t1 = _mm512_extractf64x4_pd(mVec, 1);

            __m256d t2 = _mm256_round_pd(t0, _MM_FROUND_TO_NEAREST_INT);
            __m256d t3 = _mm256_round_pd(t1, _MM_FROUND_TO_NEAREST_INT);
            __m512d t4 = _mm512_castpd256_pd512(t2);
            __m512d t5 = _mm512_insertf64x4(t4, t3, 1);
            __m512d t6 = _mm512_mask_mov_pd(mVec, mask.mMask, t5);
            return SIMDVec_f(t6);
        }
        // TRUNC
        UME_FORCE_INLINE SIMDVec_i<int64_t, 8> trunc() const {
#if defined(__AVX512DQ__)
            __m512i t0 = _mm512_cvttpd_epi64(mVec);
            return SIMDVec_i<int64_t, 8>(t0);
#else
            alignas(64) double raw_d[8];
            alignas(64) int64_t raw_i[8];
            _mm512_store_pd(raw_d, mVec);
            raw_i[0] = (int64_t)raw_d[0];
            raw_i[1] = (int64_t)raw_d[1];
            raw_i[2] = (int64_t)raw_d[2];
            raw_i[3] = (int64_t)raw_d[3];
            raw_i[4] = (int64_t)raw_d[4];
            raw_i[5] = (int64_t)raw_d[5];
            raw_i[6] = (int64_t)raw_d[6];
            raw_i[7] = (int64_t)raw_d[7];
            __m512i t0 = _mm512_load_epi64(raw_i);
            return SIMDVec_i<int64_t, 8>(t0);
#endif
        }
        // MTRUNC
        UME_FORCE_INLINE SIMDVec_i<int64_t, 8> trunc(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512DQ__)
            __m512i t0 = _mm512_mask_cvttpd_epi64(_mm512_setzero_si512(), mask.mMask, mVec);
            return SIMDVec_i<int64_t, 8>(t0);
#else
            alignas(64) double raw_d[8];
            alignas(64) int64_t raw_i[8];
            __m512d t0 = _mm512_set1_pd(0.0);
            __m512d t1 = _mm512_mask_mov_pd(t0, mask.mMask, mVec);
            _mm512_store_pd(raw_d, t1);
            raw_i[0] = (int64_t)raw_d[0];
            raw_i[1] = (int64_t)raw_d[1];
            raw_i[2] = (int64_t)raw_d[2];
            raw_i[3] = (int64_t)raw_d[3];
            raw_i[4] = (int64_t)raw_d[4];
            raw_i[5] = (int64_t)raw_d[5];
            raw_i[6] = (int64_t)raw_d[6];
            raw_i[7] = (int64_t)raw_d[7];
            __m512i t2 = _mm512_load_epi64(raw_i);
            return SIMDVec_i<int64_t, 8>(t2);
#endif
        }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m512d t0 = _mm512_floor_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_floor_pd(mVec);
            __m512d t1 = _mm512_mask_mov_pd(mVec, mask.mMask, t0);
            return SIMDVec_f(t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m512d t0 = _mm512_ceil_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_ceil_pd(mVec);
            __m512d t1 = _mm512_mask_mov_pd(mVec, mask.mMask, t0);
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
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_exp_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 8>>(*this);
        #endif
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<8> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_mask_exp_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 8>, SIMDVecMask<8>>(mask, *this);
        #endif
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_log_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 8>>(*this);
        #endif
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<8> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_mask_log_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 8>, SIMDVecMask<8>>(mask, *this);
        #endif
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_sin_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this);
        #endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<8> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_mask_sin_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
        #endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_cos_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this);
        #endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<8> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_mask_cos_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
        #endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos[8];
            sinvec.mVec = _mm512_sincos_pd((__m512d*)raw_cos, mVec);
            cosvec.mVec = _mm512_load_pd(raw_cos);
        #else
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this, sinvec, cosvec);
        #endif
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<8> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos[8]; // 64B aligned data for 512b vector operation is needed.
            sinvec.mVec = _mm512_mask_sincos_pd((__m512d*)raw_cos, mVec, mVec, mask.mMask, mVec);
            cosvec.mVec = _mm512_load_pd(raw_cos);
        #else
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<8>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<8>>(mask, *this);
        #endif
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        SIMDVec_f & pack(SIMDVec_f<double, 4> const & a, SIMDVec_f<double, 4> const & b) {
            mVec = _mm512_insertf64x4(mVec, a.mVec, 0);
            mVec = _mm512_insertf64x4(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        SIMDVec_f & packlo(SIMDVec_f<double, 4> const & a) {
            mVec = _mm512_insertf64x4(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        SIMDVec_f & packhi(SIMDVec_f<double, 4> const & b) {
            mVec = _mm512_insertf64x4(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        void unpack(SIMDVec_f<double, 4> & a, SIMDVec_f<double, 4> & b) const {
            a.mVec = _mm512_extractf64x4_pd(mVec, 0);
            b.mVec = _mm512_extractf64x4_pd(mVec, 1);
        }
        // UNPACKLO
        SIMDVec_f<double, 4> unpacklo() const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec, 0);
            return SIMDVec_f<double, 4>(t0);
        }
        // UNPACKHI
        SIMDVec_f<double, 4> unpackhi() const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec, 1);
            return SIMDVec_f<double, 4>(t0);
        }
        
        // PROMOTE
        // -    
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_f<float, 8>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 8>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 8>() const;
    };

}
}

#endif
