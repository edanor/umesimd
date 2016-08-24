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

        inline SIMDVec_f(__m512d const & x) {
            mVec = x;
        }

    public:
        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline SIMDVec_f(double d) {
            mVec = _mm512_set1_pd(d);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, double>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<double>(i)) {}
        // LOAD-CONSTR
        inline explicit SIMDVec_f(double const *p) {
            mVec = _mm512_loadu_pd(p);
        }
        // FULL-CONSTR
        inline SIMDVec_f(double d0, double d1, double d2, double d3, 
                         double d4, double d5, double d6, double d7) {
            mVec = _mm512_set_pd(d7, d6, d5, d4, d3, d2, d1, d0);
        }
        // EXTRACT
        inline double extract(uint32_t index) const {
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            return raw[index];
        }
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
            alignas(64) double raw[8];
            _mm512_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_pd(raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<8>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mov_pd(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec = _mm512_set1_pd(b);
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_mov_pd(mVec, mask.mMask, _mm512_set1_pd(b));
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec = _mm512_loadu_pd(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<8> const & mask, double const * p) {
            mVec = _mm512_mask_loadu_pd(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec = _mm512_load_pd(p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<8> const & mask, double const * p) {
            mVec = _mm512_mask_load_pd(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline double* store(double * p) const {
            _mm512_storeu_pd(p, mVec);
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<8> const & mask, double * p) const {
            _mm512_mask_storeu_pd(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline double* storea(double * p) const {
            _mm512_store_pd(p, mVec);
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<8> const & mask, double * p) const {
             _mm512_mask_store_pd(p, mask.mMask, mVec);
            return p;
        }
        
        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec, mask.mMask, b.mVec);
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec, mask.mMask, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            __m512d t0 = _mm512_add_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm512_add_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(double b) {
            mVec = _mm512_add_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
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
            __m512d t0 = mVec;
            mVec = _mm512_add_pd(mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<8> const & mask) {
            __m512d t0 = mVec;
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec = _mm512_add_pd(mVec, _mm512_set1_pd(1));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_add_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_sub_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            __m512d t0 = _mm512_sub_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm512_sub_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(const double b) {
            mVec = _mm512_sub_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<8> const & mask, const double b) {
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
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
        inline SIMDVec_f subfrom(SIMDVec_f const & a) const {
            __m512d t0 = _mm512_sub_pd(a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<8> const & mask, SIMDVec_f const & a) const {
            __m512d t0 = _mm512_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double a) const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(a), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<8> const & mask, double a) const {
            __m512d t0 = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask, _mm512_set1_pd(a), mVec);
            return SIMDVec_f(t0);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = _mm512_sub_pd(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, SIMDVec_f const & a) {
            mVec = _mm512_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(double a) {
            mVec = _mm512_sub_pd(_mm512_set1_pd(a), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<8> const & mask, double a) {
            mVec = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask, _mm512_set1_pd(a), mVec);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m512d t0 = mVec;
            mVec = _mm512_sub_pd(mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<8> const & mask) {
            __m512d t0 = mVec;
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return SIMDVec_f(t0);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec = _mm512_sub_pd(mVec, _mm512_set1_pd(1));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(1));
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            __m512d t0 = _mm512_mul_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm512_mul_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            mVec = _mm512_mul_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_div_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // DIVS
        inline SIMDVec_f div(double b) const {
            __m512d t0 = _mm512_div_pd(mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return SIMDVec_f(t0);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm512_div_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(double b) {
            mVec = _mm512_div_pd(mVec, _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, mVec, _mm512_set1_pd(b));
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m512d t0 = _mm512_div_pd(_mm512_set1_pd(1.0), mVec);
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(1.0), mVec);
            return SIMDVec_f(t0);
        }
        // RCPS
        inline SIMDVec_f rcp(double b) const {
            __m512d t0 = _mm512_div_pd(_mm512_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(b), mVec);
            return SIMDVec_f(t0);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec = _mm512_div_pd(_mm512_set1_pd(1.0), mVec);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(1.0), mVec);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(double b) {
            mVec = _mm512_div_pd(_mm512_set1_pd(b), mVec);
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<8> const & mask, double b) {
            mVec = _mm512_mask_div_pd(mVec, mask.mMask, _mm512_set1_pd(b), mVec);
            return *this;
        }

        // CMPEQV
        inline SIMDVecMask<8> cmpeq(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 0);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<8> cmpeq(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 0);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<8> cmpne(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 12);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<8> cmpne(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 12);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator!= (double b) const {
            return cmpne(b);
        }

        // CMPGTV
        inline SIMDVecMask<8> cmpgt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 30);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<8> cmpgt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 30);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<8> cmplt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 17);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<8> cmplt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 17);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<8> cmpge(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 29);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<8> cmpge(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 29);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<8> cmple(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 18);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<8> cmple(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 18);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        inline SIMDVecMask<8> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, b.mVec, 0);
            return (m0 == 0x03);
        }
        // CMPES
        inline bool cmpe(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec, t0, 0);
            return (m0 == 0x03);
        }
        // UNIQUE
        inline bool unique() const {
            alignas(16) double raw[2];
            _mm512_store_pd(raw, mVec);
            return raw[0] != raw[1];
        }
        // HADD
        inline double hadd() const {
            double retval = _mm512_reduce_add_pd(mVec);
            return retval;
        }
        // MHADD
        inline double hadd(SIMDVecMask<8> const & mask) const {
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, mVec);
            return retval;
        }
        // HADDS
        inline double hadd(double b) const {
            double retval = _mm512_reduce_add_pd(mVec);
            return retval + b;
        }
        // MHADDS
        inline double hadd(SIMDVecMask<8> const & mask, double b) const {
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, mVec);
            return retval + b;
        }
        // HMUL
        inline double hmul() const {
            double retval = _mm512_reduce_mul_pd(mVec);
            return retval;
        }
        // MHMUL
        inline double hmul(SIMDVecMask<8> const & mask) const {
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, mVec);
            return retval;
        }
        // HMULS
        inline double hmul(double b) const {
            double retval = _mm512_reduce_mul_pd(mVec);
            return b * retval;
        }
        // MHMULS
        inline double hmul(SIMDVecMask<8> const & mask, double b) const {
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, mVec);
            return b * retval;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmadd_pd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmadd_pd(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmsub_pd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmsub_pd(mVec, mask.mMask, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_add_pd(mVec, b.mVec);
            __m512d t1 = _mm512_mul_pd(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            __m512d t1 = _mm512_mask_mul_pd(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_sub_pd(mVec, b.mVec);
            __m512d t1 = _mm512_mul_pd(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<8> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            __m512d t1 = _mm512_mask_mul_pd(mVec, mask.mMask, t0, c.mVec);
            return SIMDVec_f(t1);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_max_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_max_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MAXS
        inline SIMDVec_f max(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_max_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_max_pd(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm512_max_pd(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_max_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_max_pd(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<8> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_mask_max_pd(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_min_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<8> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_min_pd(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MINS
        inline SIMDVec_f min(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_min_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<8> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_min_pd(mVec, mask.mMask, mVec, t0);
            return SIMDVec_f(t1);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm512_min_pd(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<8> const & mask, SIMDVec_f const & b) {
            mVec = _mm512_mask_min_pd(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_min_pd(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<8> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec = _mm512_mask_min_pd(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        inline double hmax() const {
            double retval = _mm512_reduce_max_pd(mVec);
            return retval;
        }
        // MHMAX
        inline double hmax(SIMDVecMask<8> const & mask) const {
            double retval = _mm512_mask_reduce_max_pd(mask.mMask, mVec);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline double hmin() const {
            double retval = _mm512_reduce_min_pd(mVec);
            return retval;
        }
        // MHMIN
        inline double hmin(SIMDVecMask<8> const & mask) const {
            double retval = _mm512_mask_reduce_min_pd(mask.mMask, mVec);
            return retval;
        }
        // IMIN
        // MIMIN

        // GATHERS
        //inline SIMDVec_f & gather(double * baseAddr, uint64_t * indices) {
        //    mVec[0] = baseAddr[indices[0]];
        //    mVec[1] = baseAddr[indices[1]];
        //    return *this;
        //}
        //// MGATHERS
        //inline SIMDVec_f & gather(SIMDVecMask<8> const & mask, double * baseAddr, uint64_t * indices) {
        //    if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
        //    if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
        //    return *this;
        //}
        //// GATHERV
        //inline SIMDVec_f & gather(double * baseAddr, VEC_UINT_TYPE const & indices) {
        //    mVec[0] = baseAddr[indices.mVec[0]];
        //    mVec[1] = baseAddr[indices.mVec[1]];
        //    return *this;
        //}
        //// MGATHERV
        //inline SIMDVec_f & gather(SIMDVecMask<8> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) {
        //    if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
        //    if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
        //    return *this;
        //}
        //// SCATTERS
        //inline double * scatter(double * baseAddr, uint64_t * indices) const {
        //    baseAddr[indices[0]] = mVec[0];
        //    baseAddr[indices[1]] = mVec[1];
        //    return baseAddr;
        //}
        //// MSCATTERS
        //inline double * scatter(SIMDVecMask<8> const & mask, double * baseAddr, uint64_t * indices) const {
        //    if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
        //    if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
        //    return baseAddr;
        //}
        //// SCATTERV
        //inline double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
        //    baseAddr[indices.mVec[0]] = mVec[0];
        //    baseAddr[indices.mVec[1]] = mVec[1];
        //    return baseAddr;
        //}
        //// MSCATTERV
        //inline double * scatter(SIMDVecMask<8> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
        //    if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
        //    if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
        //    return baseAddr;
        //}

        // NEG
        
        inline SIMDVec_f neg() const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_setzero_pd();
            __m512d t1 = _mm512_mask_sub_pd(mVec, mask.mMask, t0, mVec);
            return SIMDVec_f(t1);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<8> const & mask) {
            __m512d t0 = _mm512_setzero_pd();
            mVec = _mm512_mask_sub_pd(mVec, mask.mMask, t0, mVec);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512d t0 = _mm512_abs_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_abs_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            mVec = _mm512_abs_pd(mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_abs_pd(mVec, mask.mMask, mVec);
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        inline SIMDVec_f sqr() const {
            __m512d t0 = _mm512_mul_pd(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, mVec);
            return SIMDVec_f(t0);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec = _mm512_mul_pd(mVec, mVec);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_mul_pd(mVec, mask.mMask, mVec, mVec);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m512d t0 = _mm512_sqrt_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<8> const & mask) const {
            __m512d t0 = _mm512_mask_sqrt_pd(mVec, mask.mMask, mVec);
            return SIMDVec_f(t0);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = _mm512_sqrt_pd(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_sqrt_pd(mVec, mask.mMask, mVec);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec, 0);
            __m256d t1 = _mm512_extractf64x4_pd(mVec, 1);

            __m256d t2 = _mm256_round_pd(t0, _MM_FROUND_TO_NEAREST_INT);
            __m256d t3 = _mm256_round_pd(t1, _MM_FROUND_TO_NEAREST_INT);
            __m512d t4 = _mm512_castpd256_pd512(t2);
            __m512d t5 = _mm512_insertf64x4(t4, t3, 1);
            return SIMDVec_f(t5);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<8> const & mask) const {
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
        inline SIMDVec_i<int64_t, 8> trunc() const {
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
        inline SIMDVec_i<int64_t, 8> trunc(SIMDVecMask<8> const & mask) const {
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
        // EXP
        UME_FORCE_INLINE SIMDVec_f exp() const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 8>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<8> const & mask) const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 8>, SIMDVecMask<8>>(mask, *this);
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 8>>(*this);
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<8> const & mask) const {
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 8>, SIMDVecMask<8>>(mask, *this);
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this);
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<8> const & mask) const {
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this);
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<8> const & mask) const {
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(mask, *this);
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 8>, SIMDVecMask<8>>(*this, sinvec, cosvec);
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<8> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<8>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<8>>(mask, *this);
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
        inline operator SIMDVec_f<float, 8>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 8>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 8>() const;
    };

}
}

#endif
