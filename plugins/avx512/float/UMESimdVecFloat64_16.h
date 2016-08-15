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

#ifndef UME_SIMD_VEC_FLOAT64_16_H_
#define UME_SIMD_VEC_FLOAT64_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 16> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 16>,
            SIMDVec_u<uint64_t, 16>,
            SIMDVec_i<int64_t, 16>,
            double,
            16,
            uint64_t,
            SIMDVecMask<16>,
            SIMDSwizzle<16>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 16>,
            SIMDVec_f<double, 8>>
    {
    private:
        __m512d mVec[2];

        typedef SIMDVec_u<uint64_t, 16>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 16>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 8>      HALF_LEN_VEC_TYPE;

        inline SIMDVec_f(__m512d const & x0, __m512d const & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }

    public:
        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline SIMDVec_f(double d) {
            mVec[0] = _mm512_set1_pd(d);
            mVec[1] = _mm512_set1_pd(d);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_f(double const *p) {
            mVec[0] = _mm512_loadu_pd(p);
            mVec[1] = _mm512_loadu_pd(p + 8);
        }
        // FULL-CONSTR
        inline SIMDVec_f(double d0,  double d1,  double d2,  double d3, 
                         double d4,  double d5,  double d6,  double d7,
                         double d8,  double d9,  double d10, double d11,
                         double d12, double d13, double d14, double d15) {
            mVec[0] = _mm512_set_pd(d7,  d6,  d5,  d4,  d3,  d2,  d1, d0);
            mVec[1] = _mm512_set_pd(d15, d14, d13, d12, d11, d10, d9, d8);
        }
        // EXTRACT
        inline double extract(uint32_t index) const {
            alignas(64) double raw[8];
            if (index < 8) {
                _mm512_store_pd(raw, mVec[0]);
                return raw[index];
            }
            else {
                _mm512_store_pd(raw, mVec[1]);
                return raw[index - 8];
            }
        }
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
            alignas(64) double raw[8];
            if (index < 8) {
                _mm512_store_pd(raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm512_load_pd(raw);
            }
            else {
                _mm512_store_pd(raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm512_load_pd(raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        inline SIMDVec_f & assign(SIMDVec_f const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        inline SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_f & assign(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec[0] = _mm512_set1_pd(b);
            mVec[1] = _mm512_set1_pd(b);
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(b));
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec[0] = _mm512_loadu_pd(p);
            mVec[1] = _mm512_loadu_pd(p + 8);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<16> const & mask, double const * p) {
            mVec[0] = _mm512_mask_loadu_pd(mVec[0], mask.mMask & 0xFF, p);
            mVec[1] = _mm512_mask_loadu_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), p + 8);
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec[0] = _mm512_load_pd(p);
            mVec[1] = _mm512_load_pd(p + 8);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<16> const & mask, double const * p) {
            mVec[0] = _mm512_mask_load_pd(mVec[0], mask.mMask & 0xFF, p);
            mVec[1] = _mm512_mask_load_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), p + 8);
            return *this;
        }
        // STORE
        inline double* store(double * p) const {
            _mm512_storeu_pd(p, mVec[0]);
            _mm512_storeu_pd(p + 8, mVec[1]);
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<16> const & mask, double * p) const {
            _mm512_mask_storeu_pd(p, mask.mMask & 0xFF, mVec[0]);
            _mm512_mask_storeu_pd(p + 8, ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return p;
        }
        // STOREA
        inline double* storea(double * p) const {
            _mm512_store_pd(p, mVec[0]);
            _mm512_store_pd(p + 8, mVec[1]);
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<16> const & mask, double * p) const {
            _mm512_mask_store_pd(p, mask.mMask & 0xFF, mVec[0]);
            _mm512_mask_store_pd(p + 8, ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0]);
            __m512d t1 = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // SORTA
        // SORTD

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_add_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_add_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            __m512d t0 = _mm512_add_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_add_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm512_add_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(double b) {
            mVec[0] = _mm512_add_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_add_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
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
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_add_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_add_pd(mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<16> const & mask) {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec[0] = _mm512_add_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_add_pd(mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_sub_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_sub_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            __m512d t0 = _mm512_sub_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_sub_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = _mm512_sub_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(const double b) {
            mVec[0] = _mm512_sub_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_sub_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<16> const & mask, const double b) {
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
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
            __m512d t0 = _mm512_sub_pd(a.mVec[0], mVec[0]);
            __m512d t1 = _mm512_sub_pd(a.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<16> const & mask, SIMDVec_f const & a) const {
            __m512d t0 = _mm512_mask_sub_pd(a.mVec[0], mask.mMask & 0xFF, a.mVec[0], mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(a.mVec[1], ((mask.mMask & 0xFF00) >> 8), a.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double a) const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(a), mVec[0]);
            __m512d t1 = _mm512_sub_pd(_mm512_set1_pd(a), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<16> const & mask, double a) const {
            __m512d t0 = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask & 0xFF, _mm512_set1_pd(a), mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(_mm512_set1_pd(a), ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(a), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec[0] = _mm512_sub_pd(a.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_pd(a.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, SIMDVec_f const & a) {
            mVec[0] = _mm512_mask_sub_pd(a.mVec[0], mask.mMask & 0xFF, a.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(a.mVec[1], ((mask.mMask & 0xFF00) >> 8), a.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(double a) {
            mVec[0] = _mm512_sub_pd(_mm512_set1_pd(a), mVec[0]);
            mVec[1] = _mm512_sub_pd(_mm512_set1_pd(a), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, double a) {
            mVec[0] = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask & 0xFF, _mm512_set1_pd(a), mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(_mm512_set1_pd(a), ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(a), mVec[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_sub_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_sub_pd(mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<16> const & mask) {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec[0] = _mm512_sub_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_sub_pd(mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mul_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mul_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            __m512d t0 = _mm512_mul_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mul_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm512_mul_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mul_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            mVec[0] = _mm512_mul_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mul_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_div_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_div_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        inline SIMDVec_f div(double b) const {
            __m512d t0 = _mm512_div_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_div_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] = _mm512_div_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_div_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(double b) {
            mVec[0] = _mm512_div_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_div_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
            __m512d t0 = _mm512_rcp14_pd(mVec[0]);
            __m512d t1 = _mm512_rcp14_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        inline SIMDVec_f rcp(double b) const {
            __m512d t0 = _mm512_rcp14_pd(mVec[0]);
            __m512d t1 = _mm512_rcp14_pd(mVec[1]);
            __m512d t2 = _mm512_mul_pd(t0, _mm512_set1_pd(b));
            __m512d t3 = _mm512_mul_pd(t1, _mm512_set1_pd(b));
            return SIMDVec_f(t2, t3);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            __m512d t2 = _mm512_mask_mul_pd(t0, mask.mMask & 0xFF, t0, _mm512_set1_pd(b));
            __m512d t3 = _mm512_mask_mul_pd(t1, ((mask.mMask & 0xFF00) >> 8), t1, _mm512_set1_pd(b));
            return SIMDVec_f(t2, t3);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
            mVec[0] = _mm512_rcp14_pd(mVec[0]);
            mVec[1] = _mm512_rcp14_pd(mVec[1]);
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(double b) {
            __m512d t0 = _mm512_rcp14_pd(mVec[0]);
            __m512d t1 = _mm512_rcp14_pd(mVec[1]);
            mVec[0] = _mm512_mul_pd(t0, _mm512_set1_pd(b));
            mVec[1] = _mm512_mul_pd(t1, _mm512_set1_pd(b));
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<16> const & mask, double b) {
            __m512d t0 = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            mVec[0] = _mm512_mask_mul_pd(t0, mask.mMask & 0xFF, t0, _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_mul_pd(t1, ((mask.mMask & 0xFF00) >> 8), t1, _mm512_set1_pd(b));
            return *this;
        }

        // CMPEQV
        inline SIMDVecMask<16> cmpeq(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 0);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 0);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 12);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 12);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 12);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 12);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 30);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 30);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 30);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 30);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 17);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 17);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 17);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 17);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 29);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 29);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 29);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 29);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 18);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 18);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 18);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 18);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<16> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 0);
            return ((m0 & m1)== 0xFF);
        }
        // CMPES
        inline bool cmpe(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 0);
            return ((m0 & m1) == 0xFF);
        }
        // UNIQUE
        // HADD
        inline double hadd() const {
            double retval = _mm512_reduce_add_pd(mVec[0]);
            retval += _mm512_reduce_add_pd(mVec[1]);
            return retval;
        }
        // MHADD
        inline double hadd(SIMDVecMask<16> const & mask) const {
            double retval = _mm512_mask_reduce_add_pd(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval;
        }
        // HADDS
        inline double hadd(double b) const {
            double retval = _mm512_reduce_add_pd(mVec[0]);
            retval += _mm512_reduce_add_pd(mVec[1]);
            return retval + b;
        }
        // MHADDS
        inline double hadd(SIMDVecMask<16> const & mask, double b) const {
            double retval = _mm512_mask_reduce_add_pd(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval + b;
        }
        // HMUL
        inline double hmul() const {
            double retval = _mm512_reduce_mul_pd(mVec[0]);
            retval *= _mm512_reduce_mul_pd(mVec[1]);
            return retval;
        }
        // MHMUL
        inline double hmul(SIMDVecMask<16> const & mask) const {
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask & 0xFF, mVec[0]);
            retval *= _mm512_mask_reduce_mul_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval;
        }
        // HMULS
        inline double hmul(double b) const {
            double retval = _mm512_reduce_mul_pd(mVec[0]);
            retval *= _mm512_reduce_mul_pd(mVec[1]);
            return b * retval;
        }
        // MHMULS
        inline double hmul(SIMDVecMask<16> const & mask, double b) const {
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask & 0xFF, mVec[0]);
            retval *= _mm512_mask_reduce_mul_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return b * retval;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmadd_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_fmadd_pd(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmadd_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_mask_fmadd_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmsub_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_fmsub_pd(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmsub_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_mask_fmsub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_add_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_add_pd(mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mul_pd(t0, c.mVec[0]);
            __m512d t3 = _mm512_mul_pd(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, t0, c.mVec[0]);
            __m512d t3 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_sub_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_sub_pd(mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mul_pd(t0, c.mVec[0]);
            __m512d t3 = _mm512_mul_pd(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, t0, c.mVec[0]);
            __m512d t3 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_max_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_max_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MAXS
        inline SIMDVec_f max(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_max_pd(mVec[0], t0);
            __m512d t2 = _mm512_max_pd(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            __m512d t2 = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = _mm512_max_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_max_pd(mVec[0], t0);
            mVec[1] = _mm512_max_pd(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<16> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            mVec[1] = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_min_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_min_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MINS
        inline SIMDVec_f min(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_min_pd(mVec[0], t0);
            __m512d t2 = _mm512_min_pd(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            __m512d t2 = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = _mm512_min_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_min_pd(mVec[0], t0);
            mVec[1] = _mm512_min_pd(mVec[1], t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<16> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            mVec[1] = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return *this;
        }
        // HMAX
        inline double hmax() const {
            double t0 = _mm512_reduce_max_pd(mVec[0]);
            double t1 = _mm512_reduce_max_pd(mVec[1]);
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        inline double hmax(SIMDVecMask<16> const & mask) const {
            double t0 = _mm512_mask_reduce_max_pd(mask.mMask & 0xFF, mVec[0]);
            double t1 = _mm512_mask_reduce_max_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return t0 > t1 ? t0 : t1;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline double hmin() const {
            double t0 = _mm512_reduce_min_pd(mVec[0]);
            double t1 = _mm512_reduce_min_pd(mVec[1]);
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        inline double hmin(SIMDVecMask<16> const & mask) const {
            double t0 = _mm512_mask_reduce_min_pd(mask.mMask & 0xFF, mVec[0]);
            double t1 = _mm512_mask_reduce_min_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return t0 < t1 ? t0 : t1;
        }
        // IMIN
        // MIMIN

        // GATHERS
/*        inline SIMDVec_f & gather(double * baseAddr, uint64_t * indices) {
            mVec[0][0] = baseAddr[indices[0]];
            mVec[0][1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<16> const & mask, double * baseAddr, uint64_t * indices) {
            if (mask.mMask[0] == true) mVec[0][0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[0][1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(double * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec[0][0] = baseAddr[indices.mVec[0][0]];
            mVec[0][1] = baseAddr[indices.mVec[0][1]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<16> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask[0] == true) mVec[0][0] = baseAddr[indices.mVec[0][0]];
            if (mask.mMask[1] == true) mVec[0][1] = baseAddr[indices.mVec[0][1]];
            return *this;
        }
        // SCATTERS
        inline double * scatter(double * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec[0][0];
            baseAddr[indices[1]] = mVec[0][1];
            return baseAddr;
        }
        // MSCATTERS
        inline double * scatter(SIMDVecMask<16> const & mask, double * baseAddr, uint64_t * indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0][0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[0][1];
            return baseAddr;
        }
        // SCATTERV
        inline double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices.mVec[0][0]] = mVec[0][0];
            baseAddr[indices.mVec[0][1]] = mVec[0][1];
            return baseAddr;
        }
        // MSCATTERV
        inline double * scatter(SIMDVecMask<16> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0][0]] = mVec[0][0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[0][1]] = mVec[0][1];
            return baseAddr;
        }*/

        // NEG
        inline SIMDVec_f neg() const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[0]);
            __m512d t1 = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_setzero_pd();
            __m512d t1 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, t0, mVec[0]);
            __m512d t2 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec[0] = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[0]);
            mVec[1] = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[1]);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<16> const & mask) {
            __m512d t0 = _mm512_setzero_pd();
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t0, mVec[1]);
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512d t0 = _mm512_abs_pd(mVec[0]);
            __m512d t1 = _mm512_abs_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_abs_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_abs_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            mVec[0] = _mm512_abs_pd(mVec[0]);
            mVec[1] = _mm512_abs_pd(mVec[1]);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_abs_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_abs_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        inline SIMDVec_f sqr() const {
            __m512d t0 = _mm512_mul_pd(mVec[0], mVec[0]);
            __m512d t1 = _mm512_mul_pd(mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], mVec[0]);
            __m512d t1 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec[0] = _mm512_mul_pd(mVec[0], mVec[0]);
            mVec[1] = _mm512_mul_pd(mVec[1], mVec[1]);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], mVec[1]);
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m512d t0 = _mm512_sqrt_pd(mVec[0]);
            __m512d t1 = _mm512_sqrt_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_sqrt_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_sqrt_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec[0] = _mm512_sqrt_pd(mVec[0]);
            mVec[1] = _mm512_sqrt_pd(mVec[1]);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_sqrt_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_sqrt_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec[0], 0);
            __m256d t1 = _mm512_extractf64x4_pd(mVec[0], 1);

            __m256d t2 = _mm256_round_pd(t0, _MM_FROUND_TO_NEAREST_INT);
            __m256d t3 = _mm256_round_pd(t1, _MM_FROUND_TO_NEAREST_INT);
            __m512d t4 = _mm512_castpd256_pd512(t2);
            __m512d t5 = _mm512_insertf64x4(t4, t3, 1);

            __m256d t6 = _mm512_extractf64x4_pd(mVec[1], 0);
            __m256d t7 = _mm512_extractf64x4_pd(mVec[1], 1);

            __m256d t8 = _mm256_round_pd(t6, _MM_FROUND_TO_NEAREST_INT);
            __m256d t9 = _mm256_round_pd(t7, _MM_FROUND_TO_NEAREST_INT);
            __m512d t10 = _mm512_castpd256_pd512(t8);
            __m512d t11 = _mm512_insertf64x4(t10, t9, 1);
            return SIMDVec_f(t5, t11);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<16> const & mask) const {
            __m256d t0 = _mm512_extractf64x4_pd(mVec[0], 0);
            __m256d t1 = _mm512_extractf64x4_pd(mVec[0], 1);

            __m256d t2 = _mm256_round_pd(t0, _MM_FROUND_TO_NEAREST_INT);
            __m256d t3 = _mm256_round_pd(t1, _MM_FROUND_TO_NEAREST_INT);
            __m512d t4 = _mm512_castpd256_pd512(t2);
            __m512d t5 = _mm512_insertf64x4(t4, t3, 1);
            __m512d t6 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, t5);

            __m256d t7 = _mm512_extractf64x4_pd(mVec[1], 0);
            __m256d t8 = _mm512_extractf64x4_pd(mVec[1], 1);

            __m256d t9 = _mm256_round_pd(t7, _MM_FROUND_TO_NEAREST_INT);
            __m256d t10 = _mm256_round_pd(t8, _MM_FROUND_TO_NEAREST_INT);
            __m512d t11 = _mm512_castpd256_pd512(t9);
            __m512d t12 = _mm512_insertf64x4(t11, t10, 1);
            __m512d t13 = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t12);
            return SIMDVec_f(t6, t13);
        }
        // TRUNC
        inline SIMDVec_i<int64_t, 16> trunc() const {
#if defined(__AVX512DQ__)
            __m512i t0 = _mm512_cvttpd_epi64(mVec[0]);
            __m512i t1 = _mm512_cvttpd_epi64(mVec[1]);
            return SIMDVec_i<int64_t, 16>(t0, t1);
#else
            alignas(64) double raw_d[16];
            alignas(64) int64_t raw_i[16];
            _mm512_store_pd(raw_d, mVec[0]);
            _mm512_store_pd(&raw_d[8], mVec[1]);
            raw_i[0] = (int64_t)raw_d[0];
            raw_i[1] = (int64_t)raw_d[1];
            raw_i[2] = (int64_t)raw_d[2];
            raw_i[3] = (int64_t)raw_d[3];
            raw_i[4] = (int64_t)raw_d[4];
            raw_i[5] = (int64_t)raw_d[5];
            raw_i[6] = (int64_t)raw_d[6];
            raw_i[7] = (int64_t)raw_d[7];

            raw_i[8] = (int64_t)raw_d[8];
            raw_i[9] = (int64_t)raw_d[9];
            raw_i[10] = (int64_t)raw_d[10];
            raw_i[11] = (int64_t)raw_d[11];
            raw_i[12] = (int64_t)raw_d[12];
            raw_i[13] = (int64_t)raw_d[13];
            raw_i[14] = (int64_t)raw_d[14];
            raw_i[15] = (int64_t)raw_d[15];
            __m512i t0 = _mm512_load_epi64(&raw_i[0]);
            __m512i t1 = _mm512_load_epi64(&raw_i[8]);
            return SIMDVec_i<int64_t, 16>(t0, t1);
#endif
        }
        // MTRUNC
        inline SIMDVec_i<int64_t, 16> trunc(SIMDVecMask<16> const & mask) const {
#if defined(__AVX512DQ__)
            __m512i t0 = _mm512_mask_cvttpd_epi64(_mm512_setzero_si512(), mask.mMask & 0xFF, mVec[0]);
            __m512i t1 = _mm512_mask_cvttpd_epi64(_mm512_setzero_si512(), ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_i<int64_t, 16>(t0, t1);
#else
            alignas(64) double raw_d[16];
            alignas(64) int64_t raw_i[16];
            __m512d t0 = _mm512_set1_pd(0.0);
            __m512d t1 = _mm512_mask_mov_pd(t0, (mask.mMask & 0x00FF), mVec[0]);
            __m512d t2 = _mm512_mask_mov_pd(t0, (mask.mMask & 0xFF00) >> 8, mVec[1]);
            _mm512_store_pd(&raw_d[0], t1);
            _mm512_store_pd(&raw_d[8], t2);
            raw_i[0] = (int64_t)raw_d[0];
            raw_i[1] = (int64_t)raw_d[1];
            raw_i[2] = (int64_t)raw_d[2];
            raw_i[3] = (int64_t)raw_d[3];
            raw_i[4] = (int64_t)raw_d[4];
            raw_i[5] = (int64_t)raw_d[5];
            raw_i[6] = (int64_t)raw_d[6];
            raw_i[7] = (int64_t)raw_d[7];
            raw_i[8] = (int64_t)raw_d[8];
            raw_i[9] = (int64_t)raw_d[9];
            raw_i[10] = (int64_t)raw_d[10];
            raw_i[11] = (int64_t)raw_d[11];
            raw_i[12] = (int64_t)raw_d[12];
            raw_i[13] = (int64_t)raw_d[13];
            raw_i[14] = (int64_t)raw_d[14];
            raw_i[15] = (int64_t)raw_d[15];
            __m512i t3 = _mm512_load_epi64(&raw_i[0]);
            __m512i t4 = _mm512_load_epi64(&raw_i[8]);
            return SIMDVec_i<int64_t, 16>(t3, t4);
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
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 16>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<16> const & mask) const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 16>, SIMDVecMask<16>>(mask, *this);
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 16>>(*this);
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<16> const & mask) const {
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 16>, SIMDVecMask<16>>(mask, *this);
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(*this);
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<16> const & mask) const {
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(mask, *this);
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(*this);
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<16> const & mask) const {
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(mask, *this);
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(*this, sinvec, cosvec);
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<16> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<16>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<16>>(mask, *this);
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // PROMOTE
        // -    
        // DEGRADE
        inline operator SIMDVec_f<float, 16>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 16>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 16>() const;
    };

}
}

#endif
