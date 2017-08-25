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
            int64_t,
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

        UME_FORCE_INLINE SIMDVec_f(__m512d const & x0, __m512d const & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }

    public:
        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double d) {
            mVec[0] = _mm512_set1_pd(d);
            mVec[1] = _mm512_set1_pd(d);
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
            mVec[0] = _mm512_loadu_pd(p);
            mVec[1] = _mm512_loadu_pd(p + 8);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double d0,  double d1,  double d2,  double d3, 
                         double d4,  double d5,  double d6,  double d7,
                         double d8,  double d9,  double d10, double d11,
                         double d12, double d13, double d14, double d15) {
            mVec[0] = _mm512_set_pd(d7,  d6,  d5,  d4,  d3,  d2,  d1, d0);
            mVec[1] = _mm512_set_pd(d15, d14, d13, d12, d11, d10, d9, d8);
        }
        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
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
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
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
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<16>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

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
            mVec[0] = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec[0] = _mm512_set1_pd(b);
            mVec[1] = _mm512_set1_pd(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(b));
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec[0] = _mm512_loadu_pd(p);
            mVec[1] = _mm512_loadu_pd(p + 8);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<16> const & mask, double const * p) {
            mVec[0] = _mm512_mask_loadu_pd(mVec[0], mask.mMask & 0xFF, p);
            mVec[1] = _mm512_mask_loadu_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), p + 8);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec[0] = _mm512_load_pd(p);
            mVec[1] = _mm512_load_pd(p + 8);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<16> const & mask, double const * p) {
            mVec[0] = _mm512_mask_load_pd(mVec[0], mask.mMask & 0xFF, p);
            mVec[1] = _mm512_mask_load_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), p + 8);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE double* store(double * p) const {
            _mm512_storeu_pd(p, mVec[0]);
            _mm512_storeu_pd(p + 8, mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<16> const & mask, double * p) const {
            _mm512_mask_storeu_pd(p, mask.mMask & 0xFF, mVec[0]);
            _mm512_mask_storeu_pd(p + 8, ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE double* storea(double * p) const {
            _mm512_store_pd(p, mVec[0]);
            _mm512_store_pd(p + 8, mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<16> const & mask, double * p) const {
            _mm512_mask_store_pd(p, mask.mMask & 0xFF, mVec[0]);
            _mm512_mask_store_pd(p + 8, ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0]);
            __m512d t1 = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // SORTA
        // SORTD

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_add_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_add_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __m512d t0 = _mm512_add_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_add_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec[0] = _mm512_add_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            mVec[0] = _mm512_add_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_add_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
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
        UME_FORCE_INLINE SIMDVec_f sadd(double b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<16> const & mask, double b) const {
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
        UME_FORCE_INLINE SIMDVec_f & sadda(double b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<16> const & mask, double b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_add_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_add_pd(mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<16> const & mask) {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec[0] = _mm512_add_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_add_pd(mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_sub_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_sub_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __m512d t0 = _mm512_sub_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_sub_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec[0] = _mm512_sub_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const double b) {
            mVec[0] = _mm512_sub_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_sub_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<16> const & mask, const double b) {
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
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
        UME_FORCE_INLINE SIMDVec_f ssub(double b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<16> const & mask, double b) const {
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
        UME_FORCE_INLINE SIMDVec_f & ssuba(double b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<16> const & mask, double b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            __m512d t0 = _mm512_sub_pd(a.mVec[0], mVec[0]);
            __m512d t1 = _mm512_sub_pd(a.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<16> const & mask, SIMDVec_f const & a) const {
            __m512d t0 = _mm512_mask_sub_pd(a.mVec[0], mask.mMask & 0xFF, a.mVec[0], mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(a.mVec[1], ((mask.mMask & 0xFF00) >> 8), a.mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double a) const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(a), mVec[0]);
            __m512d t1 = _mm512_sub_pd(_mm512_set1_pd(a), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<16> const & mask, double a) const {
            __m512d t0 = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask & 0xFF, _mm512_set1_pd(a), mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(_mm512_set1_pd(a), ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(a), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec[0] = _mm512_sub_pd(a.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_pd(a.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, SIMDVec_f const & a) {
            mVec[0] = _mm512_mask_sub_pd(a.mVec[0], mask.mMask & 0xFF, a.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(a.mVec[1], ((mask.mMask & 0xFF00) >> 8), a.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(double a) {
            mVec[0] = _mm512_sub_pd(_mm512_set1_pd(a), mVec[0]);
            mVec[1] = _mm512_sub_pd(_mm512_set1_pd(a), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<16> const & mask, double a) {
            mVec[0] = _mm512_mask_sub_pd(_mm512_set1_pd(a), mask.mMask & 0xFF, _mm512_set1_pd(a), mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(_mm512_set1_pd(a), ((mask.mMask & 0xFF00) >> 8), _mm512_set1_pd(a), mVec[1]);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_sub_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_sub_pd(mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<16> const & mask) {
            __m512d t0 = mVec[0];
            __m512d t1 = mVec[1];
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return SIMDVec_f(t0, t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec[0] = _mm512_sub_pd(mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_sub_pd(mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(1));
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(1));
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mul_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mul_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __m512d t0 = _mm512_mul_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mul_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec[0] = _mm512_mul_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mul_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            mVec[0] = _mm512_mul_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mul_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_div_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_div_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            __m512d t0 = _mm512_div_pd(mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_div_pd(mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            __m512d t1 = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return SIMDVec_f(t0, t1);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec[0] = _mm512_div_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_div_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(double b) {
            mVec[0] = _mm512_div_pd(mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_div_pd(mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<16> const & mask, double b) {
            mVec[0] = _mm512_mask_div_pd(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_div_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_pd(b));
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            __m512d t0 = _mm512_rcp14_pd(mVec[0]);
            __m512d t1 = _mm512_rcp14_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
            __m512d t0 = _mm512_rcp14_pd(mVec[0]);
            __m512d t1 = _mm512_rcp14_pd(mVec[1]);
            __m512d t2 = _mm512_mul_pd(t0, _mm512_set1_pd(b));
            __m512d t3 = _mm512_mul_pd(t1, _mm512_set1_pd(b));
            return SIMDVec_f(t2, t3);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            __m512d t2 = _mm512_mask_mul_pd(t0, mask.mMask & 0xFF, t0, _mm512_set1_pd(b));
            __m512d t3 = _mm512_mask_mul_pd(t1, ((mask.mMask & 0xFF00) >> 8), t1, _mm512_set1_pd(b));
            return SIMDVec_f(t2, t3);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
            mVec[0] = _mm512_rcp14_pd(mVec[0]);
            mVec[1] = _mm512_rcp14_pd(mVec[1]);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(double b) {
            __m512d t0 = _mm512_rcp14_pd(mVec[0]);
            __m512d t1 = _mm512_rcp14_pd(mVec[1]);
            mVec[0] = _mm512_mul_pd(t0, _mm512_set1_pd(b));
            mVec[1] = _mm512_mul_pd(t1, _mm512_set1_pd(b));
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<16> const & mask, double b) {
            __m512d t0 = _mm512_mask_rcp14_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_rcp14_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            mVec[0] = _mm512_mask_mul_pd(t0, mask.mMask & 0xFF, t0, _mm512_set1_pd(b));
            mVec[1] = _mm512_mask_mul_pd(t1, ((mask.mMask & 0xFF00) >> 8), t1, _mm512_set1_pd(b));
            return *this;
        }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 0);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 0);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 12);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 12);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 12);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 12);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 30);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 30);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 30);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 30);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 17);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 17);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 17);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 17);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 29);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 29);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 29);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 29);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<16> cmple(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 18);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 18);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<16> cmple(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 18);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 18);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            SIMDVecMask<16> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], b.mVec[0], 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], b.mVec[1], 0);
            return ((m0 & m1)== 0xFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(mVec[0], t0, 0);
            __mmask8 m1 = _mm512_cmp_pd_mask(mVec[1], t0, 0);
            return ((m0 & m1) == 0xFF);
        }
        // UNIQUE
        // HADD
        UME_FORCE_INLINE double hadd() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[8];
            __m512d t0 = _mm512_add_pd(mVec[0], mVec[1]);
            _mm512_store_pd(raw, t0);
            return raw[0] + raw[1] + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7];
#else
            double retval = _mm512_reduce_add_pd(mVec[0]);
            retval += _mm512_reduce_add_pd(mVec[1]);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE double hadd(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[16];
            _mm512_store_pd(raw, mVec[0]);
            _mm512_store_pd((raw + 8), mVec[1]);
            double t0 = 0;
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
            double retval = _mm512_mask_reduce_add_pd(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE double hadd(double b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[8];
            __m512d t0 = _mm512_add_pd(mVec[0], mVec[1]);
            _mm512_store_pd(raw, t0);
            return b + raw[0] + raw[1] + raw[2]  + raw[3];
#else
            double retval = _mm512_reduce_add_pd(mVec[0]);
            retval += _mm512_reduce_add_pd(mVec[1]);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE double hadd(SIMDVecMask<16> const & mask, double b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[16];
            _mm512_store_pd(raw, mVec[0]);
            _mm512_store_pd((raw + 8), mVec[1]);
            double t0 = b;
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
            double retval = _mm512_mask_reduce_add_pd(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE double hmul() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[8];
            __m512d t0 = _mm512_add_pd(mVec[0], mVec[1]);
            _mm512_store_pd(raw, t0);
            return raw[0] * raw[1] * raw[2] * raw[3] * raw[4] * raw[5] * raw[6] * raw[7];
#else
            double retval = _mm512_reduce_mul_pd(mVec[0]);
            retval *= _mm512_reduce_mul_pd(mVec[1]);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE double hmul(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[16];
            _mm512_store_pd(raw, mVec[0]);
            _mm512_store_pd((raw + 8), mVec[1]);
            double t0 = 1;
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
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask & 0xFF, mVec[0]);
            retval *= _mm512_mask_reduce_mul_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE double hmul(double b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[8];
            __m512d t0 = _mm512_add_pd(mVec[0], mVec[1]);
            _mm512_store_pd(raw, t0);
            return b * raw[0] * raw[1] * raw[2] * raw[3] * raw[4] * raw[5] * raw[6] * raw[7];
#else
            double retval = _mm512_reduce_mul_pd(mVec[0]);
            retval *= _mm512_reduce_mul_pd(mVec[1]);
            return b * retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE double hmul(SIMDVecMask<16> const & mask, double b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[16];
            _mm512_store_pd(raw, mVec[0]);
            _mm512_store_pd((raw + 8), mVec[1]);
            double t0 = b;
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
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask & 0xFF, mVec[0]);
            retval *= _mm512_mask_reduce_mul_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return b * retval;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmadd_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_fmadd_pd(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmadd_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_mask_fmadd_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_fmsub_pd(mVec[0], b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_fmsub_pd(mVec[1], b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_fmsub_pd(mVec[0], mask.mMask & 0xFF, b.mVec[0], c.mVec[0]);
            __m512d t1 = _mm512_mask_fmsub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1], c.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_add_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_add_pd(mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mul_pd(t0, c.mVec[0]);
            __m512d t3 = _mm512_mul_pd(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_add_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_add_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, t0, c.mVec[0]);
            __m512d t3 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_sub_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_sub_pd(mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mul_pd(t0, c.mVec[0]);
            __m512d t3 = _mm512_mul_pd(t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<16> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m512d t0 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            __m512d t2 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, t0, c.mVec[0]);
            __m512d t3 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t1, c.mVec[1]);
            return SIMDVec_f(t2, t3);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_max_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_max_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_max_pd(mVec[0], t0);
            __m512d t2 = _mm512_max_pd(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            __m512d t2 = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec[0] = _mm512_max_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_max_pd(mVec[0], t0);
            mVec[1] = _mm512_max_pd(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<16> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_mask_max_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            mVec[1] = _mm512_mask_max_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m512d t0 = _mm512_min_pd(mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_min_pd(mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<16> const & mask, SIMDVec_f const & b) const {
            __m512d t0 = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512d t1 = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_min_pd(mVec[0], t0);
            __m512d t2 = _mm512_min_pd(mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<16> const & mask, double b) const {
            __m512d t0 = _mm512_set1_pd(b);
            __m512d t1 = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            __m512d t2 = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return SIMDVec_f(t1, t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec[0] = _mm512_min_pd(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_pd(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<16> const & mask, SIMDVec_f const & b) {
            mVec[0] = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_min_pd(mVec[0], t0);
            mVec[1] = _mm512_min_pd(mVec[1], t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<16> const & mask, double b) {
            __m512d t0 = _mm512_set1_pd(b);
            mVec[0] = _mm512_mask_min_pd(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            mVec[1] = _mm512_mask_min_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE double hmax() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[8];
            __m512d t0 = _mm512_max_pd(mVec[0], mVec[1]);
            _mm512_store_pd(raw, t0);
            double t1 = raw[0] > raw[1] ? raw[0] : raw[1];
            double t2 = raw[2] > raw[3] ? raw[2] : raw[3];
            double t3 = raw[4] > raw[5] ? raw[4] : raw[5];
            double t4 = raw[6] > raw[7] ? raw[6] : raw[7];
            double t5 = t1 > t2 ? t1 : t2;
            double t6 = t3 > t4 ? t3 : t4;
            return t5 > t6 ? t5 : t6;
#else
            double t0 = _mm512_reduce_max_pd(mVec[0]);
            double t1 = _mm512_reduce_max_pd(mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE double hmax(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[16];
            _mm512_store_pd(raw, mVec[0]);
            _mm512_store_pd((raw + 8), mVec[1]);
            double t0 =  ((mask.mMask & 0x0001) != 0) ? raw[0] : std::numeric_limits<double>::lowest();
            double t1 = (((mask.mMask & 0x0002) != 0) && raw[1] > t0) ? raw[1] : t0;
            double t2 = (((mask.mMask & 0x0004) != 0) && raw[2] > t1) ? raw[2] : t1;
            double t3 = (((mask.mMask & 0x0008) != 0) && raw[3] > t2) ? raw[3] : t2;
            double t4 = (((mask.mMask & 0x0010) != 0) && raw[4] > t3) ? raw[4] : t3;
            double t5 = (((mask.mMask & 0x0020) != 0) && raw[5] > t4) ? raw[5] : t4;
            double t6 = (((mask.mMask & 0x0040) != 0) && raw[6] > t5) ? raw[6] : t5;
            double t7 = (((mask.mMask & 0x0080) != 0) && raw[7] > t6) ? raw[7] : t6;
            double t8 = (((mask.mMask & 0x0100) != 0) && raw[8] > t7) ? raw[8] : t7;
            double t9 = (((mask.mMask & 0x0200) != 0) && raw[9] > t8) ? raw[9] : t8;
            double t10 = (((mask.mMask & 0x0400) != 0) && raw[10] > t9) ? raw[10] : t9;
            double t11 = (((mask.mMask & 0x0800) != 0) && raw[11] > t10) ? raw[11] : t10;
            double t12 = (((mask.mMask & 0x1000) != 0) && raw[12] > t11) ? raw[12] : t11;
            double t13 = (((mask.mMask & 0x2000) != 0) && raw[13] > t12) ? raw[13] : t12;
            double t14 = (((mask.mMask & 0x4000) != 0) && raw[14] > t13) ? raw[14] : t13;
            double t15 = (((mask.mMask & 0x8000) != 0) && raw[15] > t14) ? raw[15] : t14;
            return t15;
#else
            double t0 = _mm512_mask_reduce_max_pd(mask.mMask & 0xFF, mVec[0]);
            double t1 = _mm512_mask_reduce_max_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE double hmin() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[8];
            __m512d t0 = _mm512_min_pd(mVec[0], mVec[1]);
            _mm512_store_pd(raw, t0);
            double t1 = raw[0] < raw[1] ? raw[0] : raw[1];
            double t2 = raw[2] < raw[3] ? raw[2] : raw[3];
            double t3 = raw[4] < raw[5] ? raw[4] : raw[5];
            double t4 = raw[6] < raw[7] ? raw[6] : raw[7];
            double t5 = t1 < t2 ? t1 : t2;
            double t6 = t3 < t4 ? t3 : t4;
            return t5 < t6 ? t5 : t6;
#else
            double t0 = _mm512_reduce_min_pd(mVec[0]);
            double t1 = _mm512_reduce_min_pd(mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE double hmin(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) double raw[16];
            _mm512_store_pd(raw, mVec[0]);
            _mm512_store_pd((raw + 8), mVec[1]);
            double t0 =  ((mask.mMask & 0x0001) != 0) ? raw[0] : std::numeric_limits<double>::max();
            double t1 = (((mask.mMask & 0x0002) != 0) && raw[1] < t0) ? raw[1] : t0;
            double t2 = (((mask.mMask & 0x0004) != 0) && raw[2] < t1) ? raw[2] : t1;
            double t3 = (((mask.mMask & 0x0008) != 0) && raw[3] < t2) ? raw[3] : t2;
            double t4 = (((mask.mMask & 0x0010) != 0) && raw[4] < t3) ? raw[4] : t3;
            double t5 = (((mask.mMask & 0x0020) != 0) && raw[5] < t4) ? raw[5] : t4;
            double t6 = (((mask.mMask & 0x0040) != 0) && raw[6] < t5) ? raw[6] : t5;
            double t7 = (((mask.mMask & 0x0080) != 0) && raw[7] < t6) ? raw[7] : t6;
            double t8 = (((mask.mMask & 0x0100) != 0) && raw[8] < t7) ? raw[8] : t7;
            double t9 = (((mask.mMask & 0x0200) != 0) && raw[9] < t8) ? raw[9] : t8;
            double t10 = (((mask.mMask & 0x0400) != 0) && raw[10] < t9) ? raw[10] : t9;
            double t11 = (((mask.mMask & 0x0800) != 0) && raw[11] < t10) ? raw[11] : t10;
            double t12 = (((mask.mMask & 0x1000) != 0) && raw[12] < t11) ? raw[12] : t11;
            double t13 = (((mask.mMask & 0x2000) != 0) && raw[13] < t12) ? raw[13] : t12;
            double t14 = (((mask.mMask & 0x4000) != 0) && raw[14] < t13) ? raw[14] : t13;
            double t15 = (((mask.mMask & 0x8000) != 0) && raw[15] < t14) ? raw[15] : t14;
            return t15;
#else
            double t0 = _mm512_mask_reduce_min_pd(mask.mMask & 0xFF, mVec[0]);
            double t1 = _mm512_mask_reduce_min_pd(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }
        // IMIN
        // MIMIN

        // GATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(double const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t3 = _mm512_mullo_epi64(t0, t1);
            __m512i t4 = _mm512_mullo_epi64(t0, t2);
#else
            __m512i t3 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
            __m512i t4 = _mm512_setr_epi64(8*stride, 9*stride, 10*stride, 11*stride, 12*stride, 13*stride, 14*stride, 15*stride);
#endif
            mVec[0] = _mm512_i64gather_pd(t3, baseAddr, 8);
            mVec[1] = _mm512_i64gather_pd(t4, baseAddr, 8);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(SIMDVecMask<16> const & mask, double const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t3 = _mm512_mullo_epi64(t0, t1);
            __m512i t4 = _mm512_mullo_epi64(t0, t2);
#else
            __m512i t3 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
            __m512i t4 = _mm512_setr_epi64(8*stride, 9*stride, 10*stride, 11*stride, 12*stride, 13*stride, 14*stride, 15*stride);
#endif
            __mmask8 m0 = mask.mMask & 0x00FF;
            __mmask8 m1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_i64gather_pd(mVec[0], m0, t3, baseAddr, 8);
            mVec[1] = _mm512_mask_i64gather_pd(mVec[1], m1, t4, baseAddr, 8);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, uint64_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 8);
            mVec[0] = _mm512_i64gather_pd(t0, baseAddr, 8);
            mVec[1] = _mm512_i64gather_pd(t1, baseAddr, 8);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<16> const & mask, double const * baseAddr, uint64_t const * indices) {
            __mmask8 m0 = mask.mMask & 0x00FF;
            __mmask8 m1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 8);
            mVec[0] = _mm512_mask_i64gather_pd(mVec[0], m0, t0, baseAddr, 8);
            mVec[1] = _mm512_mask_i64gather_pd(mVec[1], m1, t1, baseAddr, 8);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, SIMDVec_u<uint64_t, 16> const & indices) {
            mVec[0] = _mm512_i64gather_pd(indices.mVec[0], baseAddr, 8);
            mVec[1] = _mm512_i64gather_pd(indices.mVec[1], baseAddr, 8);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<16> const & mask, double const * baseAddr, SIMDVec_u<uint64_t, 16> const & indices) {
            __mmask8 m0 = mask.mMask & 0x00FF;
            __mmask8 m1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_i64gather_pd(mVec[0], m0, indices.mVec[0], baseAddr, 8);
            mVec[1] = _mm512_mask_i64gather_pd(mVec[1], m1, indices.mVec[1], baseAddr, 8);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE double* scatteru(double* baseAddr, uint64_t stride) const {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t3 = _mm512_mullo_epi64(t0, t1);
            __m512i t4 = _mm512_mullo_epi64(t0, t2);
#else
            __m512i t3 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
            __m512i t4 = _mm512_setr_epi64(8*stride, 9*stride, 10*stride, 11*stride, 12*stride, 13*stride, 14*stride, 15*stride);
#endif
            _mm512_i64scatter_pd(baseAddr, t3, mVec[0], 8);
            _mm512_i64scatter_pd(baseAddr, t4, mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE double* scatteru(SIMDVecMask<16> const & mask, double* baseAddr, uint64_t stride) const {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_setr_epi64(8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t3 = _mm512_mullo_epi64(t0, t1);
            __m512i t4 = _mm512_mullo_epi64(t0, t2);
#else
            __m512i t3 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
            __m512i t4 = _mm512_setr_epi64(8*stride, 9*stride, 10*stride, 11*stride, 12*stride, 13*stride, 14*stride, 15*stride);
#endif
            __mmask8 m0 = mask.mMask & 0x00FF;
            __mmask8 m1 = (mask.mMask & 0xFF00) >> 8;
            _mm512_mask_i64scatter_pd(baseAddr, m0, t3, mVec[0], 8);
            _mm512_mask_i64scatter_pd(baseAddr, m1, t4, mVec[1], 8);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE double* scatter(double* baseAddr, uint64_t* indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 8);
            _mm512_i64scatter_pd(baseAddr, t0, mVec[0], 8);
            _mm512_i64scatter_pd(baseAddr, t1, mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE double* scatter(SIMDVecMask<16> const & mask, double* baseAddr, uint64_t* indices) {
            __mmask8 m0 = mask.mMask & 0x00FF;
            __mmask8 m1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices + 8);
            _mm512_mask_i64scatter_pd(baseAddr, m0, t0, mVec[0], 8);
            _mm512_mask_i64scatter_pd(baseAddr, m1, t1, mVec[1], 8);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE double* scatter(double* baseAddr, SIMDVec_u<uint64_t, 16> const & indices) {
            _mm512_i64scatter_pd(baseAddr, indices.mVec[0], mVec[0], 8);
            _mm512_i64scatter_pd(baseAddr, indices.mVec[1], mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE double* scatter(SIMDVecMask<16> const & mask, double* baseAddr, SIMDVec_u<uint64_t, 16> const & indices) {
            __mmask8 m0 = mask.mMask & 0x00FF;
            __mmask8 m1 = (mask.mMask & 0xFF00) >> 8;
            _mm512_mask_i64scatter_pd(baseAddr, m0, indices.mVec[0], mVec[0], 8);
            _mm512_mask_i64scatter_pd(baseAddr, m1, indices.mVec[1], mVec[1], 8);
            return baseAddr;
        }

        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m512d t0 = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[0]);
            __m512d t1 = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_setzero_pd();
            __m512d t1 = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, t0, mVec[0]);
            __m512d t2 = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t0, mVec[1]);
            return SIMDVec_f(t1, t2);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec[0] = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[0]);
            mVec[1] = _mm512_sub_pd(_mm512_set1_pd(0.0), mVec[1]);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<16> const & mask) {
            __m512d t0 = _mm512_setzero_pd();
            mVec[0] = _mm512_mask_sub_pd(mVec[0], mask.mMask & 0xFF, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t0, mVec[1]);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
#if defined (WA_GCC_INTR_SUPPORT_7_1)
            __m512i t0 = _mm512_castpd_si512(mVec[0]);
            __m512i t1 = _mm512_castpd_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t0, t2);
            __m512i t4 = _mm512_and_epi64(t1, t2);
            __m512d t5 = _mm512_castsi512_pd(t3);
            __m512d t6 = _mm512_castsi512_pd(t4);
            return SIMDVec_f(t5, t6);
#else
            __m512d t0 = _mm512_abs_pd(mVec[0]);
            __m512d t1 = _mm512_abs_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
#endif
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<16> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_7_1)
            __m512i t0 = _mm512_castpd_si512(mVec[0]);
            __m512i t1 = _mm512_castpd_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t0, t2);
            __m512i t4 = _mm512_and_epi64(t1, t2);
            __m512d t5 = _mm512_castsi512_pd(t3);
            __m512d t6 = _mm512_castsi512_pd(t4);
            __m512d t7 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, t5);
            __m512d t8 = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t6);
            return SIMDVec_f(t7, t8);
#else
            __m512d t0 = _mm512_mask_abs_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_abs_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_f(t0, t1);
#endif
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
#if defined (WA_GCC_INTR_SUPPORT_7_1)
            __m512i t0 = _mm512_castpd_si512(mVec[0]);
            __m512i t1 = _mm512_castpd_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t0, t2);
            __m512i t4 = _mm512_and_epi64(t1, t2);
            mVec[0] = _mm512_castsi512_pd(t3);
            mVec[1] = _mm512_castsi512_pd(t4);
            return *this;
#else
            mVec[0] = _mm512_abs_pd(mVec[0]);
            mVec[1] = _mm512_abs_pd(mVec[1]);
            return *this;
#endif
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<16> const & mask) {
#if defined (WA_GCC_INTR_SUPPORT_7_1)
            __m512i t0 = _mm512_castpd_si512(mVec[0]);
            __m512i t1 = _mm512_castpd_si512(mVec[1]);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t0, t2);
            __m512i t4 = _mm512_and_epi64(t1, t2);
            __m512d t5 = _mm512_castsi512_pd(t3);
            __m512d t6 = _mm512_castsi512_pd(t4);
            mVec[0] = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0xFF, t5);
            mVec[1] = _mm512_mask_mov_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), t6);
            return *this;
#else
            mVec[0] = _mm512_mask_abs_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_abs_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
#endif
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m512d t0 = _mm512_mul_pd(mVec[0], mVec[0]);
            __m512d t1 = _mm512_mul_pd(mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], mVec[0]);
            __m512d t1 = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec[0] = _mm512_mul_pd(mVec[0], mVec[0]);
            mVec[1] = _mm512_mul_pd(mVec[1], mVec[1]);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_mul_pd(mVec[0], mask.mMask & 0xFF, mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_mul_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], mVec[1]);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m512d t0 = _mm512_sqrt_pd(mVec[0]);
            __m512d t1 = _mm512_sqrt_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_mask_sqrt_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512d t1 = _mm512_mask_sqrt_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec[0] = _mm512_sqrt_pd(mVec[0]);
            mVec[1] = _mm512_sqrt_pd(mVec[1]);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_sqrt_pd(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_sqrt_pd(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
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
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<16> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_i<int64_t, 16> trunc() const {
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
        UME_FORCE_INLINE SIMDVec_i<int64_t, 16> trunc(SIMDVecMask<16> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m512d t0 = _mm512_floor_pd(mVec[0]);
            __m512d t1 = _mm512_floor_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_floor_pd(mVec[0]);
            __m512d t1 = _mm512_floor_pd(mVec[1]);
            __m512d t2 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0x00FF, t0);
            __m512d t3 = _mm512_mask_mov_pd(mVec[1], (mask.mMask & 0xFF00) >> 8, t1);
            return SIMDVec_f(t2, t3);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m512d t0 = _mm512_ceil_pd(mVec[0]);
            __m512d t1 = _mm512_ceil_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<16> const & mask) const {
            __m512d t0 = _mm512_ceil_pd(mVec[0]);
            __m512d t1 = _mm512_ceil_pd(mVec[1]);
            __m512d t2 = _mm512_mask_mov_pd(mVec[0], mask.mMask & 0x00FF, t0);
            __m512d t3 = _mm512_mask_mov_pd(mVec[1], (mask.mMask & 0xFF00) >> 8, t1);
            return SIMDVec_f(t2, t3);
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
            __m512d t0 = _mm512_exp_pd(mVec[0]);
            __m512d t1 = _mm512_exp_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 16>>(*this);
        #endif
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x00FF;
            __mmask16 m1 = (mask.mMask & 0xFF00) >> 8;
            __m512d t0 = _mm512_mask_exp_pd(mVec[0], m0, mVec[0]);
            __m512d t1 = _mm512_mask_exp_pd(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_log_pd(mVec[0]);
            __m512d t1 = _mm512_log_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 16>>(*this);
        #endif
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x00FF;
            __mmask16 m1 = (mask.mMask & 0xFF00) >> 8;
            __m512d t0 = _mm512_mask_log_pd(mVec[0], m0, mVec[0]);
            __m512d t1 = _mm512_mask_log_pd(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_sin_pd(mVec[0]);
            __m512d t1 = _mm512_sin_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(*this);
        #endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x00FF;
            __mmask16 m1 = (mask.mMask & 0xFF00) >> 16;
            __m512d t0 = _mm512_mask_sin_pd(mVec[0], m0, mVec[0]);
            __m512d t1 = _mm512_mask_sin_pd(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_cos_pd(mVec[0]);
            __m512d t1 = _mm512_cos_pd(mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(*this);
        #endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<16> const & mask) const {
        #if defined(UME_USE_SVML)
            __mmask16 m0 = mask.mMask & 0x00FF;
            __mmask16 m1 = (mask.mMask & 0xFF00) >> 16;
            __m512d t0 = _mm512_mask_cos_pd(mVec[0], m0, mVec[0]);
            __m512d t1 = _mm512_mask_cos_pd(mVec[1], m1, mVec[1]);
            return SIMDVec_f(t0, t1);
        #else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(mask, *this);
        #endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos0[8];
            alignas(64) double raw_cos1[8];
            sinvec.mVec[0] = _mm512_sincos_pd((__m512d*)raw_cos0, mVec[0]);
            sinvec.mVec[1] = _mm512_sincos_pd((__m512d*)raw_cos1, mVec[1]);
            cosvec.mVec[0] = _mm512_load_pd(raw_cos0);
            cosvec.mVec[1] = _mm512_load_pd(raw_cos1);
        #else
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 16>, SIMDVecMask<16>>(*this, sinvec, cosvec);
        #endif
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<16> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos0[8];
            alignas(64) double raw_cos1[8];
            __mmask16 m0 = mask.mMask & 0x00FF;
            __mmask16 m1 = (mask.mMask & 0xFF00) >> 16;
            sinvec.mVec[0] = _mm512_mask_sincos_pd((__m512d*)raw_cos0, mVec[0], mVec[0], m0, mVec[0]);
            sinvec.mVec[1] = _mm512_mask_sincos_pd((__m512d*)raw_cos1, mVec[1], mVec[1], m1, mVec[1]);
            cosvec.mVec[0] = _mm512_load_pd(raw_cos0);
            cosvec.mVec[1] = _mm512_load_pd(raw_cos1);
        #else
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<16>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<16>>(mask, *this);
        #endif
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
        UME_FORCE_INLINE operator SIMDVec_f<float, 16>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 16>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 16>() const;
    };

}
}

#endif
