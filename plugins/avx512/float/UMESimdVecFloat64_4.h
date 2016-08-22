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

#ifndef UME_SIMD_VEC_FLOAT64_4_H_
#define UME_SIMD_VEC_FLOAT64_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define EXPAND_CALL_UNARY(a_256d, unary_op) \
            _mm512_castpd512_pd256( \
                unary_op( \
                    _mm512_castpd256_pd512(a_256d)))

#define EXPAND_CALL_UNARY_MASK(a_256d, mask8, unary_op) \
            _mm512_castpd512_pd256( \
                unary_op( \
                    _mm512_castpd256_pd512(a_256d), \
                    mask8, \
                    _mm512_castpd256_pd512(a_256d)))

#define EXPAND_CALL_BINARY(a_256d, b_256d, binary_op) \
            _mm512_castpd512_pd256( \
                binary_op( \
                    _mm512_castpd256_pd512(a_256d), \
                    _mm512_castpd256_pd512(b_256d)))

#define EXPAND_CALL_BINARY_MASK(a_256d, b_256d, mask8, binary_op) \
            _mm512_castpd512_pd256( \
                binary_op( \
                    _mm512_castpd256_pd512(a_256d), \
                    mask8, \
                    _mm512_castpd256_pd512(a_256d), \
                    _mm512_castpd256_pd512(b_256d)))

#define EXPAND_CALL_BINARY_SCALAR_MASK(a_256d, b_64f, mask8, binary_op) \
            _mm512_castpd512_pd256( \
                binary_op( \
                    _mm512_castpd256_pd512(a_256d), \
                    mask8, \
                    _mm512_castpd256_pd512(a_256d), \
                    _mm512_set1_pd(b_64f)))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<double, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<double, 4>,
            SIMDVec_u<uint64_t, 4>,
            SIMDVec_i<int64_t, 4>,
            double,
            4,
            uint64_t,
            SIMDVecMask<4>,
            SIMDSwizzle<4>>,
        public SIMDVecPackableInterface<
            SIMDVec_f<double, 4>,
            SIMDVec_f<double, 2>>
    {
    private:
        __m256d mVec;

        typedef SIMDVec_u<uint64_t, 4>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int64_t, 4>     VEC_INT_TYPE;
        typedef SIMDVec_f<double, 2>      HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<double, 8>;

        inline SIMDVec_f(__m256d const & x) {
            mVec = x;
        }

    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_f() {}
        // SET-CONSTR
        inline explicit SIMDVec_f(double d) {
            mVec = _mm256_set1_pd(d);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_f(double const *p) {
            mVec = _mm256_loadu_pd(p);
        }
        // FULL-CONSTR
        inline SIMDVec_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_set_pd(d3, d2, d1, d0);
        }
        // EXTRACT
        inline double extract(uint32_t index) const {
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[index];
        }
        inline double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline SIMDVec_f & insert(uint32_t index, double value) {
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_pd(raw);
            return *this;
        }
        inline IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
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
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_pd(mVec, mask.mMask, b.mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __m512d t2 = _mm512_mask_mov_pd(t0, mask.mMask, t1);
            mVec = _mm512_castpd512_pd256(t2);
#endif
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_f & assign(double b) {
            mVec = _mm256_set1_pd(b);
            return *this;
        }
        inline SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_f & assign(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_pd(mVec, mask.mMask, _mm256_set1_pd(b));
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_mask_mov_pd(t0, mask.mMask, _mm512_set1_pd(b));
            mVec = _mm512_castpd512_pd256(t1);
#endif
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_f & load(double const * p) {
            mVec = _mm256_loadu_pd(p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_f & load(SIMDVecMask<4> const & mask, double const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_loadu_pd(mVec, mask.mMask, p);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m256d t1 = _mm256_loadu_pd(p);
            __m512d t2 = _mm512_castpd256_pd512(t1);
            __m512d t3 = _mm512_mask_mov_pd(t0, mask.mMask, t2);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // LOADA
        inline SIMDVec_f & loada(double const * p) {
            mVec = _mm256_load_pd(p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_f & loada(SIMDVecMask<4> const & mask, double const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_load_pd(mVec, mask.mMask, p);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m256d t1 = _mm256_loadu_pd(p);
            __m512d t2 = _mm512_castpd256_pd512(t1);
            __m512d t3 = _mm512_mask_mov_pd(t0, mask.mMask, t2);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // STORE
        inline double* store(double * p) const {
            _mm256_storeu_pd(p, mVec);
            return p;
        }
        // MSTORE
        inline double* store(SIMDVecMask<4> const & mask, double * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_storeu_pd(p, mask.mMask, mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m256d t1 = _mm256_loadu_pd(p);
            __m512d t2 = _mm512_castpd256_pd512(t1);
            __m512d t3 = _mm512_mask_mov_pd(t2, mask.mMask, t0);
            __m256d t4 = _mm512_castpd512_pd256(t3);
            _mm256_storeu_pd(p, t4);
#endif
            return p;
        }
        // STOREA
        inline double* storea(double * p) const {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTOREA
        inline double* storea(SIMDVecMask<4> const & mask, double * p) const {
#if defined(__AVX512VL__)
             _mm256_mask_store_pd(p, mask.mMask, mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m256d t1 = _mm256_load_pd(p);
            __m512d t2 = _mm512_castpd256_pd512(t1);
            __m512d t3 = _mm512_mask_mov_pd(t2, mask.mMask, t0);
            __m256d t4 = _mm512_castpd512_pd256(t3);
            _mm256_store_pd(p, t4);
#endif
            return p;
        }

        // BLENDV
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mov_pd(mVec, mask.mMask, b.mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_mask_mov_pd(t1, mask.mMask, t2);
            __m256d t0 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // BLENDS
        inline SIMDVec_f blend(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mov_pd(mVec, mask.mMask, _mm256_set1_pd(b));
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_set1_pd(b);
            __m512d t3 = _mm512_mask_mov_pd(t1, mask.mMask, t2);
            __m256d t0 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_pd);
#endif
            return SIMDVec_f(t0);
        }
        // ADDS
        inline SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_add_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_f add(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_pd);
#endif
            return SIMDVec_f(t0);
        }
        // ADDVA
        inline SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_pd);
#endif
            return *this;
        }
        // ADDSA
        inline SIMDVec_f & adda(double b) {
            mVec = _mm256_add_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_f & adda(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_pd);
#endif
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
            __m256d t0 = mVec;
            mVec = _mm256_add_pd(mVec, _mm256_set1_pd(1));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m256d t0 = mVec;
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(1));
#else
            __m256d t0 = mVec;
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_pd);
#endif
            return SIMDVec_f(t0);
        }
        // PREFINC
        inline SIMDVec_f & prefinc() {
            mVec = _mm256_add_pd(mVec, _mm256_set1_pd(1));
            return *this;
        }
        inline SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_pd);
#endif
            return *this;
        }
        // SUBV
        inline SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBS
        inline SIMDVec_f sub(double b) const {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_f sub(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBVA
        inline SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm256_sub_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // SUBSA
        inline SIMDVec_f & suba(const double b) {
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_f & suba(SIMDVecMask<4> const & mask, const double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_pd);
#endif
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
            __m256d t0 = _mm256_sub_pd(a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & a) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(a.mVec, mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        inline SIMDVec_f subfrom(double a) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(a), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        inline SIMDVec_f subfrom(SIMDVecMask<4> const & mask, double a) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(_mm256_set1_pd(a), mask.mMask, _mm256_set1_pd(a), mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(_mm256_set1_pd(a), mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = _mm256_sub_pd(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & a) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(a.mVec, mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_f & subfroma(double a) {
            mVec = _mm256_sub_pd(_mm256_set1_pd(a), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, double a) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(_mm256_set1_pd(a), mask.mMask, _mm256_set1_pd(a), mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(_mm256_set1_pd(a), mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // POSTDEC
        inline SIMDVec_f postdec() {
            __m256d t0 = mVec;
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(1));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m256d t0 = mVec;
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(1));
#else
            __m256d t0 = mVec;
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // PREFDEC
        inline SIMDVec_f & prefdec() {
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(1));
            return *this;
        }
        inline SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // MULV
        inline SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_mul_pd);
#endif
            return SIMDVec_f(t0);
        }
        // MULS
        inline SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_mul_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_f mul(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_mul_pd);
#endif
            return SIMDVec_f(t0);
        }
        // MULVA
        inline SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm256_mul_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_mul_pd);
#endif
            return *this;
        }
        // MULSA
        inline SIMDVec_f & mula(double b) {
            mVec = _mm256_mul_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_f & mula(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_mul_pd);
#endif
            return *this;
        }
        // DIVV
        inline SIMDVec_f div(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_div_pd);
#endif
            return SIMDVec_f(t0);
        }
        // DIVS
        inline SIMDVec_f div(double b) const {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        inline SIMDVec_f div(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_div_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_div_pd);
#endif
            return SIMDVec_f(t0);
        }
        // DIVVA
        inline SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm256_div_pd(mVec, b.mVec);
            return *this;
        }
        inline SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_div_pd);
#endif
            return *this;
        }
        // DIVSA
        inline SIMDVec_f & diva(double b) {
            mVec = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        inline SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        inline SIMDVec_f & diva(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_div_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_div_pd);
#endif
            return *this;
        }
        // RCP
        inline SIMDVec_f rcp() const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_rcp14_pd(mVec);
#else
            __m256d t0 = EXPAND_CALL_UNARY(mVec, _mm512_rcp14_pd);
#endif
            return SIMDVec_f(t0);
        }
        // MRCP
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_rcp14_pd(mVec, mask.mMask, mVec);
#else
            __m256d t0 = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_rcp14_pd);
#endif
            return SIMDVec_f(t0);
        }
        // RCPS
        inline SIMDVec_f rcp(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_rcp14_pd(mVec);
            __m256d t1 = _mm256_mul_pd(t0, _mm256_set1_pd(b));
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_rcp14_pd(t0);
            __m512d t3 = _mm512_mul_pd(t2, _mm512_set1_pd(b));
            __m256d t1 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MRCPS
        inline SIMDVec_f rcp(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_rcp14_pd(mVec, mask.mMask, mVec);
            __m256d t1 = _mm256_mask_mul_pd(t0, mask.mMask, t0, _mm256_set1_pd(b));
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_rcp14_pd(t0, mask.mMask, t0);
            __m512d t3 = _mm512_mask_mul_pd(t2, mask.mMask, t2, _mm512_set1_pd(b));
            __m256d t1 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // RCPA
        inline SIMDVec_f & rcpa() {
#if defined(__AVX512VL__)
            mVec = _mm256_rcp14_pd(mVec);
#else
            mVec = EXPAND_CALL_UNARY(mVec, _mm512_rcp14_pd);
#endif
            return *this;
        }
        // MRCPA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_rcp14_pd(mVec, mask.mMask, mVec);
#else
            mVec = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_rcp14_pd);
#endif
            return *this;
        }
        // RCPSA
        inline SIMDVec_f & rcpa(double b) {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_rcp14_pd(mVec);
            mVec = _mm256_mul_pd(t0, _mm256_set1_pd(b));
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_rcp14_pd(t0);
            __m512d t3 = _mm512_mul_pd(t2, _mm512_set1_pd(b));
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // MRCPSA
        inline SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_rcp14_pd(mVec, mask.mMask, mVec);
            mVec = _mm256_mask_mul_pd(t0, mask.mMask, t0, _mm256_set1_pd(b));
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_rcp14_pd(t0, mask.mMask, t0);
            __m512d t3 = _mm512_mask_mul_pd(t2, mask.mMask, t2, _mm512_set1_pd(b));
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }

        // CMPEQV
        inline SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 0);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 0);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 0);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 0);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 12);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 12);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 12);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 12);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 30);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 30);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 30);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 30);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 17);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 17);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 17);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 17);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 29);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 29);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 29);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 29);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 18);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 18);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 18);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 18);
#endif
            return SIMDVecMask<4>(m0);
        }
        inline SIMDVecMask<4> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 0);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 0);
#endif
            return (m0 == 0x03);
        }
        // CMPES
        inline bool cmpe(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 0);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 0);
#endif
            return (m0 == 0x03);
        }
        // UNIQUE
        inline bool unique() const {
            alignas(16) double raw[2];
            _mm256_store_pd(raw, mVec);
            return raw[0] != raw[1];
        }
        // HADD
        inline double hadd() const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(0xF, t0);
            return retval;
        }
        // MHADD
        inline double hadd(SIMDVecMask<4> const & mask) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, t0);
            return retval;
        }
        // HADDS
        inline double hadd(double b) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(0xF, t0);
            return retval + b;
        }
        // MHADDS
        inline double hadd(SIMDVecMask<4> const & mask, double b) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, t0);
            return retval + b;
        }
        // HMUL
        inline double hmul() const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(0xF, t0);
            return retval;
        }
        // MHMUL
        inline double hmul(SIMDVecMask<4> const & mask) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, t0);
            return retval;
        }
        // HMULS
        inline double hmul(double b) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(0xF, t0);
            return b * retval;
        }
        // MHMULS
        inline double hmul(SIMDVecMask<4> const & mask, double b) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, t0);
            return b * retval;
        }

        // FMULADDV
        inline SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_fmadd_pd(t1, t2, t3);
            __m256d t0 = _mm512_castpd512_pd256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        inline SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_fmadd_pd(mVec, mask.mMask, b.mVec, c.mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_mask_fmadd_pd(t1, mask.mMask, t2, t3);
            __m256d t0 = _mm512_castpd512_pd256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // FMULSUBV
        inline SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_fmsub_pd(mVec, b.mVec, c.mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_fmsub_pd(t1, t2, t3);
            __m256d t0 = _mm512_castpd512_pd256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        inline SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_fmsub_pd(mVec, mask.mMask, b.mVec, c.mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_mask_fmsub_pd(t1, mask.mMask, t2, t3);
            __m256d t0 = _mm512_castpd512_pd256(t4);
#endif
            return SIMDVec_f(t0);
        }
        // FADDMULV
        inline SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d t1 = _mm256_mul_pd(t0, c.mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_add_pd(t0, t2);
            __m512d t5 = _mm512_mul_pd(t4, t3);
            __m256d t1 = _mm512_castpd512_pd256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        inline SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
            __m256d t1 = _mm256_mask_mul_pd(mVec, mask.mMask, t0, c.mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_mask_add_pd(t0, mask.mMask, t0, t2);
            __m512d t5 = _mm512_mask_mul_pd(t4, mask.mMask, t4, t3);
            __m256d t1 = _mm512_castpd512_pd256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // FSUBMULV
        inline SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            __m256d t1 = _mm256_mul_pd(t0, c.mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_sub_pd(t0, t2);
            __m512d t5 = _mm512_mul_pd(t4, t3);
            __m256d t1 = _mm512_castpd512_pd256(t5);
#endif
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        inline SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
            __m256d t1 = _mm256_mask_mul_pd(mVec, mask.mMask, t0, c.mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_castpd256_pd512(c.mVec);
            __m512d t4 = _mm512_mask_sub_pd(t0, mask.mMask, t0, t2);
            __m512d t5 = _mm512_mask_mul_pd(t4, mask.mMask, t4, t3);
            __m256d t1 = _mm512_castpd512_pd256(t5);
#endif
            return SIMDVec_f(t1);
        }

        // MAXV
        inline SIMDVec_f max(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_max_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_max_pd(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_mask_max_pd(t1, mask.mMask, t1, t2);
            __m256d t0 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MAXS
        inline SIMDVec_f max(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_max_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        inline SIMDVec_f max(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mask_max_pd(mVec, mask.mMask, mVec, t0);
#else 
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_set1_pd(b);
            __m512d t3 = _mm512_mask_max_pd(t0, mask.mMask, t0, t2);
            __m256d t1 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MAXVA
        inline SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm256_max_pd(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_max_pd(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_mask_max_pd(t1, mask.mMask, t1, t2);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // MAXSA
        inline SIMDVec_f & maxa(double b) {
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_max_pd(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_f & maxa(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_mask_max_pd(mVec, mask.mMask, mVec, t0);
#else 
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_set1_pd(b);
            __m512d t3 = _mm512_mask_max_pd(t0, mask.mMask, t0, t2);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // MINV
        inline SIMDVec_f min(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_min_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_min_pd(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_mask_min_pd(t1, mask.mMask, t1, t2);
            __m256d t0 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t0);
        }
        // MINS
        inline SIMDVec_f min(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_min_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        inline SIMDVec_f min(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_mask_min_pd(mVec, mask.mMask, mVec, t0);
#else 
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_set1_pd(b);
            __m512d t3 = _mm512_mask_min_pd(t0, mask.mMask, t0, t2);
            __m256d t1 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // MINVA
        inline SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm256_min_pd(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_min_pd(mVec, mask.mMask, mVec, b.mVec);
#else 
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_castpd256_pd512(b.mVec);
            __m512d t3 = _mm512_mask_min_pd(t1, mask.mMask, t1, t2);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // MINSA
        inline SIMDVec_f & mina(double b) {
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_min_pd(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_f & mina(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_mask_min_pd(mVec, mask.mMask, mVec, t0);
#else 
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_set1_pd(b);
            __m512d t3 = _mm512_mask_min_pd(t0, mask.mMask, t0, t2);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // HMAX
        inline double hmax() const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_max_pd(0xF, t0);
            return retval;
        }
        // MHMAX
        inline double hmax(SIMDVecMask<4> const & mask) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_max_pd(mask.mMask, t0);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline double hmin() const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_min_pd(0xF, t0);
            return retval;
        }
        // MHMIN
        inline double hmin(SIMDVecMask<4> const & mask) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_min_pd(mask.mMask, t0);
            return retval;
        }
        // IMIN
        // MIMIN

        // GATHERS
/*        inline SIMDVec_f & gather(double * baseAddr, uint64_t * indices) {
            mVec[0] = baseAddr[indices[0]];
            mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // MGATHERS
        inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, double * baseAddr, uint64_t * indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
            return *this;
        }
        // GATHERV
        inline SIMDVec_f & gather(double * baseAddr, VEC_UINT_TYPE const & indices) {
            mVec[0] = baseAddr[indices.mVec[0]];
            mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // MGATHERV
        inline SIMDVec_f & gather(SIMDVecMask<4> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) {
            if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
            if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
            return *this;
        }
        // SCATTERS
        inline double * scatter(double * baseAddr, uint64_t * indices) const {
            baseAddr[indices[0]] = mVec[0];
            baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERS
        inline double * scatter(SIMDVecMask<4> const & mask, double * baseAddr, uint64_t * indices) const {
            if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
            return baseAddr;
        }
        // SCATTERV
        inline double * scatter(double * baseAddr, VEC_UINT_TYPE const & indices) const {
            baseAddr[indices.mVec[0]] = mVec[0];
            baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }
        // MSCATTERV
        inline double * scatter(SIMDVecMask<4> const & mask, double * baseAddr, VEC_UINT_TYPE const & indices) const {
            if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
            if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
            return baseAddr;
        }*/
        // NEG
        inline SIMDVec_f neg() const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return SIMDVec_f(t0);
        }
        inline SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_setzero_pd();
            __m256d t1 = _mm256_mask_sub_pd(mVec, mask.mMask, t0, mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_setzero_pd();
            __m512d t3 = _mm512_mask_sub_pd(t0, mask.mMask, t2, t0);
            __m256d t1 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // NEGA
        inline SIMDVec_f & nega() {
            mVec = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_setzero_pd();
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, t0, mVec);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_setzero_pd();
            __m512d t3 = _mm512_mask_sub_pd(t0, mask.mMask, t2, t0);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // ABS
        inline SIMDVec_f abs() const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_abs_pd(t0);
            __m256d t2 = _mm512_castpd512_pd256(t1);
            return SIMDVec_f(t2);
        }
        // MABS
        inline SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_abs_pd(t0, mask.mMask, t0);
            __m256d t3 = _mm512_castpd512_pd256(t2);
            return SIMDVec_f(t3);
        }
        // ABSA
        inline SIMDVec_f & absa() {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_abs_pd(t0);
            mVec = _mm512_castpd512_pd256(t1);
            return *this;
        }
        // MABSA
        inline SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_abs_pd(t0, mask.mMask, t0);
            mVec = _mm512_castpd512_pd256(t2);
            return *this;
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        inline SIMDVec_f sqr() const {
            __m256d t0 = _mm256_mul_pd(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        inline SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_mul_pd(t1, mask.mMask, t1, t1);
            __m256d t0 = _mm512_castpd512_pd256(t2);
#endif
            return SIMDVec_f(t0);
        }
        // SQRA
        inline SIMDVec_f & sqra() {
            mVec = _mm256_mul_pd(mVec, mVec);
            return *this;
        }
        // MSQRA
        inline SIMDVec_f & sqra(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_mul_pd(t1, mask.mMask, t1, t1);
            mVec = _mm512_castpd512_pd256(t2);
#endif
            return *this;
        }
        // SQRT
        inline SIMDVec_f sqrt() const {
            __m256d t0 = _mm256_sqrt_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        inline SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sqrt_pd(mVec, mask.mMask, mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_sqrt_pd(t1, mask.mMask, t1);
            __m256d t0 = _mm512_castpd512_pd256(t2);
#endif
            return SIMDVec_f(t0);
        }
        // SQRTA
        inline SIMDVec_f & sqrta() {
            mVec = _mm256_sqrt_pd(mVec);
            return *this;
        }
        // MSQRTA
        inline SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sqrt_pd(mVec, mask.mMask, mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_sqrt_pd(t1, mask.mMask, t1);
            mVec = _mm512_castpd512_pd256(t2);
#endif
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        inline SIMDVec_f round() const {
            __m256d t0 = _mm256_round_pd(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        inline SIMDVec_f round(SIMDVecMask<4> const & mask) const {
            __m256d t0 = _mm256_round_pd(mVec, _MM_FROUND_TO_NEAREST_INT);
#if defined(__AVX512VL__)
            __m256d t1 = _mm256_mask_mov_pd(mVec, mask.mMask, t0);
#else
            __m512d t2 = _mm512_castpd256_pd512(t0);
            __m512d t3 = _mm512_mask_mov_pd(t2, mask.mMask, t2);
            __m256d t1 = _mm512_castpd512_pd256(t3);
#endif
            return SIMDVec_f(t1);
        }
        // TRUNC
        inline SIMDVec_i<int64_t, 4> trunc() const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_cvttpd_epi64(mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512i t2 = _mm512_cvttpd_epi64(t1);
            __m256i t0 = _mm512_castsi512_si256(t2);
#endif
            return SIMDVec_i<int64_t, 4>(t0);
#else
            alignas(32) double raw_d[4];
            alignas(32) int64_t raw_i[4];
            _mm256_store_pd(raw_d, mVec);
            raw_i[0] = (int64_t)raw_d[0];
            raw_i[1] = (int64_t)raw_d[1];
            raw_i[2] = (int64_t)raw_d[2];
            raw_i[3] = (int64_t)raw_d[3];
            return SIMDVec_i<int64_t, 4>(raw_i);
#endif
        }
        // MTRUNC
        inline SIMDVec_i<int64_t, 4> trunc(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_cvttpd_epi64(_mm256_setzero_si256(), mask.mMask, mVec);
#else
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512i t2 = _mm512_setzero_si512();
            __m512i t3 = _mm512_mask_cvttpd_epi64(t2, mask.mMask, t1);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i<int64_t, 4>(t0);
#else
            alignas(32) double raw_d[4];
            alignas(32) int64_t raw_i[4];
            _mm256_store_pd(raw_d, mVec);
            raw_i[0] = (mask.mMask & 0x1) ? (int64_t)raw_d[0] : 0;
            raw_i[1] = (mask.mMask & 0x2) ? (int64_t)raw_d[1] : 0;
            raw_i[2] = (mask.mMask & 0x4) ? (int64_t)raw_d[2] : 0;
            raw_i[3] = (mask.mMask & 0x8) ? (int64_t)raw_d[3] : 0;
            return SIMDVec_i<int64_t, 4>(raw_i);
#endif
        }
        // FLOOR
        inline SIMDVec_f floor() const {
            __m256d t0 = _mm256_floor_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        inline SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __m256d t0 = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_floor_pd);
            return SIMDVec_f(t0);
        }
        // CEIL
        inline SIMDVec_f ceil() const {
            __m256d t0 = _mm256_ceil_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        inline SIMDVec_f ceil(SIMDVecMask<4> const & mask) const {
            __m256d t0 = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_ceil_pd);
            return SIMDVec_f(t0);
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
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 4>>(*this);
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<4> const & mask) const {
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 4>, SIMDVecMask<4>> (mask, *this);
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 4>>(*this);
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<4> const & mask) const {
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 4>, SIMDVecMask<4>>(mask, *this);
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(*this);
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<4> const & mask) const {
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(mask, *this);
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(*this);
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<4> const & mask) const {
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(mask, *this);
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(*this, sinvec, cosvec);
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<4> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<4>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<4>>(mask, *this);
        }
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
        SIMDVec_f & pack(SIMDVec_f<double, 2> const & a, SIMDVec_f<double, 2> const & b) {
            mVec = _mm256_insertf128_pd(mVec, a.mVec, 0);
            mVec = _mm256_insertf128_pd(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        SIMDVec_f & packlo(SIMDVec_f<double, 2> const & a) {
            mVec = _mm256_insertf128_pd(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        SIMDVec_f & packhi(SIMDVec_f<double, 2> const & b) {
            mVec = _mm256_insertf128_pd(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        void unpack(SIMDVec_f<double, 2> & a, SIMDVec_f<double, 2> & b) const {
            a.mVec = _mm256_extractf128_pd(mVec, 0);
            b.mVec = _mm256_extractf128_pd(mVec, 1);
        }
        // UNPACKLO
        SIMDVec_f<double, 2> unpacklo() const {
            __m128d t0 = _mm256_extractf128_pd(mVec, 0);
            return SIMDVec_f<double, 2>(t0);
        }
        // UNPACKHI
        SIMDVec_f<double, 2> unpackhi() const {
            __m128d t0 = _mm256_extractf128_pd(mVec, 1);
            return SIMDVec_f<double, 2>(t0);
        }

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_f<float, 4>() const;

        // FTOU
        inline operator SIMDVec_u<uint64_t, 4>() const;
        // FTOI
        inline operator SIMDVec_i<int64_t, 4>() const;
    };

}
}

#undef EXPAND_CALL_UNARY
#undef EXPAND_CALL_UNARY_MASK
#undef EXPAND_CALL_BINARY
#undef EXPAND_CALL_BINARY_MASK
#undef EXPAND_CALL_BINARY_SCALAR_MASK

#endif
