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

        UME_FORCE_INLINE SIMDVec_f(__m256d const & x) {
            mVec = x;
        }

    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double d) {
            mVec = _mm256_set1_pd(d);
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
            mVec = _mm256_loadu_pd(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(double d0, double d1, double d2, double d3) {
            mVec = _mm256_set_pd(d3, d2, d1, d0);
        }
        // EXTRACT
        UME_FORCE_INLINE double extract(uint32_t index) const {
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE double operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, double value) {
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_pd(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, double> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, double>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, double, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
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
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FORCE_INLINE SIMDVec_f & assign(double b) {
            mVec = _mm256_set1_pd(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (double b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, double b) {
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
        UME_FORCE_INLINE SIMDVec_f & load(double const * p) {
            mVec = _mm256_loadu_pd(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<4> const & mask, double const * p) {
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
        UME_FORCE_INLINE SIMDVec_f & loada(double const * p) {
            mVec = _mm256_load_pd(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<4> const & mask, double const * p) {
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
        UME_FORCE_INLINE double* store(double * p) const {
            _mm256_storeu_pd(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE double* store(SIMDVecMask<4> const & mask, double * p) const {
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
        UME_FORCE_INLINE double* storea(double * p) const {
            _mm256_store_pd(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE double* storea(SIMDVecMask<4> const & mask, double * p) const {
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
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, double b) const {
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
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_pd);
#endif
            return SIMDVec_f(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(double b) const {
            __m256d t0 = _mm256_add_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (double b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_pd);
#endif
            return SIMDVec_f(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = _mm256_add_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_pd);
#endif
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(double b) {
            mVec = _mm256_add_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (double b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_pd);
#endif
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            return add(b);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            return add(mask, b);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(double b) const {
            return add(b);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<4> const & mask, double b) const {
            return add(mask, b);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            return adda(b);
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            return adda(mask, b);
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(double b) {
            return adda(b);
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<4> const & mask, double b) {
            return adda(mask, b);
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __m256d t0 = mVec;
            mVec = _mm256_add_pd(mVec, _mm256_set1_pd(1));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            mVec = _mm256_add_pd(mVec, _mm256_set1_pd(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_pd);
#endif
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(double b) const {
            __m256d t0 = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (double b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = _mm256_sub_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(const double b) {
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (double b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, const double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            return sub(mask, b);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(double b) const {
            return sub(b);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<4> const & mask, double b) const {
            return sub(mask, b);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
            return suba(b);
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            return suba(mask, b);
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(double b) {
            return suba(b);
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, double b) {
            return suba(mask, b);
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & a) const {
            __m256d t0 = _mm256_sub_pd(a.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & a) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(a.mVec, mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(double a) const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(a), mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, double a) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_sub_pd(_mm256_set1_pd(a), mask.mMask, _mm256_set1_pd(a), mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(_mm256_set1_pd(a), mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return SIMDVec_f(t0);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & a) {
            mVec = _mm256_sub_pd(a.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & a) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(a.mVec, mask.mMask, a.mVec, mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(a.mVec, mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(double a) {
            mVec = _mm256_sub_pd(_mm256_set1_pd(a), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, double a) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(_mm256_set1_pd(a), mask.mMask, _mm256_set1_pd(a), mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(_mm256_set1_pd(a), mVec, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __m256d t0 = mVec;
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(1));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            mVec = _mm256_sub_pd(mVec, _mm256_set1_pd(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_pd);
#endif
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_mul_pd);
#endif
            return SIMDVec_f(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(double b) const {
            __m256d t0 = _mm256_mul_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (double b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_mul_pd);
#endif
            return SIMDVec_f(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = _mm256_mul_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_mul_pd);
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(double b) {
            mVec = _mm256_mul_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (double b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mul_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_mul_pd);
#endif
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_div_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256d t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_div_pd);
#endif
            return SIMDVec_f(t0);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(double b) const {
            __m256d t0 = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (double b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_div_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            __m256d t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_div_pd);
#endif
            return SIMDVec_f(t0);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = _mm256_div_pd(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_div_pd(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_div_pd);
#endif
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(double b) {
            mVec = _mm256_div_pd(mVec, _mm256_set1_pd(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (double b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, double b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_div_pd(mVec, mask.mMask, mVec, _mm256_set1_pd(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_div_pd);
#endif
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_rcp14_pd(mVec);
#else
            __m256d t0 = EXPAND_CALL_UNARY(mVec, _mm512_rcp14_pd);
#endif
            return SIMDVec_f(t0);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_mask_rcp14_pd(mVec, mask.mMask, mVec);
#else
            __m256d t0 = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_rcp14_pd);
#endif
            return SIMDVec_f(t0);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(double b) const {
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
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask, double b) const {
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
        UME_FORCE_INLINE SIMDVec_f & rcpa() {
#if defined(__AVX512VL__)
            mVec = _mm256_rcp14_pd(mVec);
#else
            mVec = EXPAND_CALL_UNARY(mVec, _mm512_rcp14_pd);
#endif
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_rcp14_pd(mVec, mask.mMask, mVec);
#else
            mVec = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_rcp14_pd);
#endif
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_f & rcpa(double b) {
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
        UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, double b) {
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
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 0);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 0);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 0);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 0);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (double b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 12);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 12);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 12);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 12);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (double b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 30);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 30);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 30);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 30);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (double b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 17);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 17);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 17);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 17);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (double b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 29);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 29);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 29);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 29);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (double b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, b.mVec, 18);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_castpd256_pd512(b.mVec);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 18);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(double b) const {
#if defined(__AVX512VL__)
            __m256d t0 = _mm256_set1_pd(b);
            __mmask8 m0 = _mm256_cmp_pd_mask(mVec, t0, 18);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_set1_pd(b);
            __mmask8 m0 = _mm512_cmp_pd_mask(t0, t1, 18);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (double b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
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
        UME_FORCE_INLINE bool cmpe(double b) const {
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
        UME_FORCE_INLINE bool unique() const {
            alignas(16) double raw[2];
            _mm256_store_pd(raw, mVec);
            return raw[0] != raw[1];
        }
        // HADD
        UME_FORCE_INLINE double hadd() const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(0xF, t0);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE double hadd(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, t0);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE double hadd(double b) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(0xF, t0);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE double hadd(SIMDVecMask<4> const & mask, double b) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = b;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_add_pd(mask.mMask, t0);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE double hmul() const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(0xF, t0);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE double hmul(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, t0);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE double hmul(double b) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(0xF, t0);
            return b * retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE double hmul(SIMDVecMask<4> const & mask, double b) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = b;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_mul_pd(mask.mMask, t0);
            return b * retval;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__FMA__)
            __m256d t0 = _mm256_fmadd_pd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
#else
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d t1 = _mm256_add_pd(t0, c.mVec);
            return SIMDVec_f(t1);
#endif
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
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
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
#if defined(__FMA__)
            __m256d t0 = _mm256_fmsub_pd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
#else
            __m256d t0 = _mm256_mul_pd(mVec, b.mVec);
            __m256d t1 = _mm256_sub_pd(t0, c.mVec);
            return SIMDVec_f(t1);
#endif
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
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
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256d t0 = _mm256_add_pd(mVec, b.mVec);
            __m256d t1 = _mm256_mul_pd(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
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
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __m256d t0 = _mm256_sub_pd(mVec, b.mVec);
            __m256d t1 = _mm256_mul_pd(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
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
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_max_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FORCE_INLINE SIMDVec_f max(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_max_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, double b) const {
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
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = _mm256_max_pd(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FORCE_INLINE SIMDVec_f & maxa(double b) {
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_max_pd(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, double b) {
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
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __m256d t0 = _mm256_min_pd(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
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
        UME_FORCE_INLINE SIMDVec_f min(double b) const {
            __m256d t0 = _mm256_set1_pd(b);
            __m256d t1 = _mm256_min_pd(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, double b) const {
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
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = _mm256_min_pd(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
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
        UME_FORCE_INLINE SIMDVec_f & mina(double b) {
            __m256d t0 = _mm256_set1_pd(b);
            mVec = _mm256_min_pd(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, double b) {
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
        UME_FORCE_INLINE double hmax() const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = raw[0] > raw[1] ? raw[0] : raw[1];
            double t1 = raw[2] > raw[3] ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_max_pd(0xF, t0);
            return retval;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE double hmax(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = (mask.mMask & 0x1) ? raw[0] : std::numeric_limits<double>::lowest();
            double t1 = ((mask.mMask & 0x2) && raw[1] > t0) ? raw[1] : t0;
            double t2 = ((mask.mMask & 0x4) && raw[2] > t1) ? raw[2] : t1;
            double t3 = ((mask.mMask & 0x8) && raw[3] > t2) ? raw[3] : t2;
            return t3;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_max_pd(mask.mMask, t0);
            return retval;
#endif
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE double hmin() const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = raw[0] < raw[1] ? raw[0] : raw[1];
            double t1 = raw[2] < raw[3] ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_min_pd(0xF, t0);
            return retval;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE double hmin(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(32) double raw[4];
            _mm256_store_pd(raw, mVec);
            double t0 = (mask.mMask & 0x1) ? raw[0] : std::numeric_limits<double>::lowest();
            double t1 = ((mask.mMask & 0x2) && raw[1] < t0) ? raw[1] : t0;
            double t2 = ((mask.mMask & 0x4) && raw[2] < t1) ? raw[2] : t1;
            double t3 = ((mask.mMask & 0x8) && raw[3] < t2) ? raw[3] : t2;
            return t3;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            double retval = _mm512_mask_reduce_min_pd(mask.mMask, t0);
            return retval;
#endif
        }
        // IMIN
        // MIMIN

        // GATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(double const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m256i t0 = _mm256_set1_epi64x(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
            mVec = _mm256_i64gather_pd(baseAddr, t2, 8);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_f & gatheru(SIMDVecMask<4> const & mask, double const * baseAddr, uint64_t stride) {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm256_set1_epi64x(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i64gather_pd(mVec, mask.mMask, t2, baseAddr, 8);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512d t4 = _mm512_castpd256_pd512(mVec);
            __m512d t5 = _mm512_mask_i64gather_pd(t4, mask.mMask & 0xF, t3, baseAddr, 8);
            mVec = _mm512_castpd512_pd256(t5);
#endif
            return *this;
        }
        //GATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, uint64_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec = _mm256_i64gather_pd(baseAddr, t0, 8);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, double const * baseAddr, uint64_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i64gather_pd(mVec, mask.mMask, t0, baseAddr, 8);
#else
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512d t2 = _mm512_castpd256_pd512(mVec);
            __m512d t3 = _mm512_mask_i64gather_pd(t2, mask.mMask & 0xF, t1, baseAddr, 8);
            mVec = _mm512_castpd512_pd256(t3);
#endif
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(double const * baseAddr, SIMDVec_u<uint64_t, 4> const & indices) {
            mVec = _mm256_i64gather_pd(baseAddr, indices.mVec, 8);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, double const * baseAddr, SIMDVec_u<uint64_t, 4> const & indices) {
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i64gather_pd(mVec, mask.mMask, indices.mVec, baseAddr, 8);
#else
            __m512i t0 = _mm512_castsi256_si512(indices.mVec);
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_i64gather_pd(t1, mask.mMask & 0xF, t0, baseAddr, 8);
            mVec = _mm512_castpd512_pd256(t2);
#endif
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE double* scatteru(double* baseAddr, uint64_t stride) const {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm256_set1_epi64x(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
#if defined(__AVX512VL__)
            _mm256_i64scatter_pd(baseAddr, t2, mVec, 8);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512d t4 = _mm512_castpd256_pd512(mVec);
            _mm512_mask_i64scatter_pd(baseAddr, 0xF, t3, t4, 8);
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE double*  scatteru(SIMDVecMask<4> const & mask, double* baseAddr, uint64_t stride) const {
#if defined(__AVX512DQ__)
            __m256i t0 = _mm256_set1_epi64x(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
#if defined(__AVX512VL__)
            _mm256_mask_i64scatter_pd(baseAddr, mask.mMask, t2, mVec, 8);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512d t4 = _mm512_castpd256_pd512(mVec);
            _mm512_mask_i64scatter_pd(baseAddr, mask.mMask, t3, t4, 8);
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE double* scatter(double* baseAddr, uint64_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
#if defined(__AVX512VL__)
            _mm256_i64scatter_pd(baseAddr, t0, mVec, 8);
#else
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512d t2 = _mm512_castpd256_pd512(mVec);
            _mm512_mask_i64scatter_pd(baseAddr, 0xF, t1, t2, 8);
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE double* scatter(SIMDVecMask<4> const & mask, double* baseAddr, uint64_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
#if defined(__AVX512VL__)
            _mm256_mask_i64scatter_pd(baseAddr, mask.mMask, t0, mVec, 8);
#else
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512d t2 = _mm512_castpd256_pd512(mVec);
            _mm512_mask_i64scatter_pd(baseAddr, mask.mMask & 0xF, t1, t2, 8);
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE double* scatter(double* baseAddr, SIMDVec_u<uint64_t, 4> const & indices) {
#if defined(__AVX512VL__)
            _mm256_i64scatter_pd(baseAddr, indices.mVec, mVec, 8);
#else
            __m512i t0 = _mm512_castsi256_si512(indices.mVec);
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            _mm512_mask_i64scatter_pd(baseAddr, 0xF, t0, t1, 8);
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE double* scatter(SIMDVecMask<4> const & mask, double* baseAddr, SIMDVec_u<uint64_t, 4> const & indices) {
#if defined(__AVX512VL__)
            _mm256_mask_i64scatter_pd(baseAddr, mask.mMask, indices.mVec, mVec, 8);
#else
            __m512i t0 = _mm512_castsi256_si512(indices.mVec);
            __m512d t1 = _mm512_castpd256_pd512(mVec);
            _mm512_mask_i64scatter_pd(baseAddr, mask.mMask & 0xF, t0, t1, 8);
#endif
            return baseAddr;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __m256d t0 = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_f & nega() {
            mVec = _mm256_sub_pd(_mm256_set1_pd(0.0), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_f abs() const {
#if defined (__GNUG__)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512i t1 = _mm512_castpd_si512(t0);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t1, t2);
            __m512d t4 = _mm512_castsi512_pd(t3);
            __m256d t5 = _mm512_castpd512_pd256(t4);
            return SIMDVec_f(t5);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_abs_pd(t0);
            __m256d t2 = _mm512_castpd512_pd256(t1);
            return SIMDVec_f(t2);
#endif
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512i t1 = _mm512_castpd_si512(t0);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t1, t2);
            __m512d t4 = _mm512_castsi512_pd(t3);
            __m512d t5 = _mm512_mask_mov_pd(t0, mask.mMask, t4);
            __m256d t6 = _mm512_castpd512_pd256(t5);
            return SIMDVec_f(t6);
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_abs_pd(t0, mask.mMask, t0);
            __m256d t3 = _mm512_castpd512_pd256(t2);
            return SIMDVec_f(t3);
#endif
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
#if defined (__GNUG__)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512i t1 = _mm512_castpd_si512(t0);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t1, t2);
            __m512d t4 = _mm512_castsi512_pd(t3);
            mVec = _mm512_castpd512_pd256(t4);
            return *this;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_abs_pd(t0);
            mVec = _mm512_castpd512_pd256(t1);
            return *this;
#endif
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
#if defined (__GNUG__)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512i t1 = _mm512_castpd_si512(t0);
            __m512i t2 = _mm512_set1_epi64(0x7FFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_and_epi64(t1, t2);
            __m512d t4 = _mm512_castsi512_pd(t3);
            __m512d t5 = _mm512_mask_mov_pd(t0, mask.mMask, t4);
            mVec = _mm512_castpd512_pd256(t5);
            return *this;
#else
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t2 = _mm512_mask_abs_pd(t0, mask.mMask, t0);
            mVec = _mm512_castpd512_pd256(t2);
            return *this;
#endif
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __m256d t0 = _mm256_mul_pd(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = _mm256_mul_pd(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<4> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __m256d t0 = _mm256_sqrt_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = _mm256_sqrt_pd(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_f round() const {
            __m256d t0 = _mm256_round_pd(mVec, _MM_FROUND_TO_NEAREST_INT);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<4> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_i<int64_t, 4> trunc() const {
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
        UME_FORCE_INLINE SIMDVec_i<int64_t, 4> trunc(SIMDVecMask<4> const & mask) const {
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
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __m256d t0 = _mm256_floor_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __m256d t0 = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_floor_pd);
            return SIMDVec_f(t0);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __m256d t0 = _mm256_ceil_pd(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<4> const & mask) const {
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
        #if defined(UME_USE_SVML)
            __m256d t0 = _mm256_exp_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 4>>(*this);
        #endif
        }
        // MEXP
        UME_FORCE_INLINE SIMDVec_f exp(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_mask_exp_pd(t0, mask.mMask, t0);
            __m256d t2 = _mm512_castpd512_pd256(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::expd<SIMDVec_f, SIMDVec_u<uint64_t, 4>, SIMDVecMask<4>> (mask, *this);
        #endif
        }
        // LOG
        UME_FORCE_INLINE SIMDVec_f log() const {
        #if defined(UME_USE_SVML)
            __m256d t0 = _mm256_log_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 4>>(*this);
        #endif
        }
        // MLOG
        UME_FORCE_INLINE SIMDVec_f log(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_mask_log_pd(t0, mask.mMask, t0);
            __m256d t2 = _mm512_castpd512_pd256(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::logd<SIMDVec_f, SIMDVec_u<uint64_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // LOG2
        // MLOG2
        // LOG10
        // MLOG10
        // SIN
        UME_FORCE_INLINE SIMDVec_f sin() const {
        #if defined(UME_USE_SVML)
            __m256d t0 = _mm256_sin_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(*this);
        #endif
        }
        // MSIN
        UME_FORCE_INLINE SIMDVec_f sin(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_mask_sin_pd(t0, mask.mMask, t0);
            __m256d t2 = _mm512_castpd512_pd256(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::sind<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // COS
        UME_FORCE_INLINE SIMDVec_f cos() const {
        #if defined(UME_USE_SVML)
            __m256d t0 = _mm256_cos_pd(mVec);
            return SIMDVec_f(t0);
        #else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(*this);
        #endif
        }
        // MCOS
        UME_FORCE_INLINE SIMDVec_f cos(SIMDVecMask<4> const & mask) const {
        #if defined(UME_USE_SVML)
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_mask_sin_pd(t0, mask.mMask, t0);
            __m256d t2 = _mm512_castpd512_pd256(t1);
            return SIMDVec_f(t2);
        #else
            return VECTOR_EMULATION::cosd<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(mask, *this);
        #endif
        }
        // SINCOS
        UME_FORCE_INLINE void sincos(SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(32) double raw_cos[4];
            sinvec.mVec = _mm256_sincos_pd((__m256d*)raw_cos, mVec);
            cosvec.mVec = _mm256_load_pd(raw_cos);
        #else
            VECTOR_EMULATION::sincosd<SIMDVec_f, SIMDVec_i<int64_t, 4>, SIMDVecMask<4>>(*this, sinvec, cosvec);
        #endif
        }
        // MSINCOS
        UME_FORCE_INLINE void sincos(SIMDVecMask<4> const & mask, SIMDVec_f & sinvec, SIMDVec_f & cosvec) const {
        #if defined(UME_USE_SVML)
            alignas(64) double raw_cos[8]; // 64B aligned data for 512b vector operation is needed.
            __m512d t0 = _mm512_castpd256_pd512(mVec);
            __m512d t1 = _mm512_mask_sincos_pd((__m512d*)raw_cos, t0, t0, mask.mMask, t0);
            sinvec.mVec = _mm512_castpd512_pd256(t1);
            cosvec.mVec = _mm256_load_pd(raw_cos);
        #else
            sinvec = SCALAR_EMULATION::MATH::sin<SIMDVec_f, SIMDVecMask<4>>(mask, *this);
            cosvec = SCALAR_EMULATION::MATH::cos<SIMDVec_f, SIMDVecMask<4>>(mask, *this);
        #endif
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
        UME_FORCE_INLINE operator SIMDVec_f<float, 4>() const;

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 4>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 4>() const;
    };

}
}

#undef EXPAND_CALL_UNARY
#undef EXPAND_CALL_UNARY_MASK
#undef EXPAND_CALL_BINARY
#undef EXPAND_CALL_BINARY_MASK
#undef EXPAND_CALL_BINARY_SCALAR_MASK

#endif
