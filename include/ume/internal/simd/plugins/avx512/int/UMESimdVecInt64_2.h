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

#ifndef UME_SIMD_VEC_INT64_2_H_
#define UME_SIMD_VEC_INT64_2_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define SET1_EPI64(x) _mm_unpacklo_epi64(_mm_cvtsi64_si128(x), _mm_cvtsi64_si128(x))

#define EXPAND_CALL_UNARY(a_128i, unary_op) \
            _mm512_castsi512_si128( \
                unary_op( \
                    _mm512_castsi128_si512(a_128i)))

#define EXPAND_CALL_UNARY_MASK(a_128i, mask8, unary_op) \
            _mm512_castsi512_si128( \
                unary_op( \
                    _mm512_castsi128_si512(a_128i), \
                    mask8, \
                    _mm512_castsi128_si512(a_128i)))

#define EXPAND_CALL_BINARY(a_128i, b_128i, binary_op) \
            _mm512_castsi512_si128( \
                binary_op( \
                    _mm512_castsi128_si512(a_128i), \
                    _mm512_castsi128_si512(b_128i)))

#define EXPAND_CALL_BINARY_MASK(a_128i, b_128i, mask8, binary_op) \
            _mm512_castsi512_si128( \
                binary_op( \
                    _mm512_castsi128_si512(a_128i), \
                    mask8, \
                    _mm512_castsi128_si512(a_128i), \
                    _mm512_castsi128_si512(b_128i)))

#define EXPAND_CALL_BINARY_SCALAR_MASK(a_128i, b_64u, mask8, binary_op) \
            _mm512_castsi512_si128( \
                binary_op( \
                    _mm512_castsi128_si512(a_128i), \
                    mask8, \
                    _mm512_castsi128_si512(a_128i), \
                    _mm512_set1_epi64(b_64u)))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 2> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 2>,
            SIMDVec_u<uint64_t, 2>,
            int64_t,
            2,
            uint64_t,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 2>,
            SIMDVec_i<int64_t, 1 >>
    {
        friend class SIMDVec_u<uint64_t, 2>;
        friend class SIMDVec_f<double, 2>;

        friend class SIMDVec_i<int64_t, 4>;
    private:
        __m128i mVec;

        UME_FORCE_INLINE explicit SIMDVec_i(__m128i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_i(const __m128i & x) { mVec = x; }

    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i) {
            mVec = SET1_EPI64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int64_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int64_t const *p) {
            __m128i t0 = _mm_cvtsi64_si128(p[0]);
            __m128i t1 = _mm_cvtsi64_si128(p[1]);
            mVec = _mm_unpacklo_epi64(t0, t1);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i0, int64_t i1) {
            __m128i t0 = _mm_cvtsi64_si128(i0);
            __m128i t1 = _mm_cvtsi64_si128(i1);
            mVec = _mm_unpacklo_epi64(t0, t1);
        }

        // EXTRACT
        UME_FORCE_INLINE int64_t extract(uint32_t index) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*) raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int64_t value) {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*) raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*) raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_epi64(mVec, mask.mMask, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi64(t0, mask.mMask, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int64_t b) {
            mVec = SET1_EPI64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<2> const & mask, int64_t b) {
            __m128i t0 = _mm_cvtsi64_si128(b);
            __m128i t1 = _mm_unpacklo_epi64(t0, t0);
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_epi64(mVec, mask.mMask, t1);
#else
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_castsi128_si512(t1);
            __m512i t4 = _mm512_mask_mov_epi64(t2, mask.mMask, t3);
            mVec = _mm512_castsi512_si128(t4);
#endif
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int64_t const *p) {
            mVec = _mm_loadu_si128((const __m128i *) p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<2> const & mask, int64_t const *p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_epi64(mVec, mask.mMask, p);
#else
            __m128i t0 = _mm_loadu_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int64_t const *p) {
            mVec = _mm_load_si128((const __m128i *) p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<2> const & mask, int64_t const *p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_load_epi64(mVec, mask.mMask, p);
#else 
            __m128i t0 = _mm_load_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int64_t* store(int64_t* p) const {
            _mm_storeu_si128((__m128i *)p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int64_t* store(SIMDVecMask<2> const & mask, int64_t* p) const {
#if defined(__AVX512VL__)
            _mm_mask_storeu_epi64(p, mask.mMask, mVec);
#else
            __m128i t0 = _mm_loadu_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            __m128i t4 = _mm512_castsi512_si128(t3);
            _mm_storeu_si128((__m128i*)p, t4);
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int64_t* storea(int64_t* p) const {
            _mm_store_si128((__m128i *)p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int64_t* storea(SIMDVecMask<2> const & mask, int64_t* p) const {
#if defined(__AVX512VL__)
            _mm_mask_store_epi64(p, mask.mMask, mVec);
#else
            __m128i t0 = _mm_load_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            __m128i t4 = _mm512_castsi512_si128(t3);
            _mm_store_si128((__m128i*)p, t4);
#endif
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mov_epi64(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi64(t0, mask.mMask, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_i(t3);
#endif
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mov_epi64(mVec, mask.mMask, SET1_EPI64(b));
            return SIMDVec_i(t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(b);
            __m512i t2 = _mm512_mask_mov_epi64(t0, mask.mMask, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_i(t3);
#endif
        }
        // SWIZZLE
        // SWIZZLEA

        // SORTA
        // SORTD

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __m128i t0 = _mm_add_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_add_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int64_t b) const {
            __m128i t0 = _mm_add_epi64(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm_add_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_epi64);
#endif
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int64_t b) {
            mVec = _mm_add_epi64(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_epi64);
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
        UME_FORCE_INLINE SIMDVec_i postinc() {
            __m128i t0 = SET1_EPI64(1);
            __m128i t1 = mVec;
            mVec = _mm_add_epi64(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<2> const & mask) {
            __m128i t0 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            __m128i t0 = SET1_EPI64(1);
            mVec = _mm_add_epi64(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<2> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_epi64);
#endif
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __m128i t0 = _mm_sub_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int64_t b) const {
            __m128i t0 = _mm_sub_epi64(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm_sub_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int64_t b) {
            mVec = _mm_sub_epi64(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_epi64);
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
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m128i t0 = _mm_sub_epi64(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi64(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(b.mVec, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int64_t b) const {
            __m128i t0 = _mm_sub_epi64(SET1_EPI64(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<2> const & mask, int64_t b) const {
            __m128i t0 = SET1_EPI64(b);
#if defined(__AVX512VL__)
            __m128i t1 = _mm_mask_sub_epi64(t0, mask.mMask, t0, mVec);
#else
            __m128i t1 = EXPAND_CALL_BINARY_MASK(t0, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm_sub_epi64(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(b.mVec, mask.mMask, b.mVec, mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(b.mVec, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int64_t b) {
            mVec = _mm_sub_epi64(SET1_EPI64(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<2> const & mask, int64_t b) {
            __m128i t0 = SET1_EPI64(b);
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(t0, mask.mMask, t0, mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(t0, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            __m128i t0 = SET1_EPI64(1);
            __m128i t1 = mVec;
            mVec = _mm_sub_epi64(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<2> const & mask) {
            __m128i t0 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            mVec = _mm_sub_epi64(mVec, SET1_EPI64(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<2> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
#if defined(__AVX512DQ__)
    #if defined(__AVX512VL__)
            __m128i t0 = _mm_mullo_epi64(mVec, b.mVec);
    #else
            __m128i t0 = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_mullo_epi64);
    #endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = _mm_extract_epi64(b.mVec, 0);
            int64_t t4 = _mm_extract_epi64(b.mVec, 1);
            int64_t t5 = t1 * t3;
            int64_t t6 = t2 * t4;
            __m128i t0 = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t5), _mm_cvtsi64_si128(t6));
#endif
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mullo_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_mask(mVec, b.mVec, mask.mMask, _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = _mm_extract_epi64(b.mVec, 0);
            int64_t t4 = _mm_extract_epi64(b.mVec, 1);
            int64_t t5 = ((mask.mMask & 0x1) != 0) ? t1 * t3 : t1;
            int64_t t6 = ((mask.mMask & 0x2) != 0) ? t2 * t4 : t2;
            __m128i t0 = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t5), _mm_cvtsi64_si128(t6));
#endif
            return SIMDVec_i(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mullo_epi64(mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = t1 * b;
            int64_t t4 = t2 * b;
            __m128i t0 = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t3), _mm_cvtsi64_si128(t4));
#endif
                return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mullo_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_mask(mVec, SET1_EPI64(b), mask.mMask, _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = ((mask.mMask & 0x1) != 0) ? t1 * b : t1;
            int64_t t4 = ((mask.mMask & 0x2) != 0) ? t2 * b : t2;
            __m128i t0 = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t3), _mm_cvtsi64_si128(t4));
#endif
                return SIMDVec_i(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm_mullo_epi64(mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = _mm_extract_epi64(b.mVec, 0);
            int64_t t4 = _mm_extract_epi64(b.mVec, 1);
            int64_t t5 = t1 * t3;
            int64_t t6 = t2 * t4;
            mVec = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t5), _mm_cvtsi64_si128(t6));
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm_mask_mullo_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_mask(mVec, b.mVec, mask.mMask, _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = _mm_extract_epi64(b.mVec, 0);
            int64_t t4 = _mm_extract_epi64(b.mVec, 1);
            int64_t t5 = ((mask.mMask & 0x1) != 0) ? t1 * t3 : t1;
            int64_t t6 = ((mask.mMask & 0x2) != 0) ? t2 * t4 : t2;
            mVec = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t5), _mm_cvtsi64_si128(t6));
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm_mullo_epi64(mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = t1 * b;
            int64_t t4 = t2 * b;
            mVec = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t3), _mm_cvtsi64_si128(t4));
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int64_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm_mask_mullo_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_mask(mVec, SET1_EPI64(b), mask.mMask, _mm512_mullo_epi64);
#endif
#else
            int64_t t1 = _mm_extract_epi64(mVec, 0);
            int64_t t2 = _mm_extract_epi64(mVec, 1);
            int64_t t3 = ((mask.mMask & 0x1) != 0) ? t1 * b : t1;
            int64_t t4 = ((mask.mMask & 0x2) != 0) ? t2 * b : t2;
            mVec = _mm_unpacklo_epi64(_mm_cvtsi64_si128(t3), _mm_cvtsi64_si128(t4));
#endif
            return *this;
        }
        // DIVV
        /*UME_FORCE_INLINE SIMDVec_i div(SIMDVec_i const & b) const {
            int64_t t0 = mVec[0] / b.mVec[0];
            int64_t t1 = mVec[1] / b.mVec[1];
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }*/
        // MDIVV
        /*UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            int64_t t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_i(t0, t1);
        }*/
        // DIVS
        /*UME_FORCE_INLINE SIMDVec_i div(int64_t b) const {
            int64_t t0 = mVec[0] / b;
            int64_t t1 = mVec[1] / b;
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (int64_t b) const {
            return div(b);
        }*/
        // MDIVS
        /*UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<2> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            int64_t t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            return SIMDVec_i(t0, t1);
        }*/
        // DIVVA
        /*UME_FORCE_INLINE SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }*/
        // MDIVVA
        /*UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return *this;
        }*/
        // DIVSA
        /*UME_FORCE_INLINE SIMDVec_i & diva(int64_t b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (int64_t b) {
            return diva(b);
        }*/
        // MDIVSA
        /*UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<2> const & mask, int64_t b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b : mVec[1];
            return *this;
        }*/
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpeq_epu64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_castsi128_si512(b.mVec));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (int64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpeq_epu64_mask(
                            _mm512_castsi128_si512(mVec), 
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (int64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (SIMDVec_i const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpneq_epi64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpneq_epi64_mask(
                            _mm512_castsi128_si512(mVec), 
                            _mm512_castsi128_si512(b.mVec));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (int64_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpneq_epi64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpneq_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (int64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpgt_epi64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpgt_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_castsi128_si512(b.mVec));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (int64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpgt_epi64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpgt_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (int64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (SIMDVec_i const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmplt_epi64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmplt_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_castsi128_si512(b.mVec));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (int64_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmplt_epi64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmplt_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (int64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (SIMDVec_i const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpge_epi64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpge_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_castsi128_si512(b.mVec));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (int64_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpge_epi64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpge_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (int64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple (SIMDVec_i const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmple_epi64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmple_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_castsi128_si512(b.mVec));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple (int64_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmple_epi64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmple_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<2> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (int64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epi64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_castsi128_si512(b.mVec));
#endif
            return m0 == 0x03;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epi64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(
                            _mm512_castsi128_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            return m0 == 0x03;
        }
        // UNIQUE
        // HADD
        UME_FORCE_INLINE int64_t hadd() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] + raw[1];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_reduce_add_epi64(t0);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE int64_t hadd(SIMDVecMask<2> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            int64_t t0 = 0;
            _mm_store_si128((__m128i*)raw, mVec);
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, t0);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE int64_t hadd(int64_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return b + raw[0] + raw[1];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_reduce_add_epi64(t0);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE int64_t hadd(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            int64_t t0 = b;
            _mm_store_si128((__m128i*)raw, mVec);
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, t0);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE int64_t hmul() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] * raw[1];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_mask_reduce_mul_epi64(0x3, t0);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE int64_t hmul(SIMDVecMask<2> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            int64_t t0 = 1;
            _mm_store_si128((__m128i*)raw, mVec);
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask, t0);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE int64_t hmul(int64_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return b * raw[0] * raw[1];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_mask_reduce_mul_epi64(0x3, t0);
            return retval * b;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE int64_t hmul(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(16) int64_t raw[2];
            int64_t t0 = b;
            _mm_store_si128((__m128i*)raw, mVec);
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            int64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask, t0);
            return retval * b;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(b)).add(c);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(mask, b)).add(mask, c);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(b)).sub(c);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(mask, b)).sub(mask, c);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (add(b)).mul(c);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (add(mask, b)).mul(mask, c);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (sub(b)).mul(c);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (sub(mask, b)).mul(mask, c);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_max_epi64(mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_max_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_max_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_max_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_max_epi64(mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_max_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_max_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_max_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_max_epi64(mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_max_epi64);
#endif
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_max_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_max_epi64);
#endif
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_max_epi64(mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_max_epi64);
#endif
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_max_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_max_epi64);
#endif
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_min_epi64(mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_min_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_min_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_min_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_min_epi64(mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_min_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_min_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_min_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_min_epi64(mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_min_epi64);
#endif
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_min_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_min_epi64);
#endif
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_min_epi64(mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_min_epi64);
#endif
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_min_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_min_epi64);
#endif
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE int64_t hmax() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] > raw[1] ? raw[0] : raw[1];
        }
        // MHMAX
        UME_FORCE_INLINE int64_t hmax(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] : std::numeric_limits<int64_t>::min();
            int64_t t1 = ((mask.mMask & 0x2) && raw[1] > t0) ? raw[1] : t0;
            return t1;
        }
        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] > raw[1] ? 0 : 1;
        }
        // MIMAX
        UME_FORCE_INLINE uint32_t imax(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t i0 = 0xFFFFFFFF;
            int64_t t0 = std::numeric_limits<int64_t>::min();
            if ((mask.mMask & 0x1) != 0) {
                i0 = 0;
                t0 = raw[0];
            }
            if (((mask.mMask & 0x2) != 0) && raw[1] > t0) {
                i0 = 1;
            }
            return i0;
        }
        // HMIN
        UME_FORCE_INLINE int64_t hmin() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] < raw[1] ? raw[0] : raw[1];
        }
        // MHMIN
        UME_FORCE_INLINE int64_t hmin(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] : std::numeric_limits<int64_t>::max();
            int64_t t1 = ((mask.mMask & 0x2) && raw[1] < t0) ? raw[1] : t0;
            return t1;
        }
        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] < raw[1] ? 0 : 1;
        }
        // MIMIN
        UME_FORCE_INLINE uint32_t imin(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t i0 = 0xFFFFFFFF;
            int64_t t0 = std::numeric_limits<int64_t>::max();
            if ((mask.mMask & 0x1) != 0) {
                i0 = 0;
                t0 = raw[0];
            }
            if (((mask.mMask & 0x2) != 0) && raw[1] < t0) {
                i0 = 1;
            }
            return i0;
        }

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            __m128i t0 = _mm_and_si128(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_and_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_and_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int64_t b) const {
            __m128i t0 = _mm_and_si128(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_and_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_and_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm_and_si128(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_and_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_and_epi64);
#endif
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int64_t b) {
            mVec = _mm_and_si128(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_and_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_and_epi64);
#endif
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            __m128i t0 = _mm_or_si128(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_or_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_or_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int64_t b) const {
            __m128i t0 = _mm_or_si128(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_or_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_or_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm_or_si128(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_or_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_or_epi64);
#endif
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int64_t b) {
            mVec = _mm_or_si128(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_or_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_or_epi64);
#endif
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m128i t0 = _mm_xor_si128(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_xor_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int64_t b) const {
            __m128i t0 = _mm_xor_si128(mVec, SET1_EPI64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<2> const & mask, int64_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_xor_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m128i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm_xor_si128(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_xor_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int64_t b) {
            mVec = _mm_xor_si128(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<2> const & mask, int64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_xor_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            __m128i t0 = _mm_xor_si128(mVec, _mm_set1_epi32(0xFFFFFFFF));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<2> const & mask) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_mask_xor_epi64(mVec, mask.mMask, mVec, t0);
#else
            __m128i t1 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 0xFFFFFFFFFFFFFFFF, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return SIMDVec_i(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            mVec = _mm_xor_si128(mVec, _mm_set1_epi32(0xFFFFFFFF));
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<2> const & mask) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            mVec = _mm_mask_xor_epi64(mVec, mask.mMask, mVec, t0);
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 0xFFFFFFFFFFFFFFFF, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE int64_t hband() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] & raw[1];
        }
        // MHBAND
        UME_FORCE_INLINE int64_t hband(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] : 0xFFFFFFFFFFFFFFFF;
            int64_t t1 = (mask.mMask & 0x2) ? raw[1] & t0 : t0;
            return t1;
        }
        // HBANDS
        UME_FORCE_INLINE int64_t hband(int64_t b) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] & raw[1] & b;
        }
        // MHBANDS
        UME_FORCE_INLINE int64_t hband(SIMDVecMask<2> const & mask, int64_t b) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] & b : b;
            int64_t t1 = (mask.mMask & 0x2) ? raw[1] & t0 : t0;
            return t1;
        }
        // HBOR
        UME_FORCE_INLINE int64_t hbor() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] | raw[1];
        }
        // MHBOR
        UME_FORCE_INLINE int64_t hbor(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] : 0;
            int64_t t1 = (mask.mMask & 0x2) ? raw[1] | t0 : t0;
            return t1;
        }
        // HBORS
        UME_FORCE_INLINE int64_t hbor(int64_t b) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] | raw[1] | b;
        }
        // MHBORS
        UME_FORCE_INLINE int64_t hbor(SIMDVecMask<2> const & mask, int64_t b) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] | b : b;
            int64_t t1 = (mask.mMask & 0x2) ? raw[1] | t0 : t0;
            return t1;
        }
        // HBXOR
        UME_FORCE_INLINE int64_t hbxor() const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1];
        }
        // MHBXOR
        UME_FORCE_INLINE int64_t hbxor(SIMDVecMask<2> const & mask) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] : 0;
            int64_t t1 = (mask.mMask & 0x2) ? raw[1] ^ t0 : t0;
            return t1;
        }
        // HBXORS
        UME_FORCE_INLINE int64_t hbxor(int64_t b) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1] ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE int64_t hbxor(SIMDVecMask<2> const & mask, int64_t b) const {
            alignas(16) int64_t raw[2];
            _mm_store_si128((__m128i*)raw, mVec);
            int64_t t0 = (mask.mMask & 0x1) ? raw[0] ^ b : b;
            int64_t t1 = (mask.mMask & 0x2) ? raw[1] ^ t0 : t0;
            return t1;
        }

        // GATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(int64_t const * baseAddr, uint64_t stride) {
            __m128i t0 = _mm_set_epi64x(stride, 0);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
            mVec = _mm_i64gather_epi64((const long long int*)baseAddr, t0, 8);
#else
            mVec = _mm_i64gather_epi64((int64_t const*)baseAddr, t0, 8);
#endif
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(SIMDVecMask<2> const & mask, int64_t const * baseAddr, uint64_t stride) {
            __m128i t0 = _mm_set_epi64x(stride, 0);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
            __m128i t1 = _mm_i64gather_epi64((const long long int*)baseAddr, t0, 8);
#else
            __m128i t1 = _mm_i64gather_epi64((int64_t const*)baseAddr, t0, 8);
#endif
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_epi64(mVec, mask.mMask, t1);
#else
            mVec = _mm512_castsi512_si128(
                    _mm512_mask_mov_epi64(
                        _mm512_castsi128_si512(mVec),
                        mask.mMask,
                        _mm512_castsi128_si512(t1)));
#endif
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int64_t const * baseAddr, uint64_t const * indices) {
            __m128i t0 =_mm_loadu_si128((__m128i *)indices);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
            mVec = _mm_i64gather_epi64((const long long int*)baseAddr, t0, 8);
#else
            mVec = _mm_i64gather_epi64((int64_t const*)baseAddr, t0, 8);
#endif
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<2> const & mask, int64_t const * baseAddr, uint64_t const * indices) {
            __m128i t0 = _mm_loadu_si128((__m128i *)indices);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
            __m128i t1 = _mm_i64gather_epi64((const long long int*)baseAddr, t0, 8);
#else
            __m128i t1 = _mm_i64gather_epi64((int64_t const*)baseAddr, t0, 8);
#endif
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_epi64(mVec, mask.mMask, t1);
#else
            mVec = _mm512_castsi512_si128(
                    _mm512_mask_mov_epi64(
                        _mm512_castsi128_si512(mVec),
                        mask.mMask,
                        _mm512_castsi128_si512(t1)));
#endif
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(int64_t const * baseAddr, SIMDVec_u<uint64_t, 2> const & indices) {
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
            mVec = _mm_i64gather_epi64((const long long int*)baseAddr, indices.mVec, 8);
#else
            mVec = _mm_i64gather_epi64((int64_t const*)baseAddr, indices.mVec, 8);
#endif
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<2> const & mask, int64_t const * baseAddr, SIMDVec_u<uint64_t, 2> const & indices) {
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
            __m128i t0 = _mm_i64gather_epi64((const long long int*)baseAddr, indices.mVec, 8);
#else
            __m128i t0 = _mm_i64gather_epi64((int64_t const*)baseAddr, indices.mVec, 8);
#endif
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_epi64(mVec, mask.mMask, t0);
#else
            mVec = _mm512_castsi512_si128(
                _mm512_mask_mov_epi64(
                    _mm512_castsi128_si512(mVec),
                    mask.mMask,
                    _mm512_castsi128_si512(t0)));
#endif
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE int64_t* scatteru(int64_t* baseAddr, uint64_t stride) const {
            __m128i t0 = _mm_set_epi64x(stride, 0);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
  #if defined(__AVX512VL__)
            _mm_i64scatter_epi64((long long int*)baseAddr, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                            (long long int*)baseAddr,
                            0x3,
                            _mm512_castsi128_si512(t0),
                            _mm512_castsi128_si512(mVec),
                            8);
  #endif
#else
  #if defined(__AVX512VL__)
            _mm_i64scatter_epi64(baseAddr, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                            baseAddr,
                            0x3,
                            _mm512_castsi128_si512(t0),
                            _mm512_castsi128_si512(mVec),
                            8);
  #endif
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE int64_t*  scatteru(SIMDVecMask<2> const & mask, int64_t* baseAddr, uint64_t stride) const {
            __m128i t0 = _mm_set_epi64x(stride, 0);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
  #if defined(__AVX512VL__)
            _mm_mask_i64scatter_epi64((long long int*)baseAddr, mask.mMask, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                (long long int*)baseAddr,
                mask.mMask,
                _mm512_castsi128_si512(t0),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#else
  #if defined(__AVX512VL__)
            _mm_mask_i64scatter_epi64(baseAddr, mask.mMask, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                mask.mMask,
                _mm512_castsi128_si512(t0),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE int64_t* scatter(int64_t* baseAddr, uint64_t* indices) const {
            __m128i t0 = _mm_loadu_si128((__m128i *)indices);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
  #if defined(__AVX512VL__)
            _mm_i64scatter_epi64((long long int*)baseAddr, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                            (long long int*)baseAddr,
                            0x3,
                            _mm512_castsi128_si512(t0),
                            _mm512_castsi128_si512(mVec),
                            8);
  #endif
#else
  #if defined(__AVX512VL__)
            _mm_i64scatter_epi64(baseAddr, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                            baseAddr,
                            0x3,
                            _mm512_castsi128_si512(t0),
                            _mm512_castsi128_si512(mVec),
                            8);
  #endif
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE int64_t* scatter(SIMDVecMask<2> const & mask, int64_t* baseAddr, uint64_t* indices) const {
            __m128i t0 = _mm_loadu_si128((__m128i *)indices);
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
  #if defined(__AVX512VL__)
            _mm_mask_i64scatter_epi64((long long int*)baseAddr, mask.mMask, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                (long long int*)baseAddr,
                mask.mMask,
                _mm512_castsi128_si512(t0),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#else
  #if defined(__AVX512VL__)
            _mm_mask_i64scatter_epi64(baseAddr, mask.mMask, t0, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                mask.mMask,
                _mm512_castsi128_si512(t0),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int64_t* scatter(int64_t* baseAddr, SIMDVec_u<uint64_t, 2> const & indices) const {
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
  #if defined(__AVX512VL__)
            _mm_i64scatter_epi64((long long int*)baseAddr, indices.mVec, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                (long long int*)baseAddr,
                0x3,
                _mm512_castsi128_si512(indices.mVec),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#else
  #if defined(__AVX512VL__)
            _mm_i64scatter_epi64(baseAddr, indices.mVec, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                0x3,
                _mm512_castsi128_si512(indices.mVec),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int64_t* scatter(SIMDVecMask<2> const & mask, int64_t* baseAddr, SIMDVec_u<uint64_t, 2> const & indices) const {
#if defined(WA_GCC_INTR_SUPPORT_6_2)
            // g++ has some interface issues.
  #if defined(__AVX512VL__)
            _mm_mask_i64scatter_epi64((long long int*)baseAddr, mask.mMask, indices.mVec, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                (long long int*)baseAddr,
                mask.mMask,
                _mm512_castsi128_si512(indices.mVec),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#else
  #if defined(__AVX512VL__)
            _mm_mask_i64scatter_epi64(baseAddr, mask.mMask, indices.mVec, mVec, 8);
  #else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                mask.mMask,
                _mm512_castsi128_si512(indices.mVec),
                _mm512_castsi128_si512(mVec),
                8);
  #endif
#endif
            return baseAddr;
        }

        //// LSHV
        //UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
        //    int64_t t0 = mVec[0] << b.mVec[0];
        //    int64_t t1 = mVec[1] << b.mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// MLSHV
        //UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// LSHS
        //UME_FORCE_INLINE SIMDVec_u lsh(int64_t b) const {
        //    int64_t t0 = mVec[0] << b;
        //    int64_t t1 = mVec[1] << b;
        //    return SIMDVec_u(t0, t1);
        //}
        //// MLSHS
        //UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<2> const & mask, int64_t b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// LSHVA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
        //    mVec[0] = mVec[0] << b.mVec[0];
        //    mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// MLSHVA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// LSHSA
        //UME_FORCE_INLINE SIMDVec_u & lsha(int64_t b) {
        //    mVec[0] = mVec[0] << b;
        //    mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        //// MLSHSA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<2> const & mask, int64_t b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b;
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        //// RSHV
        //UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
        //    int64_t t0 = mVec[0] >> b.mVec[0];
        //    int64_t t1 = mVec[1] >> b.mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// MRSHV
        //UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// RSHS
        //UME_FORCE_INLINE SIMDVec_u rsh(int64_t b) const {
        //    int64_t t0 = mVec[0] >> b;
        //    int64_t t1 = mVec[1] >> b;
        //    return SIMDVec_u(t0, t1);
        //}
        //// MRSHS
        //UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<2> const & mask, int64_t b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// RSHVA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
        //    mVec[0] = mVec[0] >> b.mVec[0];
        //    mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// MRSHVA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
        //    if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
        //    if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// RSHSA
        //UME_FORCE_INLINE SIMDVec_u & rsha(int64_t b) {
        //    mVec[0] = mVec[0] >> b;
        //    mVec[1] = mVec[1] >> b;
        //    return *this;
        //}
        //// MRSHSA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<2> const & mask, int64_t b) {
        //    if (mask.mMask[0]) mVec[0] = mVec[0] >> b;
        //    if (mask.mMask[1]) mVec[1] = mVec[1] >> b;
        //    return *this;
        //}
        // ROLV
        // MROLV
        // ROLS
        // MROLS
        // ROLVA
        // MROLVA
        // ROLSA
        // MROLSA
        // RORV
        // MRORV
        // RORS
        // MRORS
        // RORVA
        // MRORVA
        // RORSA
        // MRORSA

        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            __m128i t0 = _mm_sub_epi64(_mm_setzero_si128(), mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<2> const & mask) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi64(mVec, mask.mMask, _mm_setzero_si128(), mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_setzero_si512();
            __m512i t3 = _mm512_mask_sub_epi64(t1, mask.mMask, t2, t1);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_i(t0);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec = _mm_sub_epi64(_mm_setzero_si128(), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<2> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi64(mVec, mask.mMask, _mm_setzero_si128(), mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_setzero_si512();
            __m512i t3 = _mm512_mask_sub_epi64(t1, mask.mMask, t2, t1);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_abs_epi64(mVec);
#else
            __m128i t0 = EXPAND_CALL_UNARY(mVec, _mm512_abs_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<2> const & mask) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_abs_epi64(mVec, mask.mMask, mVec);
#else
            __m128i t0 = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_abs_epi64);
#endif
            return SIMDVec_i(t0);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
#if defined(__AVX512VL__)
            mVec = _mm_abs_epi64(mVec);
#else
            mVec = EXPAND_CALL_UNARY(mVec, _mm512_abs_epi64);
#endif
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<2> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_abs_epi64(mVec, mask.mMask, mVec);
#else
            mVec = EXPAND_CALL_UNARY_MASK(mVec, mask.mMask, _mm512_mask_abs_epi64);
#endif
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int64_t, 1> const & a, SIMDVec_i<int64_t, 1> const & b) {
            __m128i t0 = _mm_cvtsi64_si128(a[0]);
            __m128i t1 = _mm_cvtsi64_si128(b[0]);
            mVec = _mm_unpacklo_epi64(t0, t1);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int64_t, 1> const & a) {
            mVec = _mm_insert_epi64(mVec, a[0], 0);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_i & packhi(SIMDVec_i<int64_t, 1> const & b) {
            mVec = _mm_insert_epi64(mVec, b[0], 1);
            return *this;
        }
        // UNPACK
        void unpack(SIMDVec_i<int64_t, 1> & a, SIMDVec_i<int64_t, 1> & b) const {
            a.insert(0, _mm_extract_epi64(mVec, 0));
            b.insert(0, _mm_extract_epi64(mVec, 1));
        }
        // UNPACKLO
        SIMDVec_i<int64_t, 1> unpacklo() const {
            return SIMDVec_i<int64_t, 1> (_mm_extract_epi64(mVec, 0));
        }
        // UNPACKHI
        SIMDVec_i<int64_t, 1> unpackhi() const {
            return SIMDVec_i<int64_t, 1> (_mm_extract_epi64(mVec, 1));
        }

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 2>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 2>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 2>() const;
    };

#undef SET1_EPI64
#undef EXPAND_CALL_UNARY
#undef EXPAND_CALL_UNARY_MASK
#undef EXPAND_CALL_BINARY
#undef EXPAND_CALL_BINARY_MASK
#undef EXPAND_CALL_BINARY_SCALAR_MASK

}
}

#endif
