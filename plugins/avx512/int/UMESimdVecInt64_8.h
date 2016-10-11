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

#ifndef UME_SIMD_VEC_INT64_8_H_
#define UME_SIMD_VEC_INT64_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define EXPAND_CALL_UNARY(a_256i, unary_op) \
            _mm512_castsi512_si256( \
                unary_op( \
                    _mm512_castsi256_si512(a_256i)))

#define EXPAND_CALL_UNARY_MASK(a_256i, mask8, unary_op) \
            _mm512_castsi512_si256( \
                unary_op( \
                    _mm512_castsi256_si512(a_256i), \
                    mask8, \
                    _mm512_castsi256_si512(a_256i)))

#define EXPAND_CALL_BINARY(a_256i, b_256i, binary_op) \
            _mm512_castsi512_si256( \
                binary_op( \
                    _mm512_castsi256_si512(a_256i), \
                    _mm512_castsi256_si512(b_256i)))

#define EXPAND_CALL_BINARY_MASK(a_256i, b_256i, mask8, binary_op) \
            _mm512_castsi512_si256( \
                binary_op( \
                    _mm512_castsi256_si512(a_256i), \
                    mask8, \
                    _mm512_castsi256_si512(a_256i), \
                    _mm512_castsi256_si512(b_256i)))

#define EXPAND_CALL_BINARY_SCALAR_MASK(a_256i, b_64u, mask8, binary_op) \
            _mm512_castsi512_si256( \
                binary_op( \
                    _mm512_castsi256_si512(a_256i), \
                    mask8, \
                    _mm512_castsi256_si512(a_256i), \
                    _mm512_set1_epi64(b_64u)))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 8> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 8>,
            SIMDVec_u<uint64_t, 8>,
            int64_t,
            8,
            uint64_t,
            SIMDVecMask<8>,
            SIMDSwizzle<8>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 8>,
            SIMDVec_i<int64_t, 4>>
    {
    public:
        friend class SIMDVec_u<uint64_t, 8>;
        friend class SIMDVec_f<double, 8>;

        friend class SIMDVec_u<int64_t, 16>;
        friend class SIMDVec_i<int64_t, 16>;

    private:
        __m512i mVec;

        UME_FORCE_INLINE explicit SIMDVec_i(__m512i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_i(const __m512i & x) { mVec = x; }

    public:
        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i) {
            mVec = _mm512_set1_epi64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, int64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int64_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int64_t const *p) {
            mVec = _mm512_loadu_si512(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i0, int64_t i1, int64_t i2, int64_t i3,
                         int64_t i4, int64_t i5, int64_t i6, int64_t i7) {
            mVec = _mm512_set_epi64(i7, i6, i5, i4, i3, i2, i1, i0);
        }

        // EXTRACT
        UME_FORCE_INLINE int64_t extract(uint32_t index) const {
            alignas(64) int64_t raw[8];
            _mm512_store_si512(raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int64_t value) {
            alignas(64) int64_t raw[8];
            _mm512_store_si512(raw, mVec);
            raw[index] = value;
            mVec = _mm512_load_si512(raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_mov_epi64(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int64_t b) {
            mVec = _mm512_set1_epi64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<8> const & mask, int64_t b) {
            __m512i t1 = _mm512_set1_epi64(b);
            mVec = _mm512_mask_mov_epi64(mVec, mask.mMask, t1);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int64_t const *p) {
            mVec = _mm512_loadu_si512((const __m512i *) p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<8> const & mask, int64_t const *p) {
            mVec = _mm512_mask_loadu_epi64(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int64_t const *p) {
            mVec = _mm512_load_si512((const __m512i *) p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<8> const & mask, int64_t const *p) {
            mVec = _mm512_mask_load_epi64(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int64_t* store(int64_t* p) const {
            _mm512_storeu_si512((__m512i *)p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int64_t* store(SIMDVecMask<8> const & mask, int64_t* p) const {
            _mm512_mask_storeu_epi64(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int64_t* storea(int64_t* p) const {
            _mm512_store_si512((__m512i *)p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int64_t* storea(SIMDVecMask<8> const & mask, int64_t* p) const {
            _mm512_mask_store_epi64(p, mask.mMask, mVec);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mov_epi64(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_mov_epi64(mVec, mask.mMask, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_add_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int64_t b) const {
            __m512i t0 = _mm512_add_epi64(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_add_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm512_add_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_add_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int64_t b) {
            mVec = _mm512_add_epi64(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_add_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
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
            __m512i t0 = _mm512_set1_epi64(1);
            __m512i t1 = mVec;
            mVec = _mm512_add_epi64(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<8> const & mask) {
            __m512i t0 = mVec;
            mVec = _mm512_mask_add_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(1));
            return SIMDVec_i(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            __m512i t0 = _mm512_set1_epi64(1);
            mVec = _mm512_add_epi64(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_add_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(1));
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int64_t b) const {
            __m512i t0 = _mm512_sub_epi64(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_sub_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm512_sub_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int64_t b) {
            mVec = _mm512_sub_epi64(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_sub_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
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
            __m512i t0 = _mm512_sub_epi64(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi64(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int64_t b) const {
            __m512i t0 = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_set1_epi64(b);
            __m512i t1 = _mm512_mask_sub_epi64(t0, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm512_sub_epi64(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_sub_epi64(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int64_t b) {
            mVec = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<8> const & mask, int64_t b) {
            __m512i t0 = _mm512_set1_epi64(b);
            mVec = _mm512_mask_sub_epi64(t0, mask.mMask, t0, mVec);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            __m512i t0 = _mm512_set1_epi64(1);
            __m512i t1 = mVec;
            mVec = _mm512_sub_epi64(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<8> const & mask) {
            __m512i t0 = mVec;
            mVec = _mm512_mask_sub_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(1));
            return SIMDVec_i(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            mVec = _mm512_sub_epi64(mVec, _mm512_set1_epi64(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_sub_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(1));
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
#if defined(__AVX512DQ__)
    #if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec, 1);
            __m256i t6 = _mm256_mullo_epi64(t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec, t3, 0);
            t0 = _mm512_inserti64x4(t0, t6, 1);
    #else
            __m512i t0 = _mm512_mullo_epi64(mVec, b.mVec);
    #endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 0);
            int64_t t4 = _mm256_extract_epi64(t1, 1);
            int64_t t5 = _mm256_extract_epi64(t1, 2);
            int64_t t6 = _mm256_extract_epi64(t1, 3);
            int64_t t7 = _mm256_extract_epi64(t2, 0);
            int64_t t8 = _mm256_extract_epi64(t2, 1);
            int64_t t9 = _mm256_extract_epi64(t2, 2);
            int64_t t10 = _mm256_extract_epi64(t2, 3);
            int64_t t11 = t3 * t7;
            int64_t t12 = t4 * t8;
            int64_t t13 = t5 * t9;
            int64_t t14 = t6 * t10;
            __m256i t15 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec, 1);
            int64_t t17 = _mm256_extract_epi64(t15, 0);
            int64_t t18 = _mm256_extract_epi64(t15, 1);
            int64_t t19 = _mm256_extract_epi64(t15, 2);
            int64_t t20 = _mm256_extract_epi64(t15, 3);
            int64_t t21 = _mm256_extract_epi64(t16, 0);
            int64_t t22 = _mm256_extract_epi64(t16, 1);
            int64_t t23 = _mm256_extract_epi64(t16, 2);
            int64_t t24 = _mm256_extract_epi64(t16, 3);
            int64_t t25 = t17 * t21;
            int64_t t26 = t18 * t22;
            int64_t t27 = t19 * t23;
            int64_t t28 = t20 * t24;
            __m512i t0 = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);
#endif
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec, 1);
            __m256i t6 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec, t3, 0);
            t0 = _mm512_inserti64x4(t0, t6, 1);
#else
            __m512i t0 = _mm512_mask_mullo_epi64(mVec, mask.mMask, mVec, b.mVec);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 0);
            int64_t t4 = _mm256_extract_epi64(t1, 1);
            int64_t t5 = _mm256_extract_epi64(t1, 2);
            int64_t t6 = _mm256_extract_epi64(t1, 3);
            int64_t t7 = _mm256_extract_epi64(t2, 0);
            int64_t t8 = _mm256_extract_epi64(t2, 1);
            int64_t t9 = _mm256_extract_epi64(t2, 2);
            int64_t t10 = _mm256_extract_epi64(t2, 3);
            int64_t t11 = ((mask.mMask & 0x1) == 0) ? t3 : t3 * t7;
            int64_t t12 = ((mask.mMask & 0x2) == 0) ? t4 : t4 * t8;
            int64_t t13 = ((mask.mMask & 0x4) == 0) ? t5 : t5 * t9;
            int64_t t14 = ((mask.mMask & 0x8) == 0) ? t6 : t6 * t10;
            __m256i t15 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec, 1);
            int64_t t17 = _mm256_extract_epi64(t15, 0);
            int64_t t18 = _mm256_extract_epi64(t15, 1);
            int64_t t19 = _mm256_extract_epi64(t15, 2);
            int64_t t20 = _mm256_extract_epi64(t15, 3);
            int64_t t21 = _mm256_extract_epi64(t16, 0);
            int64_t t22 = _mm256_extract_epi64(t16, 1);
            int64_t t23 = _mm256_extract_epi64(t16, 2);
            int64_t t24 = _mm256_extract_epi64(t16, 3);
            int64_t t25 = ((mask.mMask & 0x10) == 0) ? t17 : t17 * t21;
            int64_t t26 = ((mask.mMask & 0x20) == 0) ? t18 : t18 * t22;
            int64_t t27 = ((mask.mMask & 0x40) == 0) ? t19 : t19 * t23;
            int64_t t28 = ((mask.mMask & 0x80) == 0) ? t20 : t20 * t24;
            __m512i t0 = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);
#endif
            return SIMDVec_i(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm256_mullo_epi64(t4, t2);
            __m512i t0 = _mm512_inserti64x4(mVec, t3, 0);
            t0 = _mm512_inserti64x4(t0, t5, 1);
#else
            __m512i t0 = _mm512_mullo_epi64(mVec, _mm256_set1_epi64(b));
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            int64_t t2 = _mm256_extract_epi64(t1, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 1);
            int64_t t4 = _mm256_extract_epi64(t1, 2);
            int64_t t5 = _mm256_extract_epi64(t1, 3);
            int64_t t6 = t2 * b;
            int64_t t7 = t3 * b;
            int64_t t8 = t4 * b;
            int64_t t9 = t5 * b;
            __m256i t10 = _mm512_extracti64x4_epi64(mVec, 1);
            int64_t t11 = _mm256_extract_epi64(t10, 0);
            int64_t t12 = _mm256_extract_epi64(t10, 1);
            int64_t t13 = _mm256_extract_epi64(t10, 2);
            int64_t t14 = _mm256_extract_epi64(t10, 3);
            int64_t t15 = t11 * b;
            int64_t t16 = t12 * b;
            int64_t t17 = t13 * b;
            int64_t t18 = t14 * b;
            __m512i t0 = _mm512_set_epi64(t18, t17, t16, t15, t9, t8, t7, t6);
#endif
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<8> const & mask, int64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t2);
            __m512i t0 = _mm512_inserti64x4(mVec, t3, 0);
            t0 = _mm512_inserti64x4(t0, t5, 1);
#else
            __m512i t0 = _mm512_mullo_epi64(mVec, _mm256_set1_epi64(b));
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            int64_t t2 = _mm256_extract_epi64(t1, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 1);
            int64_t t4 = _mm256_extract_epi64(t1, 2);
            int64_t t5 = _mm256_extract_epi64(t1, 3);
            int64_t t6 = ((mask.mMask & 0x1) == 0) ? t2 : t2 * b;
            int64_t t7 = ((mask.mMask & 0x2) == 0) ? t3 : t3 * b;
            int64_t t8 = ((mask.mMask & 0x4) == 0) ? t4 : t4 * b;
            int64_t t9 = ((mask.mMask & 0x8) == 0) ? t5 : t5 * b;
            __m256i t10 = _mm512_extracti64x4_epi64(mVec, 1);
            int64_t t11 = _mm256_extract_epi64(t10, 0);
            int64_t t12 = _mm256_extract_epi64(t10, 1);
            int64_t t13 = _mm256_extract_epi64(t10, 2);
            int64_t t14 = _mm256_extract_epi64(t10, 3);
            int64_t t15 = ((mask.mMask & 0x10) == 0) ? t11 : t11 * b;
            int64_t t16 = ((mask.mMask & 0x20) == 0) ? t12 : t12 * b;
            int64_t t17 = ((mask.mMask & 0x40) == 0) ? t13 : t13 * b;
            int64_t t18 = ((mask.mMask & 0x80) == 0) ? t14 : t14 * b;
            __m512i t0 = _mm512_set_epi64(t18, t17, t16, t15, t9, t8, t7, t6);
#endif
            return SIMDVec_i(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec, 1);
            __m256i t6 = _mm256_mullo_epi64(t4, t5);
            mVec = _mm512_inserti64x4(mVec, t3, 0);
            mVec = _mm512_inserti64x4(mVec, t6, 1);
#else
            mVec = _mm512_mullo_epi64(mVec, b.mVec);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 0);
            int64_t t4 = _mm256_extract_epi64(t1, 1);
            int64_t t5 = _mm256_extract_epi64(t1, 2);
            int64_t t6 = _mm256_extract_epi64(t1, 3);
            int64_t t7 = _mm256_extract_epi64(t2, 0);
            int64_t t8 = _mm256_extract_epi64(t2, 1);
            int64_t t9 = _mm256_extract_epi64(t2, 2);
            int64_t t10 = _mm256_extract_epi64(t2, 3);
            int64_t t11 = t3 * t7;
            int64_t t12 = t4 * t8;
            int64_t t13 = t5 * t9;
            int64_t t14 = t6 * t10;
            __m256i t15 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec, 1);
            int64_t t17 = _mm256_extract_epi64(t15, 0);
            int64_t t18 = _mm256_extract_epi64(t15, 1);
            int64_t t19 = _mm256_extract_epi64(t15, 2);
            int64_t t20 = _mm256_extract_epi64(t15, 3);
            int64_t t21 = _mm256_extract_epi64(t16, 0);
            int64_t t22 = _mm256_extract_epi64(t16, 1);
            int64_t t23 = _mm256_extract_epi64(t16, 2);
            int64_t t24 = _mm256_extract_epi64(t16, 3);
            int64_t t25 = t17 * t21;
            int64_t t26 = t18 * t22;
            int64_t t27 = t19 * t23;
            int64_t t28 = t20 * t24;
            mVec = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec, 1);
            __m256i t6 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t5);
            mVec = _mm512_inserti64x4(mVec, t3, 0);
            mVec = _mm512_inserti64x4(mVec, t6, 1);
#else
            mVec = _mm512_mask_mullo_epi64(mVec, mask.mMask, mVec, b.mVec);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 0);
            int64_t t4 = _mm256_extract_epi64(t1, 1);
            int64_t t5 = _mm256_extract_epi64(t1, 2);
            int64_t t6 = _mm256_extract_epi64(t1, 3);
            int64_t t7 = _mm256_extract_epi64(t2, 0);
            int64_t t8 = _mm256_extract_epi64(t2, 1);
            int64_t t9 = _mm256_extract_epi64(t2, 2);
            int64_t t10 = _mm256_extract_epi64(t2, 3);
            int64_t t11 = ((mask.mMask & 0x1) == 0) ? t3 : t3 * t7;
            int64_t t12 = ((mask.mMask & 0x2) == 0) ? t4 : t4 * t8;
            int64_t t13 = ((mask.mMask & 0x4) == 0) ? t5 : t5 * t9;
            int64_t t14 = ((mask.mMask & 0x8) == 0) ? t6 : t6 * t10;
            __m256i t15 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec, 1);
            int64_t t17 = _mm256_extract_epi64(t15, 0);
            int64_t t18 = _mm256_extract_epi64(t15, 1);
            int64_t t19 = _mm256_extract_epi64(t15, 2);
            int64_t t20 = _mm256_extract_epi64(t15, 3);
            int64_t t21 = _mm256_extract_epi64(t16, 0);
            int64_t t22 = _mm256_extract_epi64(t16, 1);
            int64_t t23 = _mm256_extract_epi64(t16, 2);
            int64_t t24 = _mm256_extract_epi64(t16, 3);
            int64_t t25 = ((mask.mMask & 0x10) == 0) ? t17 : t17 * t21;
            int64_t t26 = ((mask.mMask & 0x20) == 0) ? t18 : t18 * t22;
            int64_t t27 = ((mask.mMask & 0x40) == 0) ? t19 : t19 * t23;
            int64_t t28 = ((mask.mMask & 0x80) == 0) ? t20 : t20 * t24;
            mVec = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm256_mullo_epi64(t4, t2);
            mVec = _mm512_inserti64x4(mVec, t3, 0);
            mVec = _mm512_inserti64x4(mVec, t5, 1);
#else
            mVec = _mm512_mullo_epi64(mVec, _mm256_set1_epi64(b));
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            int64_t t2 = _mm256_extract_epi64(t1, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 1);
            int64_t t4 = _mm256_extract_epi64(t1, 2);
            int64_t t5 = _mm256_extract_epi64(t1, 3);
            int64_t t6 = t2 * b;
            int64_t t7 = t3 * b;
            int64_t t8 = t4 * b;
            int64_t t9 = t5 * b;
            __m256i t10 = _mm512_extracti64x4_epi64(mVec, 1);
            int64_t t11 = _mm256_extract_epi64(t10, 0);
            int64_t t12 = _mm256_extract_epi64(t10, 1);
            int64_t t13 = _mm256_extract_epi64(t10, 2);
            int64_t t14 = _mm256_extract_epi64(t10, 3);
            int64_t t15 = t11 * b;
            int64_t t16 = t12 * b;
            int64_t t17 = t13 * b;
            int64_t t18 = t14 * b;
            mVec = _mm512_set_epi64(t18, t17, t16, t15, t9, t8, t7, t6);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int64_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<8> const & mask, int64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec, 1);
            __m256i t5 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t2);
            mVec = _mm512_inserti64x4(mVec, t3, 0);
            mVec = _mm512_inserti64x4(mVec, t5, 1);
#else
            mVec = _mm512_mullo_epi64(mVec, _mm256_set1_epi64(b));
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec, 0);
            int64_t t2 = _mm256_extract_epi64(t1, 0);
            int64_t t3 = _mm256_extract_epi64(t1, 1);
            int64_t t4 = _mm256_extract_epi64(t1, 2);
            int64_t t5 = _mm256_extract_epi64(t1, 3);
            int64_t t6 = ((mask.mMask & 0x1) == 0) ? t2 : t2 * b;
            int64_t t7 = ((mask.mMask & 0x2) == 0) ? t3 : t3 * b;
            int64_t t8 = ((mask.mMask & 0x4) == 0) ? t4 : t4 * b;
            int64_t t9 = ((mask.mMask & 0x8) == 0) ? t5 : t5 * b;
            __m256i t10 = _mm512_extracti64x4_epi64(mVec, 1);
            int64_t t11 = _mm256_extract_epi64(t10, 0);
            int64_t t12 = _mm256_extract_epi64(t10, 1);
            int64_t t13 = _mm256_extract_epi64(t10, 2);
            int64_t t14 = _mm256_extract_epi64(t10, 3);
            int64_t t15 = ((mask.mMask & 0x10) == 0) ? t11 : t11 * b;
            int64_t t16 = ((mask.mMask & 0x20) == 0) ? t12 : t12 * b;
            int64_t t17 = ((mask.mMask & 0x40) == 0) ? t13 : t13 * b;
            int64_t t18 = ((mask.mMask & 0x80) == 0) ? t14 : t14 * b;
            mVec = _mm512_set_epi64(t18, t17, t16, t15, t9, t8, t7, t6);
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
        /*UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
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
        /*UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<8> const & mask, int64_t b) const {
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
        /*UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        /*UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<8> const & mask, int64_t b) {
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
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec, b.mVec);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq (int64_t b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec, _mm512_set1_epi64(b));
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (int64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpneq_epi64_mask(mVec, b.mVec);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne (int64_t b) const {
            __mmask8 m0 = _mm512_cmpneq_epi64_mask(mVec, _mm512_set1_epi64(b));
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (int64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpgt_epi64_mask(mVec, b.mVec);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt (int64_t b) const {
            __mmask8 m0 = _mm512_cmpgt_epi64_mask(mVec, _mm512_set1_epi64(b));
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (int64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmplt_epi64_mask(mVec, b.mVec);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt (int64_t b) const {
            __mmask8 m0 = _mm512_cmplt_epi64_mask(mVec, _mm512_set1_epi64(b));
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (int64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpge_epi64_mask(mVec, b.mVec);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge (int64_t b) const {
            __mmask8 m0 = _mm512_cmpge_epi64_mask(mVec, _mm512_set1_epi64(b));
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (int64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmple_epi64_mask(mVec, b.mVec);
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple (int64_t b) const {
            __mmask8 m0 = _mm512_cmple_epi64_mask(mVec, _mm512_set1_epi64(b));
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (int64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec, b.mVec);
            return m0 == 0x03;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int64_t b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec, _mm512_set1_epi64(b));
            return m0 == 0x03;
        }
        // UNIQUE
        // HADD
        UME_FORCE_INLINE int64_t hadd() const {
            int64_t retval = _mm512_reduce_add_epi64(mVec);
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE int64_t hadd(SIMDVecMask<8> const & mask) const {
            int64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, mVec);
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE int64_t hadd(int64_t b) const {
            int64_t retval = _mm512_reduce_add_epi64(mVec);
            return retval + b;
        }
        // MHADDS
        UME_FORCE_INLINE int64_t hadd(SIMDVecMask<8> const & mask, int64_t b) const {
            int64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, mVec);
            return retval + b;
        }
        // HMUL
        UME_FORCE_INLINE int64_t hmul() const {
            int64_t retval = _mm512_reduce_mul_epi64(mVec);
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE int64_t hmul(SIMDVecMask<8> const & mask) const {
            int64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask, mVec);
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE int64_t hmul(int64_t b) const {
            int64_t retval = _mm512_reduce_mul_epi64(mVec);
            return retval * b;
        }
        // MHMULS
        UME_FORCE_INLINE int64_t hmul(SIMDVecMask<8> const & mask, int64_t b) const {
            int64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask, mVec);
            return retval * b;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(b)).add(c);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(mask, b)).add(mask, c);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(b)).sub(c);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (mul(mask, b)).sub(mask, c);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (add(b)).mul(c);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (add(mask, b)).mul(mask, c);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (sub(b)).mul(c);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            return (sub(mask, b)).mul(mask, c);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_max_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_max_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int64_t b) const {
            __m512i t0 = _mm512_max_epi64(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_max_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = _mm512_max_epi64(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_max_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int64_t b) {
            mVec = _mm512_max_epi64(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_max_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_min_epi64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_min_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int64_t b) const {
            __m512i t0 = _mm512_min_epi64(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_min_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = _mm512_min_epi64(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_min_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int64_t b) {
            mVec = _mm512_min_epi64(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_min_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE int64_t hmax() const {
            int64_t t0 = _mm512_reduce_max_epi64(mVec);
            return t0;
        }
        // MHMAX
        UME_FORCE_INLINE int64_t hmax(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_set1_epi64(std::numeric_limits<int64_t>::min());
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            int64_t t2 = _mm512_reduce_max_epi64(t1);
            return t2;
        }
        // IMAX
        /*UME_FORCE_INLINE int64_t imax() const {
            return mVec[0] > mVec[1] ? 0 : 1;
        }*/
        // MIMAX
        /*UME_FORCE_INLINE int64_t imax(SIMDVecMask<8> const & mask) const {
            int64_t i0 = 0xFFFFFFFFFFFFFFFF;
            int64_t t0 = std::numeric_limits<int64_t>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] > t0) {
                i0 = 1;
            }
            return i0;
        }*/
        // HMIN
        UME_FORCE_INLINE int64_t hmin() const {
            int64_t t0 = _mm512_reduce_min_epi64(mVec);
            return t0;
        }
        // MHMIN
        UME_FORCE_INLINE int64_t hmin(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_set1_epi64(std::numeric_limits<int64_t>::max());
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            int64_t t2 = _mm512_reduce_min_epi64(t1);
            return t2;
        }
        // IMIN
        /*UME_FORCE_INLINE int64_t imin() const {
            return mVec[0] < mVec[1] ? 0 : 1;
        }*/
        // MIMIN
        /*UME_FORCE_INLINE int64_t imin(SIMDVecMask<8> const & mask) const {
            int64_t i0 = 0xFFFFFFFFFFFFFFFF;
            int64_t t0 = std::numeric_limits<int64_t>::max();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0];
            }
            if(mask.mMask[1] == true && mVec[1] < t0) {
                i0 = 1;
            }
            return i0;
        }*/

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_si512(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_and_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int64_t b) const {
            __m512i t0 = _mm512_and_si512(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_and_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm512_and_si512(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_and_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int64_t b) {
            mVec = _mm512_and_si512(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_and_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_si512(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_or_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int64_t b) const {
            __m512i t0 = _mm512_or_si512(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_or_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm512_or_si512(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_or_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int64_t b) {
            mVec = _mm512_or_si512(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_or_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_si512(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_xor_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int64_t b) const {
            __m512i t0 = _mm512_xor_si512(mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_xor_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return SIMDVec_i(t0);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm512_xor_si512(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm512_mask_xor_epi64(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int64_t b) {
            mVec = _mm512_xor_si512(mVec, _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<8> const & mask, int64_t b) {
            mVec = _mm512_mask_xor_epi64(mVec, mask.mMask, mVec, _mm512_set1_epi64(b));
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            __m512i t0 = _mm512_xor_si512(mVec, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            __m512i t1 = _mm512_mask_xor_epi64(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            mVec = _mm512_xor_si512(mVec, _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<8> const & mask) {
            __m512i t0 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            mVec = _mm512_mask_xor_epi64(mVec, mask.mMask, mVec, t0);
            return *this;
        }

        //HBAND
        UME_FORCE_INLINE int64_t hband() const {
            int64_t t0 = _mm512_reduce_and_epi64(mVec);
            return t0;
        }
        // MHBAND
        UME_FORCE_INLINE int64_t hband(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            int64_t t2 = _mm512_reduce_and_epi64(t1);
            return t2;
        }
        // HBANDS
        UME_FORCE_INLINE int64_t hband(int64_t b) const {
            int64_t t0 = _mm512_reduce_and_epi64(mVec);
            return t0 & b;
        }
        // MHBANDS
        UME_FORCE_INLINE int64_t hband(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            int64_t t2 = _mm512_reduce_and_epi64(t1);
            return t2 & b;
        }
        // HBOR
        UME_FORCE_INLINE int64_t hbor() const {
            int64_t t0 = _mm512_reduce_or_epi64(mVec);
            return t0;
        }
        // MHBOR
        UME_FORCE_INLINE int64_t hbor(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_set1_epi64(0);
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            int64_t t2 = _mm512_reduce_or_epi64(t1);
            return t2;
        }
        // HBORS
        UME_FORCE_INLINE int64_t hbor(int64_t b) const {
            int64_t t0 = _mm512_reduce_or_epi64(mVec);
            return t0 | b;
        }
        // MHBORS
        UME_FORCE_INLINE int64_t hbor(SIMDVecMask<8> const & mask, int64_t b) const {
            __m512i t0 = _mm512_set1_epi64(0);
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            int64_t t2 = _mm512_reduce_or_epi64(t1);
            return t2 | b;
        }
        // HBXOR
        UME_FORCE_INLINE int64_t hbxor() const {
            alignas(64) int64_t raw[8];
            _mm512_store_si512(raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXOR
        UME_FORCE_INLINE int64_t hbxor(SIMDVecMask<8> const & mask) const {
            alignas(64) int64_t raw[8];
            __m512i t0 = _mm512_set1_epi64(0);
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            _mm512_store_si512(raw, t1);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // HBXORS
        UME_FORCE_INLINE int64_t hbxor(int64_t b) const {
            alignas(64) int64_t raw[8];
            _mm512_store_si512(raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE int64_t hbxor(SIMDVecMask<8> const & mask, int64_t b) const {
            alignas(64) int64_t raw[8];
            __m512i t0 = _mm512_set1_epi64(0);
            __m512i t1 = _mm512_mask_mov_epi64(t0, mask.mMask, mVec);
            _mm512_store_si512(raw, t1);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ raw[4] ^ raw[5] ^ raw[6] ^ raw[7] ^ b;
        }

        // GATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(int64_t const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            mVec = _mm512_i64gather_epi64(t2, (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(SIMDVecMask<8> const & mask, int64_t const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            __m512i t3 = _mm512_i64gather_epi64(t2, (__int64 const*)baseAddr, 8);
            mVec = _mm512_mask_mov_epi64(mVec, mask.mMask, t3);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int64_t const * baseAddr, uint64_t const * indices) {
            __m512i t0 =_mm512_loadu_si512((__m512i *)indices);
            mVec = _mm512_i64gather_epi64(t0, (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<8> const & mask, int64_t const * baseAddr, uint64_t const * indices) {
            __m512i t0 = _mm512_loadu_si512((__m512i *)indices);
            __m512i t1 = _mm512_i64gather_epi64(t0, (__int64 const*)baseAddr, 8);
            mVec = _mm512_mask_mov_epi64(mVec, mask.mMask, t1);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(int64_t const * baseAddr, SIMDVec_u<uint64_t, 8> const & indices) {
            mVec = _mm512_i64gather_epi64(indices.mVec, (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<8> const & mask, int64_t const * baseAddr, SIMDVec_u<uint64_t, 8> const & indices) {
            __m512i t0 = _mm512_i64gather_epi64(indices.mVec, (__int64 const*)baseAddr, 8);
            mVec = _mm512_mask_mov_epi64(mVec, mask.mMask, t0);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE int64_t* scatteru(int64_t* baseAddr, uint64_t stride) const {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            _mm512_i64scatter_epi64(baseAddr, t2, mVec, 8);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE int64_t* scatteru(SIMDVecMask<8> const & mask, int64_t* baseAddr, uint64_t stride) const {
#if defined (__AVX512DQ__)
            __m512i t0 = _mm512_set1_epi64(stride);
            __m512i t1 = _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7);
            __m512i t2 = _mm512_mullo_epi64(t0, t1);
#else
            __m512i t2 = _mm512_setr_epi64(0, stride, 2*stride, 3*stride, 4*stride, 5*stride, 6*stride, 7*stride);
#endif
            _mm512_mask_i64scatter_epi64(baseAddr, mask.mMask, t2, mVec, 8);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE int64_t* scatter(int64_t* baseAddr, uint64_t* indices) const {
            __m512i t0 = _mm512_loadu_si512((__m512i *)indices);
            _mm512_i64scatter_epi64(baseAddr, t0, mVec, 8);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE int64_t* scatter(SIMDVecMask<8> const & mask, int64_t* baseAddr, uint64_t* indices) const {
            __m512i t0 = _mm512_loadu_si512((__m512i *)indices);
            _mm512_mask_i64scatter_epi64(baseAddr, mask.mMask, t0, mVec, 8);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int64_t* scatter(int64_t* baseAddr, SIMDVec_u<uint64_t, 8> const & indices) const {
            _mm512_i64scatter_epi64(baseAddr, indices.mVec, mVec, 8);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int64_t* scatter(SIMDVecMask<8> const & mask, int64_t* baseAddr, SIMDVec_u<uint64_t, 8> const & indices) const {
            _mm512_mask_i64scatter_epi64(baseAddr, mask.mMask, indices.mVec, mVec, 8);
            return baseAddr;
        }

        //// LSHV
        //UME_FORCE_INLINE SIMDVec_i lsh(SIMDVec_i const & b) const {
        //    int64_t t0 = mVec[0] << b.mVec[0];
        //    int64_t t1 = mVec[1] << b.mVec[1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// MLSHV
        //UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// LSHS
        //UME_FORCE_INLINE SIMDVec_i lsh(int64_t b) const {
        //    int64_t t0 = mVec[0] << b;
        //    int64_t t1 = mVec[1] << b;
        //    return SIMDVec_i(t0, t1);
        //}
        //// MLSHS
        //UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<8> const & mask, int64_t b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// LSHVA
        //UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVec_i const & b) {
        //    mVec[0] = mVec[0] << b.mVec[0];
        //    mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// MLSHVA
        //UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// LSHSA
        //UME_FORCE_INLINE SIMDVec_i & lsha(int64_t b) {
        //    mVec[0] = mVec[0] << b;
        //    mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        //// MLSHSA
        //UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<8> const & mask, int64_t b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b;
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        //// RSHV
        //UME_FORCE_INLINE SIMDVec_i rsh(SIMDVec_i const & b) const {
        //    int64_t t0 = mVec[0] >> b.mVec[0];
        //    int64_t t1 = mVec[1] >> b.mVec[1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// MRSHV
        //UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// RSHS
        //UME_FORCE_INLINE SIMDVec_i rsh(int64_t b) const {
        //    int64_t t0 = mVec[0] >> b;
        //    int64_t t1 = mVec[1] >> b;
        //    return SIMDVec_i(t0, t1);
        //}
        //// MRSHS
        //UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<8> const & mask, int64_t b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
        //    int64_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// RSHVA
        //UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVec_i const & b) {
        //    mVec[0] = mVec[0] >> b.mVec[0];
        //    mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// MRSHVA
        //UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
        //    if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
        //    if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// RSHSA
        //UME_FORCE_INLINE SIMDVec_i & rsha(int64_t b) {
        //    mVec[0] = mVec[0] >> b;
        //    mVec[1] = mVec[1] >> b;
        //    return *this;
        //}
        //// MRSHSA
        //UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<8> const & mask, int64_t b) {
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
            __m512i t0 = _mm512_sub_epi64(_mm512_setzero_si512(), mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_mask_sub_epi64(mVec, mask.mMask, _mm512_setzero_si512(), mVec);
            return SIMDVec_i(t0);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec = _mm512_sub_epi64(_mm512_setzero_si512(), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_sub_epi64(mVec, mask.mMask, _mm512_setzero_si512(), mVec);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            __m512i t0 = _mm512_abs_epi64(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_mask_abs_epi64(mVec, mask.mMask, mVec);
            return SIMDVec_i(t0);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            mVec = _mm512_abs_epi64(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<8> const & mask) {
            mVec = _mm512_mask_abs_epi64(mVec, mask.mMask, mVec);
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int64_t, 4> const & a, SIMDVec_i<int64_t, 4> const & b) {
            __m512i t0 = _mm512_castsi256_si512(a.mVec);
            mVec = _mm512_inserti64x4(t0, b.mVec, 1);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int64_t, 4> const & a) {
            mVec = _mm512_inserti64x4(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_i & packhi(SIMDVec_i<int64_t, 4> const & b) {
            mVec = _mm512_inserti64x4(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        void unpack(SIMDVec_i<int64_t, 4> & a, SIMDVec_i<int64_t, 4> & b) const {
            a.mVec = _mm512_extracti64x4_epi64(mVec, 0);
            b.mVec = _mm512_extracti64x4_epi64(mVec, 1);
        }
        // UNPACKLO
        SIMDVec_i<int64_t, 4> unpacklo() const {
            __m256i t0 = _mm512_extracti64x4_epi64(mVec, 0);
            return SIMDVec_i<int64_t, 4> (t0);
        }
        // UNPACKHI
        SIMDVec_i<int64_t, 4> unpackhi() const {
            __m256i t0 = _mm512_extracti64x4_epi64(mVec, 1);
            return SIMDVec_i<int64_t, 4>(t0);
        }

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 8>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 8>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 8>() const;
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
