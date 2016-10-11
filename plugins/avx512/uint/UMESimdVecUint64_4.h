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

#ifndef UME_SIMD_VEC_UINT64_4_H_
#define UME_SIMD_VEC_UINT64_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define SET1_EPI64(x) _mm256_set1_epi64x(x)

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
    class SIMDVec_u<uint64_t, 4> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint64_t, 4>,
            uint64_t,
            4,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint64_t, 4>,
            SIMDVec_u<uint64_t, 2>>
    {
    public:
        friend class SIMDVec_i<int64_t, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_u<uint64_t, 8>;

    private:
        __m256i mVec;

        UME_FORCE_INLINE explicit SIMDVec_u(__m256i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_u(const __m256i & x) { mVec = x; }

    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i) {
            mVec = SET1_EPI64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, uint64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint64_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint64_t const *p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t i3) {
            mVec = _mm256_set_epi64x(i3, i2, i1, i0);
        }

        // EXTRACT
        UME_FORCE_INLINE uint64_t extract(uint32_t index) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*) raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE uint64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint64_t value) {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*) raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*) raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint64_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_epi64(mVec, mask.mMask, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi64(t0, mask.mMask, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint64_t b) {
            mVec = SET1_EPI64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<4> const & mask, uint64_t b) {
            __m256i t1 = SET1_EPI64(b);
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_epi64(mVec, mask.mMask, t1);
#else
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_castsi256_si512(t1);
            __m512i t4 = _mm512_mask_mov_epi64(t2, mask.mMask, t3);
            mVec = _mm512_castsi512_si256(t4);
#endif
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint64_t const *p) {
            mVec = _mm256_loadu_si256((const __m256i *) p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<4> const & mask, uint64_t const *p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_loadu_epi64(mVec, mask.mMask, p);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint64_t const *p) {
            mVec = _mm256_load_si256((const __m256i *) p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<4> const & mask, uint64_t const *p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_load_epi64(mVec, mask.mMask, p);
#else
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint64_t* store(uint64_t* p) const {
            _mm256_storeu_si256((__m256i *)p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint64_t* store(SIMDVecMask<4> const & mask, uint64_t* p) const {
#if defined(__AVX512VL__)
            _mm256_mask_storeu_epi64(p, mask.mMask, mVec);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            __m256i t4 = _mm512_castsi512_si256(t3);
            _mm256_storeu_si256((__m256i*)p, t4);
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint64_t* storea(uint64_t* p) const {
            _mm256_store_si256((__m256i *)p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint64_t* storea(SIMDVecMask<4> const & mask, uint64_t* p) const {
#if defined(__AVX512VL__)
            _mm256_mask_store_epi64(p, mask.mMask, mVec);
#else
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi64(t1, mask.mMask, t2);
            __m256i t4 = _mm512_castsi512_si256(t3);
            _mm256_store_si256((__m256i*)p, t4);
#endif
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mov_epi64(mVec, mask.mMask, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi64(t0, mask.mMask, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mov_epi64(mVec, mask.mMask, SET1_EPI64(b));
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(b);
            __m512i t2 = _mm512_mask_mov_epi64(t0, mask.mMask, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // SWIZZLE
        // SWIZZLEA

        // SORTA
        // SORTD

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi64(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_add_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint64_t b) const {
            __m256i t0 = _mm256_add_epi64(mVec, SET1_EPI64(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_add_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm256_add_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_add_epi64);
#endif
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint64_t b) {
            mVec = _mm256_add_epi64(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
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
        UME_FORCE_INLINE SIMDVec_u postinc() {
            __m256i t0 = SET1_EPI64(1);
            __m256i t1 = mVec;
            mVec = _mm256_add_epi64(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<4> const & mask) {
            __m256i t0 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m256i t0 = SET1_EPI64(1);
            mVec = _mm256_add_epi64(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_add_epi64);
#endif
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi64(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint64_t b) const {
            __m256i t0 = _mm256_sub_epi64(mVec, SET1_EPI64(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm256_sub_epi64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint64_t b) {
            mVec = _mm256_sub_epi64(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
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
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi64(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi64(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(b.mVec, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint64_t b) const {
            __m256i t0 = _mm256_sub_epi64(SET1_EPI64(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<4> const & mask, uint64_t b) const {
            __m256i t0 = SET1_EPI64(b);
#if defined(__AVX512VL__)
            __m256i t1 = _mm256_mask_sub_epi64(t0, mask.mMask, t0, mVec);
#else
            __m256i t1 = EXPAND_CALL_BINARY_MASK(t0, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_u(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm256_sub_epi64(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi64(b.mVec, mask.mMask, b.mVec, mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(b.mVec, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint64_t b) {
            mVec = _mm256_sub_epi64(SET1_EPI64(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, uint64_t b) {
            __m256i t0 = SET1_EPI64(b);
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi64(t0, mask.mMask, t0, mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(t0, mVec, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m256i t0 = SET1_EPI64(1);
            __m256i t1 = mVec;
            mVec = _mm256_sub_epi64(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<4> const & mask) {
            __m256i t0 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            mVec = _mm256_sub_epi64(mVec, SET1_EPI64(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi64(mVec, mask.mMask, mVec, SET1_EPI64(1));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 1, mask.mMask, _mm512_mask_sub_epi64);
#endif
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
#if defined(__AVX512DQ__)
    #if defined(__AVX512VL__)
            __m256i t0 = _mm256_mullo_epi64(mVec, b.mVec);
    #else
            __m256i t0 = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_mullo_epi64);
    #endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = _mm256_extract_epi64(b.mVec, 0);
            uint64_t t6 = _mm256_extract_epi64(b.mVec, 1);
            uint64_t t7 = _mm256_extract_epi64(b.mVec, 2);
            uint64_t t8 = _mm256_extract_epi64(b.mVec, 3);
            uint64_t t9 = t1 * t5;
            uint64_t t10 = t2 * t6;
            uint64_t t11 = t3 * t7;
            uint64_t t12 = t4 * t8;
            __m256i t0 = _mm256_set_epi64x(t12, t11, t10, t9);
#endif
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mullo_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_mask(mVec, b.mVec, mask.mMask, _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = _mm256_extract_epi64(b.mVec, 0);
            uint64_t t6 = _mm256_extract_epi64(b.mVec, 1);
            uint64_t t7 = _mm256_extract_epi64(b.mVec, 2);
            uint64_t t8 = _mm256_extract_epi64(b.mVec, 3);
            uint64_t t9 = ((mask.mMask & 0x1) != 0) ? t1 * t5 : t1;
            uint64_t t10 = ((mask.mMask & 0x2) != 0) ? t2 * t6 : t2;
            uint64_t t11 = ((mask.mMask & 0x4) != 0) ? t3 * t7 : t3;
            uint64_t t12 = ((mask.mMask & 0x8) != 0) ? t4 * t8 : t4;
            __m256i t0 = _mm256_set_epi64x(t12, t11, t10, t9);
#endif
            return SIMDVec_u(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mullo_epi64(mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = t1 * b;
            uint64_t t6 = t2 * b;
            uint64_t t7 = t3 * b;
            uint64_t t8 = t4 * b;
            __m256i t0 = _mm256_set_epi64x(t8, t7, t6, t5);
#endif
                return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mullo_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_mask(mVec, SET1_EPI64(b), mask.mMask, _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = ((mask.mMask & 0x1) != 0) ? t1 * b : t1;
            uint64_t t6 = ((mask.mMask & 0x2) != 0) ? t2 * b : t2;
            uint64_t t7 = ((mask.mMask & 0x4) != 0) ? t3 * b : t3;
            uint64_t t8 = ((mask.mMask & 0x8) != 0) ? t4 * b : t4;
            __m256i t0 = _mm256_set_epi64x(t8, t7, t6, t5);
#endif
                return SIMDVec_u(t0);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm256_mullo_epi64(mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = _mm256_extract_epi64(b.mVec, 0);
            uint64_t t6 = _mm256_extract_epi64(b.mVec, 1);
            uint64_t t7 = _mm256_extract_epi64(b.mVec, 2);
            uint64_t t8 = _mm256_extract_epi64(b.mVec, 3);
            uint64_t t9 = t1 * t5;
            uint64_t t10 = t2 * t6;
            uint64_t t11 = t3 * t7;
            uint64_t t12 = t4 * t8;
            mVec = _mm256_set_epi64x(t12, t11, t10, t9);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mullo_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_mask(mVec, b.mVec, mask.mMask, _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = _mm256_extract_epi64(b.mVec, 0);
            uint64_t t6 = _mm256_extract_epi64(b.mVec, 1);
            uint64_t t7 = _mm256_extract_epi64(b.mVec, 2);
            uint64_t t8 = _mm256_extract_epi64(b.mVec, 3);
            uint64_t t9 = ((mask.mMask & 0x1) != 0) ? t1 * t5 : t1;
            uint64_t t10 = ((mask.mMask & 0x2) != 0) ? t2 * t6 : t2;
            uint64_t t11 = ((mask.mMask & 0x4) != 0) ? t3 * t7 : t3;
            uint64_t t12 = ((mask.mMask & 0x8) != 0) ? t4 * t8 : t4;
            mVec = _mm256_set_epi64x(t12, t11, t10, t9);
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm256_mullo_epi64(mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = t1 * b;
            uint64_t t6 = t2 * b;
            uint64_t t7 = t3 * b;
            uint64_t t8 = t4 * b;
            mVec = _mm256_set_epi64x(t8, t7, t6, t5);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint64_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mullo_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_mask(mVec, SET1_EPI64(b), mask.mMask, _mm512_mullo_epi64);
#endif
#else
            uint64_t t1 = _mm256_extract_epi64(mVec, 0);
            uint64_t t2 = _mm256_extract_epi64(mVec, 1);
            uint64_t t3 = _mm256_extract_epi64(mVec, 2);
            uint64_t t4 = _mm256_extract_epi64(mVec, 3);
            uint64_t t5 = ((mask.mMask & 0x1) != 0) ? t1 * b : t1;
            uint64_t t6 = ((mask.mMask & 0x2) != 0) ? t2 * b : t2;
            uint64_t t7 = ((mask.mMask & 0x4) != 0) ? t3 * b : t3;
            uint64_t t8 = ((mask.mMask & 0x8) != 0) ? t4 * b : t4;
            mVec = _mm256_set_epi64x(t8, t7, t6, t5);
#endif
            return *this;
        }
        // DIVV
        /*UME_FORCE_INLINE SIMDVec_u div(SIMDVec_u const & b) const {
            uint64_t t0 = mVec[0] / b.mVec[0];
            uint64_t t1 = mVec[1] / b.mVec[1];
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }*/
        // MDIVV
        /*UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            uint64_t t1 = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // DIVS
        /*UME_FORCE_INLINE SIMDVec_u div(uint64_t b) const {
            uint64_t t0 = mVec[0] / b;
            uint64_t t1 = mVec[1] / b;
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (uint64_t b) const {
            return div(b);
        }*/
        // MDIVS
        /*UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<4> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] / b : mVec[0];
            uint64_t t1 = mask.mMask[1] ? mVec[1] / b : mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // DIVVA
        /*UME_FORCE_INLINE SIMDVec_u & diva(SIMDVec_u const & b) {
            mVec[0] /= b.mVec[0];
            mVec[1] /= b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }*/
        // MDIVVA
        /*UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec[0] = mask.mMask[0] ? mVec[0] / b.mVec[0] : mVec[0];
            mVec[1] = mask.mMask[1] ? mVec[1] / b.mVec[1] : mVec[1];
            return *this;
        }*/
        // DIVSA
        /*UME_FORCE_INLINE SIMDVec_u & diva(uint64_t b) {
            mVec[0] /= b;
            mVec[1] /= b;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (uint64_t b) {
            return diva(b);
        }*/
        // MDIVSA
        /*UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<4> const & mask, uint64_t b) {
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
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = 0xF & _mm512_cmpeq_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_castsi256_si512(b.mVec));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq (uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpeq_epu64_mask(
                            _mm512_castsi256_si512(mVec), 
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (uint64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpneq_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpneq_epu64_mask(
                            _mm512_castsi256_si512(mVec), 
                            _mm512_castsi256_si512(b.mVec));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne (uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpneq_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpneq_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (uint64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpgt_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpgt_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_castsi256_si512(b.mVec));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt (uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpgt_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpgt_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (uint64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmplt_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmplt_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_castsi256_si512(b.mVec));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt (uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmplt_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmplt_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (uint64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpge_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpge_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_castsi256_si512(b.mVec));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge (uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpge_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpge_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (uint64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmple_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmple_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_castsi256_si512(b.mVec));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple (uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmple_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmple_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (uint64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu64_mask(mVec, b.mVec);
#else
            __mmask8 m0 = _mm512_cmpeq_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_castsi256_si512(b.mVec));
#endif
            return m0 == 0x03;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint64_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu64_mask(mVec, SET1_EPI64(b));
#else
            __mmask8 m0 = _mm512_cmpeq_epu64_mask(
                            _mm512_castsi256_si512(mVec),
                            _mm512_set1_epi64(b));
#endif
            return m0 == 0x03;
        }
        // UNIQUE
        // HADD
        UME_FORCE_INLINE uint64_t hadd() const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_reduce_add_epi64(t0);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE uint64_t hadd(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, t0);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE uint64_t hadd(uint64_t b) const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_reduce_add_epi64(t0);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE uint64_t hadd(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = b;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, t0);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE uint64_t hmul() const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_mask_reduce_mul_epi64(0xF, t0);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE uint64_t hmul(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask, t0);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE uint64_t hmul(uint64_t b) const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_mask_reduce_mul_epi64(0xF, t0);
            return retval * b;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE uint64_t hmul(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined (__GNUG__)
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = b;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask, t0);
            return retval * b;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (mul(b)).add(c);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (mul(mask, b)).add(mask, c);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (mul(b)).sub(c);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (mul(mask, b)).sub(mask, c);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (add(b)).mul(c);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (add(mask, b)).mul(mask, c);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (sub(b)).mul(c);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            return (sub(mask, b)).mul(mask, c);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_max_epu64(mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_max_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_max_epu64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_max_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_max_epu64(mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_max_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_max_epu64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_max_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_max_epu64(mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_max_epu64);
#endif
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_max_epu64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_max_epu64);
#endif
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_max_epu64(mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_max_epu64);
#endif
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_max_epu64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_max_epu64);
#endif
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_min_epu64(mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_min_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_min_epu64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_min_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_min_epu64(mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_min_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_min_epu64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_min_epu64);
#endif
            return SIMDVec_u(t0);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_min_epu64(mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY(mVec, b.mVec, _mm512_min_epu64);
#endif
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_min_epu64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_min_epu64);
#endif
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_min_epu64(mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY(mVec, SET1_EPI64(b), _mm512_min_epu64);
#endif
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_min_epu64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_min_epu64);
#endif
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint64_t hmax() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = raw[0] > raw[1] ? raw[0] : raw[1];
            uint64_t t1 = raw[2] > raw[3] ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
        }
        // MHMAX
        UME_FORCE_INLINE uint64_t hmax(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] : std::numeric_limits<uint64_t>::min();
            uint64_t t1 = ((mask.mMask & 0x2) && raw[1] > t0) ? raw[1] : t0;
            uint64_t t2 = ((mask.mMask & 0x4) && raw[2] > t1) ? raw[2] : t1;
            uint64_t t3 = ((mask.mMask & 0x8) && raw[3] > t2) ? raw[3] : t2;
            return t3;
        }
        // IMAX
        UME_FORCE_INLINE uint32_t imax() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = raw[0] > raw[1] ? 0 : 1;
            uint64_t t1 = raw[2] > raw[3] ? 2 : 3;
            return raw[t0] > raw[t1] ? t0 : t1;
        }
        // MIMAX
        UME_FORCE_INLINE uint32_t imax(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t i0 = 0xFFFFFFFF;
            uint64_t t0 = std::numeric_limits<uint64_t>::min();
            if ((mask.mMask & 0x1) != 0) {
                i0 = 0;
                t0 = raw[0];
            }
            if (((mask.mMask & 0x2) != 0) && raw[1] > t0) {
                i0 = 1;
                t0 = raw[1];
            }
            if (((mask.mMask & 0x4) != 0) && raw[2] > t0) {
                i0 = 2;
                t0 = raw[2];
            }
            if (((mask.mMask & 0x8) != 0) && raw[3] > t0) {
                i0 = 3;
            }
            return i0;
        }
        // HMIN
        UME_FORCE_INLINE uint64_t hmin() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = raw[0] < raw[1] ? raw[0] : raw[1];
            uint64_t t1 = raw[2] < raw[3] ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
        }
        // MHMIN
        UME_FORCE_INLINE uint64_t hmin(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] : std::numeric_limits<uint64_t>::max();
            uint64_t t1 = ((mask.mMask & 0x2) && raw[1] < t0) ? raw[1] : t0;
            uint64_t t2 = ((mask.mMask & 0x4) && raw[2] < t1) ? raw[2] : t1;
            uint64_t t3 = ((mask.mMask & 0x8) && raw[3] < t2) ? raw[3] : t2;
            return t3;
        }
        // IMIN
        UME_FORCE_INLINE uint32_t imin() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = raw[0] < raw[1] ? 0 : 1;
            uint32_t t1 = raw[2] < raw[3] ? 2 : 3;
            return raw[t0] < raw[t1] ? t0 : t1;
        }
        // MIMIN
        UME_FORCE_INLINE uint64_t imin(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t i0 = 0xFFFFFFFF;
            uint64_t t0 = std::numeric_limits<uint64_t>::max();
            if ((mask.mMask & 0x1) != 0) {
                i0 = 0;
                t0 = raw[0];
            }
            if (((mask.mMask & 0x2) != 0) && raw[1] < t0) {
                i0 = 1;
                t0 = raw[1];
            }
            if (((mask.mMask & 0x4) != 0) && raw[2] < t0) {
                i0 = 2;
                t0 = raw[2];
            }
            if (((mask.mMask & 0x8) != 0) && raw[3] < t0) {
                i0 = 3;
            }
            return i0;
        }

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_and_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_and_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint64_t b) const {
            __m256i t0 = _mm256_and_si256(mVec, SET1_EPI64(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_and_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_and_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm256_and_si256(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_and_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_and_epi64);
#endif
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint64_t b) {
            mVec = _mm256_and_si256(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_and_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_and_epi64);
#endif
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_or_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_or_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint64_t b) const {
            __m256i t0 = _mm256_or_si256(mVec, SET1_EPI64(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_or_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_or_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm256_or_si256(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_or_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_or_epi64);
#endif
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint64_t b) {
            mVec = _mm256_or_si256(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_or_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_or_epi64);
#endif
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_xor_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            __m256i t0 = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint64_t b) const {
            __m256i t0 = _mm256_xor_si256(mVec, SET1_EPI64(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_xor_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            __m256i t0 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return SIMDVec_u(t0);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm256_xor_si256(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_xor_epi64(mVec, mask.mMask, mVec, b.mVec);
#else
            mVec = EXPAND_CALL_BINARY_MASK(mVec, b.mVec, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint64_t b) {
            mVec = _mm256_xor_si256(mVec, SET1_EPI64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<4> const & mask, uint64_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_xor_epi64(mVec, mask.mMask, mVec, SET1_EPI64(b));
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, b, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m256i t0 = _mm256_xor_si256(mVec, _mm256_set1_epi32(0xFFFFFFFF));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_mask_xor_epi64(mVec, mask.mMask, mVec, t0);
#else
            __m256i t1 = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 0xFFFFFFFFFFFFFFFF, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return SIMDVec_u(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            mVec = _mm256_xor_si256(mVec, _mm256_set1_epi32(0xFFFFFFFF));
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_mask_xor_epi64(mVec, mask.mMask, mVec, t0);
#else
            mVec = EXPAND_CALL_BINARY_SCALAR_MASK(mVec, 0xFFFFFFFFFFFFFFFF, mask.mMask, _mm512_mask_xor_epi64);
#endif
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint64_t hband() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3];
        }
        // MHBAND
        UME_FORCE_INLINE uint64_t hband(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] : 0xFFFFFFFFFFFFFFFF;
            uint64_t t1 = (mask.mMask & 0x2) ? raw[1] & t0 : t0;
            uint64_t t2 = (mask.mMask & 0x4) ? raw[2] & t1 : t1;
            uint64_t t3 = (mask.mMask & 0x8) ? raw[3] & t2 : t2;
            return t3;
        }
        // HBANDS
        UME_FORCE_INLINE uint64_t hband(uint64_t b) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3] & b;
        }
        // MHBANDS
        UME_FORCE_INLINE uint64_t hband(SIMDVecMask<4> const & mask, uint64_t b) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] & b : b;
            uint64_t t1 = (mask.mMask & 0x2) ? raw[1] & t0 : t0;
            uint64_t t2 = (mask.mMask & 0x4) ? raw[2] & t1 : t1;
            uint64_t t3 = (mask.mMask & 0x8) ? raw[3] & t2 : t2;
            return t3;
        }
        // HBOR
        UME_FORCE_INLINE uint64_t hbor() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3];
        }
        // MHBOR
        UME_FORCE_INLINE uint64_t hbor(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] : 0;
            uint64_t t1 = (mask.mMask & 0x2) ? raw[1] | t0 : t0;
            uint64_t t2 = (mask.mMask & 0x4) ? raw[2] | t1 : t1;
            uint64_t t3 = (mask.mMask & 0x8) ? raw[3] | t2 : t2;
            return t3;
        }
        // HBORS
        UME_FORCE_INLINE uint64_t hbor(uint64_t b) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3] | b;
        }
        // MHBORS
        UME_FORCE_INLINE uint64_t hbor(SIMDVecMask<4> const & mask, uint64_t b) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] | b : b;
            uint64_t t1 = (mask.mMask & 0x2) ? raw[1] | t0 : t0;
            uint64_t t2 = (mask.mMask & 0x4) ? raw[2] | t1 : t1;
            uint64_t t3 = (mask.mMask & 0x8) ? raw[3] | t2 : t2;
            return t3;
        }
        // HBXOR
        UME_FORCE_INLINE uint64_t hbxor() const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXOR
        UME_FORCE_INLINE uint64_t hbxor(SIMDVecMask<4> const & mask) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] : 0;
            uint64_t t1 = (mask.mMask & 0x2) ? raw[1] ^ t0 : t0;
            uint64_t t2 = (mask.mMask & 0x4) ? raw[2] ^ t1 : t1;
            uint64_t t3 = (mask.mMask & 0x8) ? raw[3] ^ t2 : t2;
            return t3;
        }
        // HBXORS
        UME_FORCE_INLINE uint64_t hbxor(uint64_t b) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE uint64_t hbxor(SIMDVecMask<4> const & mask, uint64_t b) const {
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint64_t t0 = (mask.mMask & 0x1) ? raw[0] ^ b : b;
            uint64_t t1 = (mask.mMask & 0x2) ? raw[1] ^ t0 : t0;
            uint64_t t2 = (mask.mMask & 0x4) ? raw[2] ^ t1 : t1;
            uint64_t t3 = (mask.mMask & 0x8) ? raw[3] ^ t2 : t2;
            return t3;
        }

        // GATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(uint64_t const * baseAddr, uint64_t stride) {
#if defined (__AVX512DQ__)
            __m256i t0 = SET1_EPI64(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
            mVec = _mm256_i64gather_epi64((int64_t const*)baseAddr, t2, 8);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(SIMDVecMask<4> const & mask, uint64_t const * baseAddr, uint64_t stride) {
#if defined(__AVX512DQ__)
            __m256i t0 = SET1_EPI64(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
            __m256i t3 = _mm256_i64gather_epi64((int64_t const*)baseAddr, t2, 8);
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_epi64(mVec, mask.mMask, t3);
#else
            mVec = _mm512_castsi512_si256(
                    _mm512_mask_mov_epi64(
                        _mm512_castsi256_si512(mVec),
                        mask.mMask,
                        _mm512_castsi256_si512(t3)));
#endif
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint64_t const * baseAddr, uint64_t const * indices) {
            __m256i t0 =_mm256_loadu_si256((__m256i *)indices);
            mVec = _mm256_i64gather_epi64((int64_t const*)baseAddr, t0, 8);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint64_t const * baseAddr, uint64_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i *)indices);
            __m256i t1 = _mm256_i64gather_epi64((int64_t const*)baseAddr, t0, 8);
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_epi64(mVec, mask.mMask, t1);
#else
            mVec = _mm512_castsi512_si256(
                    _mm512_mask_mov_epi64(
                        _mm512_castsi256_si512(mVec),
                        mask.mMask,
                        _mm512_castsi256_si512(t1)));
#endif
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint64_t const * baseAddr, SIMDVec_u const & indices) {
            mVec = _mm256_i64gather_epi64((int64_t const*)baseAddr, indices.mVec, 8);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint64_t const * baseAddr, SIMDVec_u const & indices) {
            __m256i t0 = _mm256_i64gather_epi64((int64_t const*)baseAddr, indices.mVec, 8);
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_epi64(mVec, mask.mMask, t0);
#else
            mVec = _mm512_castsi512_si256(
                _mm512_mask_mov_epi64(
                    _mm512_castsi256_si512(mVec),
                    mask.mMask,
                    _mm512_castsi256_si512(t0)));
#endif
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE uint64_t* scatteru(uint64_t* baseAddr, uint64_t stride) const {
#if defined(__AVX512DQ__)
            __m256i t0 = SET1_EPI64(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
#if defined(__AVX512VL__)
            _mm256_i64scatter_epi64(baseAddr, t2, mVec, 8);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i64scatter_epi64(baseAddr, 0xF, t3, t4, 8);
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE uint64_t*  scatteru(SIMDVecMask<4> const & mask, uint64_t* baseAddr, uint64_t stride) const {
#if defined(__AVX512DQ__)
            __m256i t0 = SET1_EPI64(stride);
            __m256i t1 = _mm256_setr_epi64x(0, 1, 2, 3);
            __m256i t2 = _mm256_mullo_epi64(t0, t1);
#else
            __m256i t2 = _mm256_setr_epi64x(0, stride, 2*stride, 3*stride);
#endif
#if defined(__AVX512VL__)
            _mm256_mask_i64scatter_epi64(baseAddr, mask.mMask, t2, mVec, 8);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i64scatter_epi64(baseAddr, mask.mMask, t3, t4, 8);
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE uint64_t* scatter(uint64_t* baseAddr, uint64_t* indices) const {
            __m256i t0 = _mm256_loadu_si256((__m256i *)indices);
#if defined(__AVX512VL__)
            _mm256_i64scatter_epi64(baseAddr, t0, mVec, 8);
#else
            _mm512_mask_i64scatter_epi64(
                            baseAddr,
                            0xF,
                            _mm512_castsi256_si512(t0),
                            _mm512_castsi256_si512(mVec),
                            8);
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint64_t* scatter(SIMDVecMask<4> const & mask, uint64_t* baseAddr, uint64_t* indices) const {
            __m256i t0 = _mm256_loadu_si256((__m256i *)indices);
#if defined(__AVX512VL__)
            _mm256_mask_i64scatter_epi64(baseAddr, mask.mMask, t0, mVec, 8);
#else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                mask.mMask,
                _mm512_castsi256_si512(t0),
                _mm512_castsi256_si512(mVec),
                8);
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint64_t* scatter(uint64_t* baseAddr, SIMDVec_u const & indices) const {
#if defined(__AVX512VL__)
            _mm256_i64scatter_epi64(baseAddr, indices.mVec, mVec, 8);
#else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                0xF,
                _mm512_castsi256_si512(indices.mVec),
                _mm512_castsi256_si512(mVec),
                8);
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint64_t* scatter(SIMDVecMask<4> const & mask, uint64_t* baseAddr, SIMDVec_u const & indices) const {
#if defined(__AVX512VL__)
            _mm256_mask_i64scatter_epi64(baseAddr, mask.mMask, indices.mVec, mVec, 8);
#else
            _mm512_mask_i64scatter_epi64(
                baseAddr,
                mask.mMask,
                _mm512_castsi256_si512(indices.mVec),
                _mm512_castsi256_si512(mVec),
                8);
#endif
            return baseAddr;
        }

        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sllv_epi64(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (SIMDVec_u const & b) const {
            return lsh(b);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sllv_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_sllv_epi64(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint64_t b) const {
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = _mm256_sllv_epi64(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (uint64_t b) const {
            return lsh(b);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = _mm256_mask_sllv_epi64(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(b);
            __m512i t2 = _mm512_mask_sllv_epi64(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        //// LSHVA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
        //    mVec[0] = mVec[0] << b.mVec[0];
        //    mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// MLSHVA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b.mVec[0];
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// LSHSA
        //UME_FORCE_INLINE SIMDVec_u & lsha(uint64_t b) {
        //    mVec[0] = mVec[0] << b;
        //    mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        //// MLSHSA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<4> const & mask, uint64_t b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b;
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_srlv_epi64(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (SIMDVec_u const & b) const {
            return rsh(b);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_srlv_epi64(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_srlv_epi64(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint64_t b) const {
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = _mm256_srlv_epi64(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (uint64_t b) const {
            return rsh(b);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<4> const & mask, uint64_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = SET1_EPI64(b);
            __m256i t1 = _mm256_mask_srlv_epi64(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi64(b);
            __m512i t2 = _mm512_mask_srlv_epi64(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        //// RSHVA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
        //    mVec[0] = mVec[0] >> b.mVec[0];
        //    mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// MRSHVA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
        //    if (mask.mMask[0]) mVec[0] = mVec[0] >> b.mVec[0];
        //    if (mask.mMask[1]) mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// RSHSA
        //UME_FORCE_INLINE SIMDVec_u & rsha(uint64_t b) {
        //    mVec[0] = mVec[0] >> b;
        //    mVec[1] = mVec[1] >> b;
        //    return *this;
        //}
        //// MRSHSA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<4> const & mask, uint64_t b) {
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

        // PACK
        SIMDVec_u & pack(SIMDVec_u<uint64_t, 2> const & a, SIMDVec_u<uint64_t, 2> const & b) {
            mVec = _mm256_insertf128_si256(mVec, a.mVec, 0);
            mVec = _mm256_insertf128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        SIMDVec_u & packlo(SIMDVec_u<uint64_t, 2> const & a) {
            mVec = _mm256_insertf128_si256(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        SIMDVec_u & packhi(SIMDVec_u<uint64_t, 2> const & b) {
            mVec = _mm256_insertf128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        void unpack(SIMDVec_u<uint64_t, 2> & a, SIMDVec_u<uint64_t, 2> & b) const {
            a.mVec = _mm256_extractf128_si256(mVec, 0);
            b.mVec = _mm256_extractf128_si256(mVec, 1);
        }
        // UNPACKLO
        SIMDVec_u<uint64_t, 2> unpacklo() const {
            return SIMDVec_u<uint64_t, 2> (_mm256_extractf128_si256(mVec, 0));
        }
        // UNPACKHI
        SIMDVec_u<uint64_t, 2> unpackhi() const {
            return SIMDVec_u<uint64_t, 2> (_mm256_extractf128_si256(mVec, 1));
        }

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 4>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 4>() const;
    };

#undef SET1_EPI64
#undef EXPAND_CALL_BINARY
#undef EXPAND_CALL_BINARY_MASK
#undef EXPAND_CALL_BINARY_SCALAR_MASK

}
}

#endif
