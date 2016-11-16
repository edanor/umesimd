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

#ifndef UME_SIMD_VEC_UINT32_8_H_
#define UME_SIMD_VEC_UINT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 8> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 8>,
            uint32_t,
            8,
            SIMDVecMask<8>,
            SIMDSwizzle<8>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 8>,
            SIMDVec_u<uint32_t, 4 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_f<float, 8>;

        friend class SIMDVec_u<uint32_t, 16>;
    private:
        __m256i mVec;

        UME_FORCE_INLINE explicit SIMDVec_u(__m256i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_u(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint32_t const *p) { load(p); }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
                         uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }
        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_u &>(*this));
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
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mov_epi32(mVec, mask.mMask, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec = _mm256_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_mov_epi32(mVec, mask.mMask, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<8> const & mask, uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_loadu_epi32(mVec, mask.mMask, p);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi32(t1, (__mmask16) mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm256_load_si256((__m256i*)p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<8> const & mask, uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_load_epi32(mVec, mask.mMask, p);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi32(t1, (__mmask16) mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t * store(uint32_t * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_storeu_epi32(p, 0xFF, mVec);
#else
            _mm256_storeu_si256((__m256i*) p, mVec);
#endif
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t * store(SIMDVecMask<8> const & mask, uint32_t * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_storeu_epi32(p, mask.mMask, mVec);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi32(t1, mask.mMask, t2);
            __m256i t4 = _mm512_castsi512_si256(t3);
            _mm256_storeu_si256((__m256i*)p, t4);
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t * storea(uint32_t * addrAligned) const {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t * storea(SIMDVecMask<8> const & mask, uint32_t * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_store_epi32(p, mask.mMask, mVec);
#else
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi32(t1, mask.mMask, t2);
            __m256i t4 = _mm512_castsi512_si256(t3);
            _mm256_store_si256((__m256i*)p, t4);
#endif
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
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
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_add_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_add_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm256_add_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_add_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_add_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
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
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<8> const & mask) {
            __m256i t0 = mVec;
#if defined(__AVX512VL__)
            __m256i t1 = _mm256_set1_epi32(1);
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t1);
#else
            __m512i t1 = _mm512_set1_epi32(1);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_add_epi32(t2, mask.mMask, t2, t1);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_mask_add_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm256_sub_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            mVec = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
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
            __m256i t0 = _mm256_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t2, mask.mMask, t2, t1);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sub_epi32(t0, mask.mMask, t0, mVec);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm256_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t2, mask.mMask, t2, t1);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u subfroma(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_sub_epi32(t0, mask.mMask, t0, mVec);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<8> const & mask) {
            __m256i t0 = mVec;
#if defined(__AVX512VL__)
            __m256i t1 = _mm256_set1_epi32(1);
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t1);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(1);
            __m512i t3 = _mm512_mask_sub_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_sub_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_mask_sub_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_mullo_epi32(t2, mask.mMask, t2, t0);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm256_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, t1); 
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            mVec = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        // DIVS
        UME_FORCE_INLINE SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        // DIVVA
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        UME_FORCE_INLINE SIMDVec_u & operator/= (uint32_t b) {
            return diva(b);
        }
        // MDIVSA
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpeq_epu32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm256_cmpneq_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(uint32_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpneq_epu32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpgt_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpgt_epu32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm256_cmplt_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(uint32_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmplt_epu32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm256_cmpge_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(uint32_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpge_epu32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm256_cmple_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmple_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple(uint32_t b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmple_epu32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmple_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            return (m0 == 0xFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            return (m0 == 0xFF);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
#if defined(__AVX512VL__) && defined(__AVX512CD__)
            __m256i t0 = _mm256_conflict_epi32(mVec);
            __mmask8 t1 = _mm256_cmpeq_epu32_mask(t0, _mm256_setzero_si256());
            return (t1 == 0xFF);
#else
        alignas(32) uint32_t raw[8];
        _mm256_store_si256((__m256i*)raw, mVec);
        for (int i = 0; i < 7; i++) {
            for (int j = i + 1; j < 8; j++) {
                if (raw[i] == raw[j]) return false;
            }
        }
        return true;
#endif
        }
        // HADD
        UME_FORCE_INLINE uint32_t hadd() const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3] + raw[4] + raw[5] + raw[6] + raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_reduce_add_epi32(t0);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<8> const & mask) const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 0;
            uint32_t t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 0;
            uint32_t t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 0;
            uint32_t t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 0;
            uint32_t t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 0;
            uint32_t t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 0;
            uint32_t t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 0;
            uint32_t t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 0;
            return t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3] + raw[4] + raw[5] + raw[6] + raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_reduce_add_epi32(t0);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 0;
            uint32_t t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 0;
            uint32_t t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 0;
            uint32_t t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 0;
            uint32_t t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 0;
            uint32_t t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 0;
            uint32_t t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 0;
            uint32_t t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 0;
            return b + t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3] * raw[4] * raw[5] * raw[6] * raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_mul_epi32(0xFF, t0);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<8> const & mask) const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 1;
            uint32_t t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 1;
            uint32_t t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 1;
            uint32_t t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 1;
            uint32_t t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 1;
            uint32_t t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 1;
            uint32_t t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 1;
            uint32_t t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 1;
            return t0 * t1 * t2 * t3 * t4 * t5 * t6 * t7;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3] * raw[4] * raw[5] * raw[6] * raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(0xFF, t0);
            return retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : 1;
            uint32_t t1 = ((mask.mMask & 0x02) != 0) ? raw[1] : 1;
            uint32_t t2 = ((mask.mMask & 0x04) != 0) ? raw[2] : 1;
            uint32_t t3 = ((mask.mMask & 0x08) != 0) ? raw[3] : 1;
            uint32_t t4 = ((mask.mMask & 0x10) != 0) ? raw[4] : 1;
            uint32_t t5 = ((mask.mMask & 0x20) != 0) ? raw[5] : 1;
            uint32_t t6 = ((mask.mMask & 0x40) != 0) ? raw[6] : 1;
            uint32_t t7 = ((mask.mMask & 0x80) != 0) ? raw[7] : 1;
            return b * t0 * t1 * t2 * t3 * t4 * t5 * t6 * t7;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
#endif
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm256_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_castsi256_si512(c.mVec);
            __m512i t4 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, t2);
            __m512i t5 = _mm512_mask_add_epi32(t4, mask.mMask, t4, t3);
            __m256i t1 = _mm512_castsi512_si256(t5);
#endif
            return SIMDVec_u(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm256_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_castsi256_si512(c.mVec);
            __m512i t4 = _mm512_mask_mullo_epi32(t0, mask.mMask, t0, t2);
            __m512i t5 = _mm512_mask_sub_epi32(t4, mask.mMask, t4, t3);
            __m256i t1 = _mm512_castsi512_si256(t5);
#endif
            return SIMDVec_u(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm256_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_castsi256_si512(c.mVec);
            __m512i t4 = _mm512_mask_add_epi32(t0, mask.mMask, t0, t2);
            __m512i t5 = _mm512_mask_mullo_epi32(t4, mask.mMask, t4, t3);
            __m256i t1 = _mm512_castsi512_si256(t5);
#endif
            return SIMDVec_u(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<8> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm256_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_castsi256_si512(c.mVec);
            __m512i t4 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            __m512i t5 = _mm512_mask_mullo_epi32(t4, mask.mMask, t4, t3);
            __m256i t1 = _mm512_castsi512_si256(t5);
#endif
            return SIMDVec_u(t1);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_max_epu32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_max_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_max_epu32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm256_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_max_epu32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_max_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_max_epu32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_min_epu32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_min_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_min_epu32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm256_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_min_epu32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_min_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_min_epu32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax() const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = raw[0] > raw[1] ? raw[0] : raw[1];
            uint32_t t1 = raw[2] > raw[3] ? raw[2] : raw[3];
            uint32_t t2 = raw[4] > raw[5] ? raw[4] : raw[5];
            uint32_t t3 = raw[6] > raw[7] ? raw[6] : raw[7];
            uint32_t t4 = t0 > t1 ? t0 : t1;
            uint32_t t5 = t2 > t3 ? t2 : t3;
            return t4 > t5 ? t4 : t5;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_max_epu32(0xFF, t0);
            return retval;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : std::numeric_limits<uint32_t>::min();
            uint32_t t1 = (((mask.mMask & 0x02) != 0) && raw[1] > t0) ? raw[1] : t0;
            uint32_t t2 = (((mask.mMask & 0x04) != 0) && raw[2] > t1) ? raw[2] : t1;
            uint32_t t3 = (((mask.mMask & 0x08) != 0) && raw[3] > t2) ? raw[3] : t2;
            uint32_t t4 = (((mask.mMask & 0x10) != 0) && raw[4] > t3) ? raw[4] : t3;
            uint32_t t5 = (((mask.mMask & 0x20) != 0) && raw[5] > t4) ? raw[5] : t4;
            uint32_t t6 = (((mask.mMask & 0x40) != 0) && raw[6] > t5) ? raw[6] : t5;
            uint32_t t7 = (((mask.mMask & 0x80) != 0) && raw[7] > t6) ? raw[7] : t6;
            return t7;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_max_epu32(mask.mMask, t0);
            return retval;
#endif
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE uint32_t hmin() const {
#if defined(__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = raw[0] < raw[1] ? raw[0] : raw[1];
            uint32_t t1 = raw[2] < raw[3] ? raw[2] : raw[3];
            uint32_t t2 = raw[4] < raw[5] ? raw[4] : raw[5];
            uint32_t t3 = raw[6] < raw[7] ? raw[6] : raw[7];
            uint32_t t4 = t0 < t1 ? t0 : t1;
            uint32_t t5 = t2 < t3 ? t2 : t3;
            return t4 < t5 ? t4 : t5;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_min_epu32(0xFF, t0);
            return retval;
#endif
        }
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = ((mask.mMask & 0x01) != 0) ? raw[0] : std::numeric_limits<uint32_t>::max();
            uint32_t t1 = (((mask.mMask & 0x02) != 0) && raw[1] < t0) ? raw[1] : t0;
            uint32_t t2 = (((mask.mMask & 0x04) != 0) && raw[2] < t1) ? raw[2] : t1;
            uint32_t t3 = (((mask.mMask & 0x08) != 0) && raw[3] < t2) ? raw[3] : t2;
            uint32_t t4 = (((mask.mMask & 0x10) != 0) && raw[4] < t3) ? raw[4] : t3;
            uint32_t t5 = (((mask.mMask & 0x20) != 0) && raw[5] < t4) ? raw[5] : t4;
            uint32_t t6 = (((mask.mMask & 0x40) != 0) && raw[6] < t5) ? raw[6] : t5;
            uint32_t t7 = (((mask.mMask & 0x80) != 0) && raw[7] < t6) ? raw[7] : t6;
            return t7;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_min_epu32(mask.mMask, t0);
            return retval;
#endif
        }
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_and_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_and_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
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
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_and_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_and_si256(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_and_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
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
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_or_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_or_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
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
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_or_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_or_si256(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_or_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
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
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_xor_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
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
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_xor_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t2, mask.mMask, t2, t0);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u bnota(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t2, mask.mMask, t2, t0);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3] &
                   raw[4] & raw[5] & raw[6] & raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_and_epi32(0xFF, t0);
            return retval;
#endif
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = 0xFFFFFFFF;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            if (mask.mMask & 0x10) t0 &= raw[4];
            if (mask.mMask & 0x20) t0 &= raw[5];
            if (mask.mMask & 0x40) t0 &= raw[6];
            if (mask.mMask & 0x80) t0 &= raw[7];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b & raw[0] & raw[1] & raw[2] & raw[3] &
                       raw[4] & raw[5] & raw[6] & raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(0xFF, t0);
            return retval;
#endif
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            if (mask.mMask & 0x10) t0 &= raw[4];
            if (mask.mMask & 0x20) t0 &= raw[5];
            if (mask.mMask & 0x40) t0 &= raw[6];
            if (mask.mMask & 0x80) t0 &= raw[7];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3] |
                   raw[4] | raw[5] | raw[6] | raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_or_epi32(0xFF, t0);
            return retval;
#endif
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<8> const & mask) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = 0;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            if (mask.mMask & 0x10) t0 |= raw[4];
            if (mask.mMask & 0x20) t0 |= raw[5];
            if (mask.mMask & 0x40) t0 |= raw[6];
            if (mask.mMask & 0x80) t0 |= raw[7];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b | raw[0] | raw[1] | raw[2] | raw[3] |
                       raw[4] | raw[5] | raw[6] | raw[7];
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(0xFF, t0);
            return retval;
#endif
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined (__GNUG__)
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            if (mask.mMask & 0x10) t0 |= raw[4];
            if (mask.mMask & 0x20) t0 |= raw[5];
            if (mask.mMask & 0x40) t0 |= raw[6];
            if (mask.mMask & 0x80) t0 |= raw[7];
            return t0;
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<8> const & mask) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = 0;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            if (mask.mMask & 0x10) t0 ^= raw[4];
            if (mask.mMask & 0x20) t0 ^= raw[5];
            if (mask.mMask & 0x40) t0 ^= raw[6];
            if (mask.mMask & 0x80) t0 ^= raw[7];
            return t0;
        }
        // HBXORS
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<8> const & mask, uint32_t b) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            if (mask.mMask & 0x10) t0 ^= raw[4];
            if (mask.mMask & 0x20) t0 ^= raw[5];
            if (mask.mMask & 0x40) t0 ^= raw[6];
            if (mask.mMask & 0x80) t0 ^= raw[7];
            return t0;
        }
        // GATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(uint32_t const * baseAddr, uint32_t stride) {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
            mVec = _mm256_i32gather_epi32((const int *)baseAddr, t2, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(SIMDVecMask<8> const & mask, uint32_t const * baseAddr, uint32_t stride) {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i32gather_epi32(mVec, mask.mMask, t2, (const int *)baseAddr, 4);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            __m512i t5 = _mm512_mask_i32gather_epi32(t4, mask.mMask, t3, (const int *)baseAddr, 4);
            mVec = _mm512_castsi512_si256(t5);
#endif
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<8> const & mask, uint32_t const * baseAddr, uint32_t const * indices) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec = _mm256_mmask_i32gather_epi32(mVec, mask.mMask, t0, (const int *)baseAddr, 4);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            __m512i t2 = _mm512_castsi256_si512(t1);
            __m512i t3 = _mm512_castsi256_si512(mVec);
            __m512i t4 = _mm512_mask_mov_epi32(t3, mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t4);
#endif
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<8> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i32gather_epi32(mVec, mask.mMask, indices.mVec, (const int *)baseAddr, 4);
#else
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec, 4);
            __m512i t2 = _mm512_castsi256_si512(t1);
            __m512i t3 = _mm512_castsi256_si512(mVec);
            __m512i t4 = _mm512_mask_mov_epi32(t3, mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t4);
#endif
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE uint32_t* scatteru(uint32_t* baseAddr, uint32_t stride) const {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm256_i32scatter_epi32((int *)baseAddr, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, 0xFF, t3, t4, 4);
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE uint32_t*  scatteru(SIMDVecMask<8> const & mask, uint32_t* baseAddr, uint32_t stride) const {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm256_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t3, t4, 4);
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            __m256i t0 = _mm256_loadu_si256((__m256i *) indices);
#if defined(__AVX512VL__)
            _mm256_i32scatter_epi32((int *)baseAddr, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, 0xFF, t1, t2, 4);
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<8> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_loadu_epi32(_mm256_set1_epi32(0), mask.mMask, (__m256i *) indices);
            _mm256_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t0, mVec, 4);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*) indices);
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t1, t2, 4);
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
#if defined(__AVX512VL__)
            _mm256_i32scatter_epi32((int *)baseAddr, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, 0xFF, t1, t0, 4);
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<8> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
#if defined(__AVX512VL__)
            _mm256_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask & 0xFF, t1, t0, 4);
#endif
            return baseAddr;
        }
        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sllv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (SIMDVec_u const & b) const {
            return lsh(b);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sllv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_sllv_epi32(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sllv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (uint32_t b) const {
            return lsh(b);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sllv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_sllv_epi32(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_srlv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (SIMDVec_u const & b) const {
            return rsh(b);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_srlv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_srlv_epi32(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_srlv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (uint32_t b) const {
            return rsh(b);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_srlv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_srlv_epi32(t0, mask.mMask, t0, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_u(t3);
#endif
        }
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
        // ROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_rolv_epi32(mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_rolv_epi32(t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_rolv_epi32(t1, __mmask16(mask.mMask), t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_u rol(uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_rolv_epi32(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rolv_epi32(t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
#else            
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rolv_epi32(t0, __mmask16(mask.mMask), t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_rolv_epi32(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_rolv_epi32(t0, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // MROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_rolv_epi32(t0, __mmask16(mask.mMask), t0, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // ROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(uint32_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_rolv_epi32(mVec, _mm256_set1_epi32(b));
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rolv_epi32(t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rolv_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // RORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_rorv_epi32(mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_rorv_epi32(t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_rorv_epi32(t1, __mmask16(mask.mMask), t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t0);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_u ror(uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_rorv_epi32(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rorv_epi32(t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<8> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
#else            
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rorv_epi32(t0, __mmask16(mask.mMask), t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_u(t1);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_rorv_epi32(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_rorv_epi32(t0, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // MRORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_rorv_epi32(t0, __mmask16(mask.mMask), t0, t1);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // RORSA
        UME_FORCE_INLINE SIMDVec_u & rora(uint32_t b) {
#if defined(__AVX512VL__)
            mVec = _mm256_rorv_epi32(mVec, _mm256_set1_epi32(b));
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rorv_epi32(t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MRORSA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<8> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rorv_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_u & pack(SIMDVec_u<uint32_t, 4> const & a, SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm256_inserti128_si256(mVec, a.mVec, 0);
            mVec = _mm256_inserti128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_u & packlo(SIMDVec_u<uint32_t, 4> const & a) {
            mVec = _mm256_inserti128_si256(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_u & packhi(SIMDVec_u<uint32_t, 4> const & b) {
            mVec = _mm256_inserti128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_u<uint32_t, 4> & a, SIMDVec_u<uint32_t, 4> & b) const {
            a.mVec = _mm256_extracti128_si256(mVec, 0);
            b.mVec = _mm256_extracti128_si256(mVec, 1);
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 4> unpacklo() const {
            __m128i t0 = _mm256_extracti128_si256(mVec, 0);
            return SIMDVec_u<uint32_t, 4>(t0);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 4> unpackhi() const {
            __m128i t0 = _mm256_extracti128_si256(mVec, 1);
            return SIMDVec_u<uint32_t, 4>(t0);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 8>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 8>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 8>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 8>() const;

    };

}
}

#endif
