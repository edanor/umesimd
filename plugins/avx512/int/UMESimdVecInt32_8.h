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

#ifndef UME_SIMD_VEC_INT32_8_H_
#define UME_SIMD_VEC_INT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 8> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 8>,
            SIMDVec_u<uint32_t, 8>,
            int32_t,
            8,
            uint32_t,
            SIMDVecMask<8>,
            SIMDSwizzle<8>> ,
        public SIMDVecPackableInterface<
           SIMDVec_i<int32_t, 8>,
           SIMDVec_i<int32_t, 4 >>
    {
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;

        friend class SIMDVec_i<int32_t, 16>;
    private:
        __m256i mVec;

        UME_FORCE_INLINE explicit SIMDVec_i(__m256i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_i(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, int32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const * p) { load(p); }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }
        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i *)raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>> operator() (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
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
#if defined(__AVX512VL_)
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
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            mVec = _mm256_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<8> const & mask, int32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_loadu_epi32(mVec, mask.mMask, p);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi32(t1, (__mmask16)mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm256_load_si256((__m256i*)p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<8> const & mask, int32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_load_epi32(mVec, mask.mMask, p);
#else
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi32(t1, (__mmask16)mask.mMask, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t * store(int32_t * p) const {
#if defined(__AVX512VL__)
            _mm256_mask_storeu_epi32(p, 0xFF, mVec);
#else
            _mm256_storeu_si256((__m256i*) p, mVec);
#endif
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t * store(SIMDVecMask<8> const & mask, int32_t * p) const {
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
        UME_FORCE_INLINE int32_t * storea(int32_t * addrAligned) {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        UME_FORCE_INLINE int32_t * storea(SIMDVecMask<8> const & mask, int32_t * p) const {
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
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_i(t3);
#endif
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_i(t1);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            __m256i t3 = _mm512_castsi512_si256(t2);
            return SIMDVec_i(t3);
#endif
        }
        // SWIZZLE
        // SWIZZLEA
        // SORTA
        // SORTD

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_add_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_add_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm256_add_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & adda(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i postinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<8> const & mask) {
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
            return SIMDVec_i(t0);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<8> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm256_sub_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & suba(int32_t b) {
            mVec = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t2, mask.mMask, t2, t1);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sub_epi32(t0, mask.mMask, t0, mVec);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_sub_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm256_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & subfroma(int32_t b) {
            mVec = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i subfroma(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<8> const & mask) {
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
            return SIMDVec_i(t0);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_sub_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<8> const & mask) {
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
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_mullo_epi32(t2, mask.mMask, t2, t0);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = _mm256_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & mula(int32_t b) {
            mVec = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<8> const & mask, int32_t b) {
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
        // MDIVV
        // DIVS
        // MDIVS
        // DIVVA
        // MDIVVA
        // DIVSA
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
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<8> cmpeq(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpeq_epi32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpneq_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!=(SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<8> cmpne(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpneq_epi32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator!=(int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpgt_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpgt_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<8> cmpgt(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpgt_epi32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpgt_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmplt_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmplt_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<8> cmplt(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmplt_epi32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmplt_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpge_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpge_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<8> cmpge(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpge_epi32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmpge_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<8> cmple(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmple_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmple_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<8> cmple(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmple_epi32_mask(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __mmask16 m1 = _mm512_cmple_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            SIMDVecMask<8> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<8> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epi32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            return (m0 == 0xFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm256_cmpeq_epi32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(t0);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x00FF;
#endif
            return (m0 == 0xFF);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            __m256i t0 = _mm256_conflict_epi32(mVec);
            __mmask8 t1 = _mm256_cmpeq_epi32_mask(t0, _mm256_set1_epi32(1));
            return (t1 == 0x00);
        }
        // HADD
        UME_FORCE_INLINE int32_t hadd() const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_reduce_add_epi32(t0);
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            int32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE int32_t hadd(int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_reduce_add_epi32(t0);
            return retval + b;
        }
        // MHADDS
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<8> const & mask, int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            int32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval + b;
        }
        // HMUL
        UME_FORCE_INLINE int32_t hmul() const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_mul_epi32(0xFF, t0);
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            int32_t retval = _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE int32_t hmul(int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(0xFF, t0);
            return retval;
        }
        // MHMULS
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<8> const & mask, int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __mmask16 t1 = 0x00FF & __mmask16(mask.mMask);
            int32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
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
            return SIMDVec_i(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
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
            return SIMDVec_i(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
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
            return SIMDVec_i(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
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
            return SIMDVec_i(t1);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_max_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_max_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_max_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = _mm256_max_epi32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_max_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_max_epi32(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<8> const & mask, int32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_max_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_min_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_min_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_min_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = _mm256_min_epi32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_min_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_min_epi32(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<8> const & mask, int32_t b) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_min_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si256(t3);
#endif
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE int32_t hmax() const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_max_epi32(0xFF, t0);
            return retval;
        }       
        // MHMAX
        UME_FORCE_INLINE int32_t hmax(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_max_epi32(mask.mMask, t0);
            return retval;
        }       
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE int32_t hmin() const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_min_epi32(0xFF, t0);
            return retval;
        }       
        // MHMIN
        UME_FORCE_INLINE int32_t hmin(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_min_epi32(mask.mMask, t0);
            return retval;
        }       
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_and_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_and_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm256_and_si256(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & banda(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_and_si256(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (int32_t b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_or_si256(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_or_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_or_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm256_or_si256(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & bora(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_or_si256(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t1, mask.mMask, t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_xor_epi32(t0, mask.mMask, t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm256_xor_si256(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & bxora(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator! () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t2, mask.mMask, t2, t0);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i bnota(SIMDVecMask<8> const & mask) {
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
        UME_FORCE_INLINE int32_t hband() const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_and_epi32(0xFF, t0);
            return retval;
        }
        // MHBAND
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
        }
        // HBANDS
        UME_FORCE_INLINE int32_t hband(int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(0xFF, t0);
            return retval;
        }
        // MHBANDS
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<8> const & mask, int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
        }
        // HBOR
        UME_FORCE_INLINE int32_t hbor() const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_or_epi32(0xFF, t0);
            return retval;
        }
        // MHBOR
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<8> const & mask) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
        }
        // HBORS
        UME_FORCE_INLINE int32_t hbor(int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(0xFF, t0);
            return retval;
        }
        // MHBORS
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m512i t0 = _mm512_castsi256_si512(mVec);
            int32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
        }
        // HBXOR
        UME_FORCE_INLINE int32_t hbxor() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXOR
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<8> const & mask) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 0;
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
        UME_FORCE_INLINE int32_t hbxor(int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXORS
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<8> const & mask, int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = b;
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
        UME_FORCE_INLINE SIMDVec_i & gatheru(int32_t const * baseAddr, uint32_t stride) {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
            mVec = _mm256_i32gather_epi32((const int *)baseAddr, t2, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(SIMDVecMask<8> const & mask, int32_t const * baseAddr, uint32_t stride) {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i32gather_epi32(mVec, mask.mMask, t2, baseAddr, 4);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            __m512i t5 = _mm512_mask_i32gather_epi32(t4, mask.mMask, t3, baseAddr, 4);
            mVec = _mm512_castsi512_si256(t5);
#endif
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, uint32_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec = _mm256_i32gather_epi32((const int *) baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<8> const & mask, int32_t const * baseAddr, uint32_t const * indices) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec = _mm256_mmask_i32gather_epi32(mVec, mask.mMask, t0, baseAddr, 4);
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
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, SIMDVec_i const & indices) {
            mVec = _mm256_i32gather_epi32((const int *) baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<8> const & mask, int32_t const * baseAddr, SIMDVec_i const & indices) {
#if defined(__AVX512VL__)
            mVec = _mm256_mmask_i32gather_epi32(mVec, mask.mMask, indices.mVec, baseAddr, 4);
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
        UME_FORCE_INLINE int32_t* scatteru(int32_t* baseAddr, uint32_t stride) const {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm256_i32scatter_epi32(baseAddr, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, 0xFF, t3, t4, 4);
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE int32_t*  scatteru(SIMDVecMask<8> const & mask, int32_t* baseAddr, uint32_t stride) const {
            __m256i t0 = _mm256_set1_epi32(stride);
            __m256i t1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
            __m256i t2 = _mm256_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm256_mask_i32scatter_epi32(baseAddr, mask.mMask, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi256_si512(t2);
            __m512i t4 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t3, t4, 4);
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i *) indices);
#if defined(__AVX512VL__)
            _mm256_i32scatter_epi32(baseAddr, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, 0xFF, t1, t2, 4);
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<8> const & mask, int32_t* baseAddr, uint32_t* indices) {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_load_si256((__m256i*) indices);
            _mm256_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 4);
#else
            __m256i t0 = _mm256_load_si256((__m256i*) indices);
            __m512i t1 = _mm512_castsi256_si512(t0);
            __m512i t2 = _mm512_castsi256_si512(mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask, t1, t2, 4);
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) {
#if defined(__AVX512VL__)
            _mm256_i32scatter_epi32(baseAddr, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, 0xFF, t1, t0, 4);
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<8> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
#if defined(__AVX512VL__)
            _mm256_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_castsi256_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask & 0xFF, t1, t0, 4);
#endif
            return baseAddr;
        }
        // LSHV
        // MLSHV
        // LSHS
        // MLSHS
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA
        // RSHV
        // MRSHV
        // RSHS
        // MRSHS
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
        // ROLV
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_rolv_epi32(mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_rolv_epi32(t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_rolv_epi32(t1, __mmask16(mask.mMask), t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_i rol(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_rolv_epi32(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rolv_epi32(t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
#else            
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rolv_epi32(t0, __mmask16(mask.mMask), t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & rola(int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVecMask<8> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_rorv_epi32(mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_rorv_epi32(t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_castsi256_si512(b.mVec);
            __m512i t3 = _mm512_mask_rorv_epi32(t1, __mmask16(mask.mMask), t1, t2);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_i ror(int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_rorv_epi32(mVec, t0);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rorv_epi32(t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVecMask<8> const & mask, int32_t b) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
#else            
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rorv_epi32(t0, __mmask16(mask.mMask), t0, t2);
            __m256i t1 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t1);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
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
        UME_FORCE_INLINE SIMDVec_i & rora(int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVecMask<8> const & mask, int32_t b) {
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
        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            __m256i t0 = _mm256_sub_epi32(_mm256_setzero_si256(), mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_sub_epi32(mVec, mask.mMask, _mm256_set1_epi32(0), mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_setzero_epi32();
            __m512i t3 = _mm512_mask_sub_epi32(t1, mask.mMask, t2, t1);
            __m256i t0 = _mm512_castsi512_si256(t3);
#endif
            return SIMDVec_i(t0);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec = _mm256_sub_epi32(_mm256_setzero_si256(), mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, _mm256_set1_epi32(0), mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_setzero_epi32();
            __m512i t2 = _mm512_mask_sub_epi32(t0, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si256(t2);
#endif
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            __m256i t0 = _mm256_abs_epi32(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<8> const & mask) const {
#if defined(__AVX512VL__)
            __m256i t0 = _mm256_mask_abs_epi32(mVec, mask.mMask, mVec);
#else
            __m512i t1 = _mm512_castsi256_si512(mVec);
            __m512i t2 = _mm512_mask_abs_epi32(t1, mask.mMask, t1);
            __m256i t0 = _mm512_castsi512_si256(t2);
#endif
            return SIMDVec_i(t0);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            mVec = _mm256_abs_epi32(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<8> const & mask) {
#if defined(__AVX512VL__)
            mVec = _mm256_mask_abs_epi32(mVec, mask.mMask, mVec);
#else
            __m512i t0 = _mm512_castsi256_si512(mVec);
            __m512i t1 = _mm512_mask_abs_epi32(t0, mask.mMask, t0);
            mVec = _mm512_castsi512_si256(t1);
#endif
            return *this;
        }
        // PACK
        UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int32_t, 4> const & a, SIMDVec_i<int32_t, 4> const & b) {
            mVec = _mm256_inserti128_si256(mVec, a.mVec, 0);
            mVec = _mm256_inserti128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int32_t, 4> const & a) {
            mVec = _mm256_inserti128_si256(mVec, a.mVec, 0);
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_i & packhi(SIMDVec_i<int32_t, 4> const & b) {
            mVec = _mm256_inserti128_si256(mVec, b.mVec, 1);
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_i<int32_t, 4> & a, SIMDVec_i<int32_t, 4> & b) const {
            a.mVec = _mm256_extracti128_si256(mVec, 0);
            b.mVec = _mm256_extracti128_si256(mVec, 1);
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_i<int32_t, 4> unpacklo() const {
            __m128i t0 = _mm256_extracti128_si256(mVec, 0);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_i<int32_t, 4> unpackhi() const {
            __m128i t0 = _mm256_extracti128_si256(mVec, 1);
            return SIMDVec_i<int32_t, 4>(t0);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 8>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 8>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 8> () const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 8>() const;

    };

}
}

#endif
