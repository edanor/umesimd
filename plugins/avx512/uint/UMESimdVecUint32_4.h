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

#ifndef UME_SIMD_VEC_UINT32_4_H_
#define UME_SIMD_VEC_UINT32_4_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 4> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 4>,
            uint32_t,
            4,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_u<uint32_t, 2 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_f<float, 4>;

        friend class SIMDVec_u<uint32_t, 8>;

    private:
        __m128i mVec;

        UME_FORCE_INLINE explicit SIMDVec_u(__m128i & x) { mVec = x; }
        UME_FORCE_INLINE explicit SIMDVec_u(const __m128i & x) { mVec = x; }

#if !defined(__AVX512VL__)
        // This function converts between mask representation and integer vector register representation
        // for 4x32b vectors. This is necessary for platforms that don't support AVX512VL instructions.
        UME_FORCE_INLINE __m128i mask8_to_m128i(__mmask8 mask) const {
            __m128i vmask = _mm_set1_epi8(mask);
            __m128i bitmask = _mm_set_epi32(0xF7F7F7F7, 0xFBFBFBFB, 0xFDFDFDFD, 0xFEFEFEFE);
            vmask = _mm_or_si128(vmask, bitmask);
            return _mm_cmpeq_epi32(vmask, _mm_set1_epi32(0xFFFFFFFF));
        }
        // This function converts between integer vector register representation and mask representation
        // for 4x32b vectors. This is necessary for platforms that don't support AVX512VL instructions.
        UME_FORCE_INLINE __mmask8 m128i_to_mask8(__m128i vec) const {
            __m128i shuffle = _mm_setr_epi8(
                0x03, 0x07, 0x0B, 0x0F, // Select first byte of every dword
                0x80, 0x80, 0x80, 0x80,
                0x80, 0x80, 0x80, 0x80,
                0x80, 0x80, 0x80, 0x80);

            __m128i vmask = _mm_shuffle_epi8(vec, shuffle);
            int bitmask = _mm_movemask_epi8(vmask);
            return __mmask8(bitmask & 0xF);
        }
#endif

    public:

        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec = _mm_set1_epi32(i);
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
        UME_FORCE_INLINE explicit SIMDVec_u(uint32_t const *p) { this->load(p); };
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
        {
            mVec = _mm_set_epi32(i3, i2, i1, i0);
        }
        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            return raw[index];
        }
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator=(SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_mov_epi32(mVec, mask.mMask, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec = _mm_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_mov_epi32(mVec, mask.mMask, t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_epi32(mVec, 0xFF, p);
#else
            mVec = _mm_loadu_si128((__m128i*)p);
#endif
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<4> const & mask, uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_epi32(mVec, mask.mMask, p);
#else
            __m128i t0 = _mm_loadu_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi32(t1, mask.mMask, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm_load_si128((__m128i*)p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<4> const & mask, uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, p);
#else
            __m128i t0 = _mm_load_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __m512i t3 = _mm512_mask_mov_epi32(t1, mask.mMask, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t * store(uint32_t * p) const {
            _mm_storeu_si128((__m128i*) p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t * store(SIMDVecMask<4> const & mask, uint32_t * p) const {
#if defined(__AVX512VL__)
            _mm_mask_storeu_epi32(p, mask.mMask, mVec);
#else
            __m128i t0 = _mm_loadu_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi32(t1, mask.mMask, t2);
            __m128i t4 = _mm512_castsi512_si128(t3);
            _mm_storeu_si128((__m128i*)p, t4);
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t * storea(uint32_t * p) const {
            _mm_store_si128((__m128i *)p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t * storea(SIMDVecMask<4> const & mask, uint32_t * p) const {
#if defined(__AVX512VL__)
            _mm_mask_store_epi32(p, mask.mMask, mVec);
#else
            __m128i t0 = _mm_load_si128((__m128i*)p);
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_mov_epi32(t1, mask.mMask, t2);
            __m128i t4 = _mm512_castsi512_si128(t3);
            _mm_store_si128((__m128i*)p, t4);
#endif
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_u(t3);
#endif
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_mov_epi32(t0, mask.mMask, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_u(t3);
#endif
        }
        // SWIZZLE
        // SWIZZLEA
        // SORTA
        // SORTD
        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t1 = _mm_add_epi32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t0 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t0);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            __m128i t0 = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __m128i t1 = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_add_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t1 = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm_add_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
#endif
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            mVec = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t1 = _mm_add_epi32(mVec, t0);
                __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t1, m0);
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
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_add_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_add_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
#if defined(__AVX512VL__)
            mVec = _mm_mask_add_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t1 = _mm_add_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t1 = _mm_sub_epi32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t0 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t0);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            __m128i t0 = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __m128i t1 = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_sub_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t1 = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm_sub_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
#endif
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            mVec = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t1 = _mm_sub_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t1, m0);
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
            __m128i t0 = _mm_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m128i t1 = _mm_sub_epi32(b.mVec, mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t0 = _mm_blendv_epi8(b.mVec, t1, m0);
#endif
            return SIMDVec_u(t0);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            __m128i t0 = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __m128i t1 = _mm_mask_sub_epi32(t0, mask.mMask, t0, mVec);
#else
            __m128i t2 = _mm_sub_epi32(t0, mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t1 = _mm_blendv_epi8(t0, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
#else
            __m128i t1 = _mm_sub_epi32(b.mVec, mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(b.mVec, t1, m0);
#endif
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u subfroma(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi32(t0, mask.mMask, t0, mVec);
#else
            __m128i t2 = _mm_sub_epi32(t0, mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(t0, t2, m0);
#endif
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_sub_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_sub_epi32(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<4> const & mask) {
            __m128i t0 = _mm_set1_epi32(1);
#if defined(__AVX512VL__)
            mVec = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_sub_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t1 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t0 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t0);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            __m128i t0 = _mm_mullo_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __m128i t1 = _mm_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_mullo_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t1 = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
#endif
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            mVec = _mm_mullo_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            mVec = _mm_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t1 = _mm_mullo_epi32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t1, m0);
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
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu32_mask(mVec, b.mVec);
#else
            __m128i t0 = _mm_cmpeq_epi32(mVec, b.mVec);
            __mmask8 m0 = m128i_to_mask8(t0);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(uint32_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu32_mask(mVec, _mm_set1_epi32(b)) & 0xF;
#else
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_cmpeq_epi32(mVec, t0);
            __mmask8 m0 = m128i_to_mask8(t1);
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpneq_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpneq_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpgt_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpgt_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmplt_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmplt_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpge_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmpge_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_u const & b) const {
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmple_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmple_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__) && !defined(WA_GCC_INTR_SUPPORT_6_2)
            __mmask8 m0 = _mm_cmple_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmple_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            SIMDVecMask<4> ret_mask;
            ret_mask.mMask = m0;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return (m0 == 0x0F);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu32_mask(mVec, _mm_set1_epi32(b));
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return (m0 == 0x0F);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
#if defined(__AVX512VL__) && defined(__AVX512CD__)
            __m128i t0 = _mm_conflict_epi32(mVec);
            __mmask8 m0 = _mm_cmpeq_epu32_mask(t0, _mm_setzero_si128());
            return (m0 == 0xF);
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            bool t0 = raw[0] != raw[1];
            bool t1 = raw[0] != raw[2];
            bool t2 = raw[0] != raw[3];
            bool t3 = raw[1] != raw[2];
            bool t4 = raw[1] != raw[3];
            bool t5 = raw[2] != raw[3];
            return t0 && t1 && t2 && t3 && t4 && t5;
#endif
        }
        // HADD
        UME_FORCE_INLINE uint32_t hadd() const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_reduce_add_epi32(t0);
            return retval;
#endif
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval;
#endif
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_reduce_add_epi32(t0);
            return retval + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_mul_epi32(0xF, t0);
            return retval;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
#endif
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(0xF, t0);
            return retval;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
#endif
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m128i t1 = _mm_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_add_epi32(t0, c.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            t1 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m128i t1 = _mm_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_sub_epi32(t0, c.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            t1 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m128i t1 = _mm_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            t1 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t1);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m128i t1 = _mm_mask_mullo_epi32(t0, mask.mMask, t0, c.mVec);
#else
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            t1 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t1);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            __m128i t0 = _mm_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t1 = _mm_max_epu32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t0 = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return SIMDVec_u(t0);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __m128i t1 = _mm_mask_max_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t2 = _mm_max_epu32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t1 = _mm_blendv_epi8(mVec, t2, m0);
#endif
            return SIMDVec_u(t1);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_max_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m128i t1 = _mm_max_epu32(mVec, b.mVec);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            mVec = _mm_mask_max_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m128i t1 = _mm_max_epu32(mVec, t0);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t1, m0);
#endif
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            __m128i t0 = _mm_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_mask_min_epu32(t1, __mmask16(mask.mMask), t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_min_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_min_epu32(t2, __mmask16(mask.mMask), t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_min_epu32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_mask_min_epu32(t1, __mmask16(mask.mMask), t1, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_min_epu32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_min_epu32(t2, __mmask16(mask.mMask), t0, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax() const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = (raw[0] > raw[1]) ? raw[0] : raw[1];
            uint32_t t1 = (raw[2] > raw[3]) ? raw[2] : raw[3];
            return t0 > t1 ? t0 : t1;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_max_epu32(0xF, t0);
            return retval;
#endif
        }
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = 0;
            if(((mask.mMask & 0x01) != 0) && (t0 < raw[0])) t0 = raw[0];
            if(((mask.mMask & 0x02) != 0) && (t0 < raw[1])) t0 = raw[1];
            if(((mask.mMask & 0x04) != 0) && (t0 < raw[2])) t0 = raw[2];
            if(((mask.mMask & 0x08) != 0) && (t0 < raw[3])) t0 = raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_max_epu32(mask.mMask, t0);
            return retval;
#endif
        }       
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE uint32_t hmin() const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = (raw[0] < raw[1]) ? raw[0] : raw[1];
            uint32_t t1 = (raw[2] < raw[3]) ? raw[2] : raw[3];
            return t0 < t1 ? t0 : t1;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_min_epu32(0xF, t0);
            return retval;
#endif
        }       
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = std::numeric_limits<uint32_t>::max();
            if(((mask.mMask & 0x01) != 0) && (t0 > raw[0])) t0 = raw[0];
            if(((mask.mMask & 0x02) != 0) && (t0 > raw[1])) t0 = raw[1];
            if(((mask.mMask & 0x04) != 0) && (t0 > raw[2])) t0 = raw[2];
            if(((mask.mMask & 0x08) != 0) && (t0 > raw[3])) t0 = raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_min_epu32(mask.mMask, t0);
            return retval;
#endif
        }       
        // IMIN
        // MIMIN
        
        // REMV
        // MREMV
        // REMS
        // MREMS
        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m128i t0 = _mm_and_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m0 = __mmask16(mask.mMask);
            __m512i t3 = _mm512_mask_and_epi32(t1, m0, t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_and_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_and_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_and_epi32(t0, mask.mMask, t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm_and_si128(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m0 = __mmask16(mask.mMask);
            __m512i t3 = _mm512_mask_and_epi32(t1, m0, t1, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_and_si128(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u operator&= (uint32_t b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_and_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_and_epi32(t0, mask.mMask, t0, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            __m128i t0 = _mm_or_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_mask_or_epi32(t1, mask.mMask, t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_or_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_or_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_or_epi32(t2, mask.mMask, t2, t0);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm_or_si128(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(b.mVec);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_mask_or_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_or_si128(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_or_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_mask_or_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m128i t0 = _mm_xor_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t1, mask.mMask, t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
       return SIMDVec_u(t0);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_xor_epi32(t0, mask.mMask, t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm_xor_si128(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(b.mVec);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_mask_xor_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_xor_si128(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_mask_xor_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator! () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<4> const & mask) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_xor_epi32(t2, mask.mMask, t2, t0);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
//TODO: replace with XOR
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            mVec = _mm_mask_andnot_epi32(mVec, 0xFF, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_andnot_epi32(t1, t0);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u bnota(SIMDVecMask<4> const & mask) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            mVec = _mm_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_mask_andnot_epi32(t1, mask.mMask, t1, t0);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // BANDNOTV
        // MBANDNOTV
        // BANDNOTS
        // MBANDNOTS
        // BANDNOTVA
        // MBANDNOTVA
        // BANDNOTSA
        // MBANDNOTSA
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_and_epi32(0xF, t0);
            return retval;
#endif
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = 0xFFFFFFFF;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b & raw[0] & raw[1] & raw[2] & raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(0xF, t0);
            return retval;
#endif
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_or_epi32(0xF, t0);
            return retval;
#endif
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<4> const & mask) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = 0;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b | raw[0] | raw[1] | raw[2] | raw[3];
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(0xF, t0);
            return retval;
#endif
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined (__GNUG__)
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            return t0;
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
#endif
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<4> const & mask) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = 0;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            return t0;
        }
        // HBXORS
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<4> const & mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            return t0;
        }

        // GATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(uint32_t const * baseAddr, uint32_t stride) {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
            mVec = _mm_i32gather_epi32((const int *)baseAddr, t2, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(SIMDVecMask<4> const & mask, uint32_t const * baseAddr, uint32_t stride) {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            mVec = _mm_mmask_i32gather_epi32(mVec, mask.mMask, t2, (const int *)baseAddr, 4);
#else
            __m512i t3 = _mm512_castsi128_si512(t2);
            __m512i t4 = _mm512_castsi128_si512(mVec);
            __m512i t5 = _mm512_mask_i32gather_epi32(t4, mask.mMask, t3, (const int *)baseAddr, 4);
            mVec = _mm512_castsi512_si128(t5);
#endif
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t const * indices) {
            __m128i t0 = _mm_loadu_si128((__m128i*)indices);
            mVec = _mm_i32gather_epi32((const int *)baseAddr, t0, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t const * baseAddr, uint32_t const * indices) {
            __m128i t0 = _mm_loadu_si128((__m128i*)indices);
#if defined(__AVX512VL__)
            mVec = _mm_mmask_i32gather_epi32(mVec, mask.mMask, t0, (const int *)baseAddr, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            __m512i t3 = _mm512_mask_i32gather_epi32(t2, mask.mMask, t1, (const int *)baseAddr, 4);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec = _mm_i32gather_epi32((const int *)baseAddr, indices.mVec, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
#if defined(__AVX512VL__)
            mVec = _mm_mmask_i32gather_epi32(mVec, mask.mMask, indices.mVec, (const int *)baseAddr, 4);
#else
            __m512i t0 = _mm512_castsi128_si512(indices.mVec);
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_mask_i32gather_epi32(t1, mask.mMask, t0, (const int *)baseAddr, 4);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE uint32_t* scatteru(uint32_t* baseAddr, uint32_t stride) const {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm_i32scatter_epi32((int *)baseAddr, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi128_si512(t2);
            __m512i t4 = _mm512_castsi128_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, 0xF, t3, t4, 4);
#endif
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE uint32_t*  scatteru(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint32_t stride) const {
            __m128i t0 = _mm_set1_epi32(stride);
            __m128i t1 = _mm_setr_epi32(0, 1, 2, 3);
            __m128i t2 = _mm_mullo_epi32(t0, t1);
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t2, mVec, 4);
#else
            __m512i t3 = _mm512_castsi128_si512(t2);
            __m512i t4 = _mm512_castsi128_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t3, t4, 4);
#endif
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) {
            __m128i t0 = _mm_loadu_si128((__m128i *) indices);
#if defined(__AVX512VL__)
            _mm_i32scatter_epi32((int *)baseAddr, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, 0xF, t1, t2, 4);
#endif
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint32_t* indices) {
            __m128i t0 = _mm_loadu_si128((__m128i *) indices);
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, t1, t2, 4);
#endif
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) {
#if defined(__AVX512VL__)
            _mm_i32scatter_epi32((int *)baseAddr, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, 0xF, t1, t0, 4);
#endif
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_epi32((int *)baseAddr, mask.mMask, indices.mVec, mVec, 4);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask & 0xF, t1, t0, 4);
#endif
            return baseAddr;
        }

        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            __m128i t0 = _mm_sllv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (SIMDVec_u const & b) const {
            return lsh(b);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_sllv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_sllv_epi32(t0, mask.mMask & 0xF, t0, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_u(t3);
#endif
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_sllv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (uint32_t b) const {
            return lsh(b);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_sllv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_sllv_epi32(t0, mask.mMask & 0xF, t0, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_u(t3);
#endif
        }
        // LSHVA
        /*
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec = _mm_sll_epi32(mVec, b.mVec);
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_sll_epi32(mVec, t0);
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }*/
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            __m128i t0 = _mm_srlv_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (SIMDVec_u const & b) const {
            return rsh(b);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_srlv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_srlv_epi32(t0, mask.mMask & 0xF, t0, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_u(t3);
#endif
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_srlv_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (uint32_t b) const {
            return rsh(b);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_srlv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_set1_epi32(b);
            __m512i t2 = _mm512_mask_srlv_epi32(t0, mask.mMask & 0xF, t0, t1);
            __m128i t3 = _mm512_castsi512_si128(t2);
            return SIMDVec_u(t3);
#endif
        }
        /*
        // RSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec = _mm_srl_epi32(mVec, b.mVec);
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec = _mm_mask_srl_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_srl_epi32(mVec, t0);
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_srl_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }*/
        // ROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_rolv_epi32(mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_rolv_epi32(t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_mask_rolv_epi32(t1, __mmask16(mask.mMask), t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_u rol(uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_rolv_epi32(mVec, t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rolv_epi32(t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
#else            
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rolv_epi32(t0, __mmask16(mask.mMask), t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_rolv_epi32(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_rolv_epi32(t0, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // MROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_rolv_epi32(t0, __mmask16(mask.mMask), t0, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // ROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(uint32_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_rolv_epi32(mVec, _mm_set1_epi32(b));
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rolv_epi32(t1, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // MROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rolv_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // RORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_rorv_epi32(mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_rorv_epi32(t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(b.mVec);
            __m512i t3 = _mm512_mask_rorv_epi32(t1, __mmask16(mask.mMask), t1, t2);
            __m128i t0 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t0);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_u ror(uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_rorv_epi32(mVec, t0);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rorv_epi32(t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<4> const & mask, uint32_t b) const {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
#else            
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rorv_epi32(t0, __mmask16(mask.mMask), t0, t2);
            __m128i t1 = _mm512_castsi512_si128(t3);
#endif
            return SIMDVec_u(t1);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_rorv_epi32(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_rorv_epi32(t0, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // MRORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __m512i t2 = _mm512_mask_rorv_epi32(t0, __mmask16(mask.mMask), t0, t1);
            mVec = _mm512_castsi512_si128(t2);
#endif
            return *this;
        }
        // RORSA
        UME_FORCE_INLINE SIMDVec_u & rora(uint32_t b) {
#if defined(__AVX512VL__)
            mVec = _mm_rorv_epi32(mVec, _mm_set1_epi32(b));
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_rorv_epi32(t1, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }
        // MRORSA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<4> const & mask, uint32_t b) {
#if defined(__AVX512VL__)
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_set1_epi32(b);
            __m512i t3 = _mm512_mask_rorv_epi32(t1, mask.mMask, t1, t2);
            mVec = _mm512_castsi512_si128(t3);
#endif
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_u & pack(SIMDVec_u<uint32_t, 2> const & a, SIMDVec_u<uint32_t, 2> const & b) {
            alignas(16) uint32_t raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_u & packlo(SIMDVec_u<uint32_t, 2> const & a) {
#if defined(__AVX512VL__)
            alignas(16) uint32_t raw[4] = { a.mVec[0], a.mVec[1], 0, 0};
            mVec = _mm_mask_load_epi32(mVec, 0x3, (__m128i*)raw);
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[0] = a.mVec[0];
            raw[1] = a.mVec[1];
            mVec = _mm_load_si128((__m128i*)raw);
#endif
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_u & packhi(SIMDVec_u<uint32_t, 2> const & b) {
#if defined(__AVX512VL__)
            alignas(16) uint32_t raw[4] = { 0, 0, b.mVec[0], b.mVec[1] };
            mVec = _mm_mask_load_epi32(mVec, 0xC, (__m128i*)raw);
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[2] = b.mVec[0];
            raw[3] = b.mVec[1];
            mVec = _mm_load_si128((__m128i*)raw);
#endif
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_u<uint32_t, 2> & a, SIMDVec_u<uint32_t, 2> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 2> unpacklo() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return SIMDVec_u<uint32_t, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 2> unpackhi() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return SIMDVec_u<uint32_t, 2>(raw[2], raw[3]);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 4>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 4>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 4>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 4>() const;
    };

}
}

#endif
