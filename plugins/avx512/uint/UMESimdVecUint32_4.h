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
            SIMDVecSwizzle<4 >> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_u<uint32_t, 2 >>
    {
        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_f<float, 4>;

        friend class SIMDVec_u<uint32_t, 8>;

    private:
        __m128i mVec;

        inline explicit SIMDVec_u(__m128i & x) { mVec = x; }
        inline explicit SIMDVec_u(const __m128i & x) { mVec = x; }

#if !defined(__AVX512VL__)
        // This function converts between mask representation and integer vector register representation
        // for 4x32b vectors. This is necessary for platforms that don't support AVX512VL instructions.
        inline __m128i mask8_to_m128i(__mmask8 mask) const {
            __m128i vmask = _mm_set1_epi8(mask);
            __m128i bitmask = _mm_set_epi32(0xF7F7F7F7, 0xFBFBFBFB, 0xFDFDFDFD, 0xFEFEFEFE);
            vmask = _mm_or_si128(vmask, bitmask);
            return _mm_cmpeq_epi32(vmask, _mm_set1_epi32(0xFFFFFFFF));
        }
        // This function converts between integer vector register representation and mask representation
        // for 4x32b vectors. This is necessary for platforms that don't support AVX512VL instructions.
        inline __mmask8 m128i_to_mask8(__m128i vec) const {
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
        inline SIMDVec_u() {}
        // SET-CONSTR
        inline explicit SIMDVec_u(uint32_t i) {
            mVec = _mm_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_u(uint32_t const *p) { this->load(p); };
        // FULL-CONSTR
        inline SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3)
        {
            mVec = _mm_set_epi32(i3, i2, i1, i0);
        }
        // EXTRACT
        inline uint32_t extract(uint32_t index) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*) raw, mVec);
            return raw[index];
        }
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }
        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // INSERT
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            raw[index] = value;
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // ASSIGNV
        inline SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        inline SIMDVec_u & operator=(SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_u & assign(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & assign(uint32_t b) {
            mVec = _mm_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_u & assign(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u & load(uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_epi32(mVec, 0xFF, p);
#else
            mVec = _mm_loadu_si128((__m128i*)p);
#endif
            return *this;
        }
        // MLOAD
        inline SIMDVec_u & load(SIMDVecMask<4> const & mask, uint32_t const * p) {
#if defined(__AVX512VL__)
            mVec = _mm_mask_loadu_epi32(mVec, mask.mMask, p);
#else
            __m128i t0 = _mm_loadu_si128((__m128i*)p);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
#endif
            return *this;
        }
        // LOADA
        inline SIMDVec_u & loada(uint32_t const * p) {
            mVec = _mm_load_si128((__m128i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_u & loada(SIMDVecMask<4> const & mask, uint32_t const * p) {
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline uint32_t * store(uint32_t * p) const {
#if defined(__AVX512VL__)
            _mm_mask_storeu_epi32(p, 0xFF, mVec);
#else
            _mm_storeu_si128((__m128i*) p, mVec);
#endif
            return p;
        }
        // MSTORE
        inline uint32_t * store(SIMDVecMask<4> const & mask, uint32_t * p) const {
            _mm_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline uint32_t * storea(uint32_t * p) const {
            _mm_store_si128((__m128i *)p, mVec);
            return p;
        }
        // MSTOREA
        inline uint32_t * storea(SIMDVecMask<4> const & mask, uint32_t * p) const {
            _mm_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        inline SIMDVec_u blend(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_u(t0);
        }
        // BLENDS
        inline SIMDVec_u blend(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_u(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_u add(SIMDVec_u const & b) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator+ (SIMDVec_u const & b) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MADDV
        inline SIMDVec_u add(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u add(uint32_t b) const {
            __m128i t0 = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        inline SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_u add(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = _mm_add_epi32(mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & adda(uint32_t b) {
            mVec = _mm_add_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u postinc() {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MPOSTINC
        inline SIMDVec_u postinc(SIMDVecMask<4> const & mask) {
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
        inline SIMDVec_u & prefinc() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_add_epi32(mVec, t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVec_u & prefinc(SIMDVecMask<4> const & mask) {
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
        inline SIMDVec_u sub(SIMDVec_u const & b) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MSUBV
        inline SIMDVec_u sub(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u sub(uint32_t b) const {
            __m128i t0 = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        // MSUBS
        inline SIMDVec_u sub(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = _mm_sub_epi32(mVec, b.mVec);
            return *this;
        }
        // MSUBVA
        inline SIMDVec_u & suba(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & suba(uint32_t b) {
            mVec = _mm_sub_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MSUBSA
        inline SIMDVec_u & suba(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u subfrom(SIMDVec_u const & b) const {
            __m128i t0 = _mm_sub_epi32(b.mVec, mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMV
        inline SIMDVec_u subfrom(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u subfrom(uint32_t b) const {
            __m128i t0 = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return SIMDVec_u(t0);
        }
        // MSUBFROMS
        inline SIMDVec_u subfrom(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = _mm_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & subfroma(uint32_t b) {
            mVec = _mm_sub_epi32(_mm_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_u subfroma(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u postdec() {
            __m128i t0 = _mm_set1_epi32(1);
            __m128i t1 = mVec;
            mVec = _mm_sub_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MPOSTDEC
        inline SIMDVec_u postdec(SIMDVecMask<4> const & mask) {
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
        inline SIMDVec_u & prefdec() {
            __m128i t0 = _mm_set1_epi32(1);
            mVec = _mm_sub_epi32(mVec, t0);
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_u & prefdec(SIMDVecMask<4> const & mask) {
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
        inline SIMDVec_u mul(SIMDVec_u const & b) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u mul(uint32_t b) const {
            __m128i t0 = _mm_mullo_epi32(mVec, _mm_set1_epi32(b));
            return SIMDVec_u(t0);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = _mm_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        // MMULVA
        inline SIMDVec_u & mula(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & mula(uint32_t b) {
            mVec = _mm_mullo_epi32(mVec, _mm_set1_epi32(b));
            return *this;
        }
        // MMULSA
        inline SIMDVec_u & mula(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVecMask<4> cmpeq(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu32_mask(mVec, b.mVec);
#else
            __m128i t0 = _mm_cmpeq_epi32(mVec, b.mVec);
            __mmask8 m0 = m128i_to_mask8(t0);
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPEQS
        inline SIMDVecMask<4> cmpeq(uint32_t b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpeq_epu32_mask(mVec, _mm_set1_epi32(b)) & 0xF;
#else
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_cmpeq_epi32(mVec, t0);
            __mmask8 m0 = m128i_to_mask8(t1);
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPNEV
        inline SIMDVecMask<4> cmpne(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpneq_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPNES
        inline SIMDVecMask<4> cmpne(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpneq_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPGTV
        inline SIMDVecMask<4> cmpgt(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpgt_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPGTS
        inline SIMDVecMask<4> cmpgt(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpgt_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPLTV
        inline SIMDVecMask<4> cmplt(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmplt_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPLTS
        inline SIMDVecMask<4> cmplt(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmplt_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPGEV
        inline SIMDVecMask<4> cmpge(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpge_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPGES
        inline SIMDVecMask<4> cmpge(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmpge_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPLEV
        inline SIMDVecMask<4> cmple(SIMDVec_u const & b) const {
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmple_epu32_mask(mVec, b.mVec);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(b.mVec);
            __mmask16 m1 = _mm512_cmple_epu32_mask(t0, t1);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPLES
        inline SIMDVecMask<4> cmple(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
#if defined(__AVX512VL__)
            __mmask8 m0 = _mm_cmple_epu32_mask(mVec, t0);
#else
            __m512i t1 = _mm512_castsi128_si512(mVec);
            __m512i t2 = _mm512_castsi128_si512(t0);
            __mmask16 m1 = _mm512_cmple_epu32_mask(t1, t2);
            __mmask8 m0 = m1 & 0x000F;
#endif
            return SIMDVecMask<4>(m0);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_u const & b) const {
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
        inline bool cmpe(uint32_t b) const {
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
#if defined(__AVX512VL__) && defined(__AVX512CD__)
        inline bool unique() const {
            __m128i t0 = _mm_conflict_epi32(mVec);
            __mmask8 m0 = _mm_cmpeq_epu32_mask(t0, _mm_setzero_si128());
            return (m0 == 0xF);
        }
#endif
        // HADD
        inline uint32_t hadd() const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_reduce_add_epi32(t0);
            return retval;
        }
        // MHADD
        inline uint32_t hadd(SIMDVecMask<4> const mask) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval;
        }
        // HADDS
        inline uint32_t hadd(uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_reduce_add_epi32(t0);
            return retval + b;
        }
        // MHADDS
        inline uint32_t hadd(SIMDVecMask<4> const mask, uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_add_epi32(t1, t0);
            return retval + b;
        }
        // HMUL
        inline uint32_t hmul() const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_mul_epi32(0xF, t0);
            return retval;
        }
        // MHMUL
        inline uint32_t hmul(SIMDVecMask<4> const mask) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
        }
        // HMULS
        inline uint32_t hmul(uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(0xF, t0);
            return retval;
        }
        // MHMULS
        inline uint32_t hmul(SIMDVecMask<4> const mask, uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __mmask16 t1 = 0x000F & __mmask16(mask.mMask);
            uint32_t retval = b;
            retval *= _mm512_mask_reduce_mul_epi32(t1, t0);
            return retval;
        }
        // FMULADDV
        inline SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_add_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULADDV
        inline SIMDVec_u fmuladd(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
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
        inline SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_mullo_epi32(mVec, b.mVec);
            __m128i t1 = _mm_sub_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFMULSUBV
        inline SIMDVec_u fmulsub(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
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
        inline SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_add_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFADDMULV
        inline SIMDVec_u faddmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
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
        inline SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m128i t0 = _mm_sub_epi32(mVec, b.mVec);
            __m128i t1 = _mm_mullo_epi32(t0, c.mVec);
            return SIMDVec_u(t1);
        }
        // MFSUBMULV
        inline SIMDVec_u fsubmul(SIMDVecMask<4> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
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
        inline SIMDVec_u max(SIMDVec_u const & b) const {
            __m128i t0 = _mm_max_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMAXV
        inline SIMDVec_u max(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u max(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_max_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMAXS
        inline SIMDVec_u max(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = _mm_max_epu32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_u & maxa(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & maxa(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_max_epu32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_u & maxa(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u min(SIMDVec_u const & b) const {
            __m128i t0 = _mm_min_epu32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMINV
        inline SIMDVec_u min(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u min(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_min_epu32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMINS
        inline SIMDVec_u min(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = _mm_min_epu32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_u & mina(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & mina(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_min_epu32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_u & mina(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline uint32_t hmax() const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_max_epu32(0xF, t0);
            return retval;
        }       
        // MHMAX
        inline uint32_t hmax(SIMDVecMask<4> const & mask) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_max_epu32(mask.mMask, t0);
            return retval;
        }       
        // IMAX
        // MIMAX
        // HMIN
        inline uint32_t hmin() const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_min_epu32(0xF, t0);
            return retval;
        }       
        // MHMIN
        inline uint32_t hmin(SIMDVecMask<4> const & mask) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_min_epu32(mask.mMask, t0);
            return retval;
        }       
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_u band(SIMDVec_u const & b) const {
            __m128i t0 = _mm_and_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MBANDV
        inline SIMDVec_u band(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u band(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_and_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBANDS
        inline SIMDVec_u band(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = _mm_and_si128(mVec, b.mVec);
            return *this;
        }
        // MBANDVA
        inline SIMDVec_u & banda(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & banda(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_and_si128(mVec, t0);
            return *this;
        }
        // MBANDSA
        inline SIMDVec_u & banda(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u bor(SIMDVec_u const & b) const {
            __m128i t0 = _mm_or_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MBORV
        inline SIMDVec_u bor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u bor(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_or_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBORS
        inline SIMDVec_u bor(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = _mm_or_si128(mVec, b.mVec);
            return *this;
        }
        // MBORVA
        inline SIMDVec_u & bora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & bora(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_or_si128(mVec, t0);
            return *this;
        }
        // MBORSA
        inline SIMDVec_u & bora(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m128i t0 = _mm_xor_si128(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MBXORV
        inline SIMDVec_u bxor(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u bxor(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBXORS
        inline SIMDVec_u bxor(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = _mm_xor_si128(mVec, b.mVec);
            return *this;
        }
        // MBXORVA
        inline SIMDVec_u & bxora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & bxora(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            mVec = _mm_xor_si128(mVec, t0);
            return *this;
        }
        // MBXORSA
        inline SIMDVec_u & bxora(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u bnot() const {
            __m128i t0 = _mm_set1_epi32(0xFFFFFFFF);
            __m128i t1 = _mm_xor_si128(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MBNOT
        inline SIMDVec_u bnot(SIMDVecMask<4> const & mask) const {
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
        inline SIMDVec_u & bnota() {
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
        inline SIMDVec_u bnota(SIMDVecMask<4> const & mask) {
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
        // HBAND
        inline uint32_t hband() const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_and_epi32(0xF, t0);
            return retval;
        }
        // MHBAND
        inline uint32_t hband(SIMDVecMask<4> const mask) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
        }
        // HBANDS
        inline uint32_t hband(uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(0xF, t0);
            return retval;
        }
        // MHBANDS
        inline uint32_t hband(SIMDVecMask<4> const mask, uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval &= _mm512_mask_reduce_and_epi32(mask.mMask, t0);
            return retval;
        }
        // HBOR
        inline uint32_t hbor() const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_or_epi32(0xF, t0);
            return retval;
        }
        // MHBOR
        inline uint32_t hbor(SIMDVecMask<4> const mask) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
        }
        // HBORS
        inline uint32_t hbor(uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(0xF, t0);
            return retval;
        }
        // MHBORS
        inline uint32_t hbor(SIMDVecMask<4> const mask, uint32_t b) const {
            __m512i t0 = _mm512_castsi128_si512(mVec);
            uint32_t retval = b;
            retval |= _mm512_mask_reduce_or_epi32(mask.mMask, t0);
            return retval;
        }
        // HBXOR
        inline uint32_t hbxor() const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXOR
        inline uint32_t hbxor(SIMDVecMask<4> const mask) const {
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
        inline uint32_t hbxor(uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3];
        }
        // MHBXORS
        inline uint32_t hbxor(SIMDVecMask<4> const mask, uint32_t b) const {
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mVec);
            uint32_t t0 = b;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            return t0;
        }

        // GATHERS
        inline SIMDVec_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { 
                baseAddr[indices[0]], baseAddr[indices[1]], 
                baseAddr[indices[2]], baseAddr[indices[3]] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t raw[4] = { 
                baseAddr[indices[0]], baseAddr[indices[1]], 
                baseAddr[indices[2]], baseAddr[indices[3]] };
#if defined(__AVX512VL__)
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, raw);
#else
            __m128i t0 = _mm_load_si128((__m128i*)raw);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
#endif
            return *this;
        }
        // GATHERV
        inline SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(16) uint32_t rawIndices[4];
            alignas(16) uint32_t rawData[4];
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            mVec = _mm_load_si128((__m128i*)rawData);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_u & gather(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(16) uint32_t rawIndices[4];
            alignas(16) uint32_t rawData[4];
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
#if defined(__AVX512VL__)
            mVec = _mm_mask_load_epi32(mVec, mask.mMask, rawData);
#else
            __m128i t0 = _mm_loadu_si128((__m128i*)rawData);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            mVec = _mm_blendv_epi8(mVec, t0, m0);
#endif
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t rawIndices[4] = { indices[0], indices[1], indices[2], indices[3] };
            __m128i t0 = _mm_load_si128((__m128i *) rawIndices);
#if defined(__AVX512VL__)
            _mm_i32scatter_epi32(baseAddr, t0, mVec, 4);
#else
            __m512i t1 = _mm512_castsi128_si512(t0);
            __m512i t2 = _mm512_castsi128_si512(mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, 0xF, t1, t2, 4);
#endif
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(16) uint32_t rawIndices[4] = { indices[0], indices[1], indices[2], indices[3] };
#if defined(__AVX512VL__)
            __m128i t0 = _mm_mask_load_epi32(_mm_set1_epi32(0), mask.mMask, (__m128i *) rawIndices);
            _mm_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 1);
#else
            __m128i t0 = _mm_load_si128((__m128i *) rawIndices);
            __m128i m0 = mask8_to_m128i(mask.mMask);
            __m128i t1 = _mm_blendv_epi8(mVec, t0, m0);
            __m512i t2 = _mm512_castsi128_si512(t1);
            __m512i t3 = _mm512_castsi128_si512(mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, 0xF, t2, t3, 1);
#endif
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) {
#if defined(__AVX512VL__)
            _mm_i32scatter_epi32(baseAddr, indices.mVec, mVec, 4);
#else
            alignas(16) uint32_t rawIndices[4];
            alignas(16) uint32_t rawValues[4];
            _mm_store_si128((__m128i*) rawIndices, indices.mVec);
            _mm_store_si128((__m128i*) rawValues, mVec);
            baseAddr[rawIndices[0]] = rawValues[0];
            baseAddr[rawIndices[1]] = rawValues[1];
            baseAddr[rawIndices[2]] = rawValues[2];
            baseAddr[rawIndices[3]] = rawValues[3];
#endif
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDVecMask<4> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
#if defined(__AVX512VL__)
            _mm_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 1);
#else
            __m512i t0 = _mm512_castsi128_si512(mVec);
            __m512i t1 = _mm512_castsi128_si512(indices.mVec);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask & 0xF, t1, t0, 4);
#endif
            return baseAddr;
        }

        // LSHV
        /*inline SIMDVec_u lsh(SIMDVec_u const & b) const {
            __m128i t0 = _mm_sll_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }*/
        // MLSHV
        /*inline SIMDVec_u lsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }*/
        // LSHS
        /*inline SIMDVec_u lsh(uint32_t b) const {
            __m128i t0 = _mm_cvtsi32_si128(b);
            __m128i t1 = _mm_sll_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }*/
        // MLSHS
        /*inline SIMDVec_u lsh(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_cvtsi32_si128(b);
            __m128i t1 = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }*/
        // LSHVA
        /*
        inline SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec = _mm_sll_epi32(mVec, b.mVec);
            return *this;
        }
        // MLSHVA
        inline SIMDVec_u & lsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // LSHSA
        inline SIMDVec_u & lsha(uint32_t b) {
            __m128i t0 = _mm_cvtsi32_si128(b);
            mVec = _mm_sll_epi32(mVec, t0);
            return *this;
        }
        // MLSHSA
        inline SIMDVec_u & lsha(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_cvtsi32_si128(b);
            mVec = _mm_mask_sll_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RSHV
        inline SIMDVec_u rsh(SIMDVec_u const & b) const {
            __m128i t0 = _mm_srl_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MRSHV
        inline SIMDVec_u rsh(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm_mask_srl_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // RSHS
        inline SIMDVec_u rsh(uint32_t b) const {
            __m128i t0 = _mm_cvtsi32_si128(b);
            __m128i t1 = _mm_srl_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MRSHS
        inline SIMDVec_u rsh(SIMDVecMask<4> const & mask, uint32_t b) const {
            __m128i t0 = _mm_cvtsi32_si128(b);
            __m128i t1 = _mm_mask_srl_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // RSHVA
        inline SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec = _mm_srl_epi32(mVec, b.mVec);
            return *this;
        }
        // MRSHVA
        inline SIMDVec_u & rsha(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
            mVec = _mm_mask_srl_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RSHSA
        inline SIMDVec_u & rsha(uint32_t b) {
            __m128i t0 = _mm_cvtsi32_si128(b);
            mVec = _mm_srl_epi32(mVec, t0);
            return *this;
        }
        // MRSHSA
        inline SIMDVec_u & rsha(SIMDVecMask<4> const & mask, uint32_t b) {
            __m128i t0 = _mm_cvtsi32_si128(b);
            mVec = _mm_mask_srl_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }*/
        // ROLV
        inline SIMDVec_u rol(SIMDVec_u const & b) const {
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
        inline SIMDVec_u rol(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u rol(uint32_t b) const {
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
        inline SIMDVec_u rol(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & rola(SIMDVec_u const & b) {
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
        inline SIMDVec_u & rola(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & rola(uint32_t b) {
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
        inline SIMDVec_u & rola(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u ror(SIMDVec_u const & b) const {
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
        inline SIMDVec_u ror(SIMDVecMask<4> const & mask, SIMDVec_u const & b) const {
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
        inline SIMDVec_u ror(uint32_t b) const {
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
        inline SIMDVec_u ror(SIMDVecMask<4> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & rora(SIMDVec_u const & b) {
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
        inline SIMDVec_u & rora(SIMDVecMask<4> const & mask, SIMDVec_u const & b) {
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
        inline SIMDVec_u & rora(uint32_t b) {
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
        inline SIMDVec_u & rora(SIMDVecMask<4> const & mask, uint32_t b) {
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
        inline SIMDVec_u & pack(SIMDVec_u<uint32_t, 2> const & a, SIMDVec_u<uint32_t, 2> const & b) {
            alignas(16) uint32_t raw[4] = { a.mVec[0], a.mVec[1], b.mVec[0], b.mVec[1] };
            mVec = _mm_load_si128((__m128i*)raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_u & packlo(SIMDVec_u<uint32_t, 2> const & a) {
            alignas(16) uint32_t raw[4] = { a.mVec[0], a.mVec[1], 0, 0};
            mVec = _mm_mask_load_epi32(mVec, 0x3, (__m128i*)raw);
            return *this;
        }
        // PACKHI
        inline SIMDVec_u & packhi(SIMDVec_u<uint32_t, 2> const & b) {
            alignas(16) uint32_t raw[4] = { 0, 0, b.mVec[0], b.mVec[1] };
            mVec = _mm_mask_load_epi32(mVec, 0xC, (__m128i*)raw);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_u<uint32_t, 2> & a, SIMDVec_u<uint32_t, 2> & b) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i *)raw, mVec);
            a.mVec[0] = raw[0];
            a.mVec[1] = raw[1];
            b.mVec[0] = raw[2];
            b.mVec[1] = raw[3];
        }
        // UNPACKLO
        inline SIMDVec_u<uint32_t, 2> unpacklo() const {
            alignas(16) uint32_t raw[4];
            _mm_mask_store_epi32((__m128i*)raw, 0x3, mVec);
            return SIMDVec_u<uint32_t, 2>(raw[0], raw[1]);
        }
        // UNPACKHI
        inline SIMDVec_u<uint32_t, 2> unpackhi() const {
            alignas(16) uint32_t raw[4];
            _mm_mask_store_epi32((__m128i*)raw, 0xC, mVec);
            return SIMDVec_u<uint32_t, 2>(raw[2], raw[3]);
        }

        // UTOI
        inline  operator SIMDVec_i<int32_t, 4> () const;
        // UTOF
        inline  operator SIMDVec_f<float, 4>() const;
    };

}
}

#endif
