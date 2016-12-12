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

#ifndef UME_SIMD_VEC_UINT32_32_H_
#define UME_SIMD_VEC_UINT32_32_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 32> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 32>,
            uint32_t,
            32,
            SIMDVecMask<32>,
            SIMDSwizzle<32>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 32>,
            SIMDVec_u<uint32_t, 16>>
    {
        friend class SIMDVec_i<int32_t, 32>;
        friend class SIMDVec_f<float, 32>;

    private:
        __m512i mVec[2];

        UME_FORCE_INLINE explicit SIMDVec_u(__m512i & x0, __m512i & x1) { 
            mVec[0] = x0;
            mVec[1] = x1;
        }
        UME_FORCE_INLINE explicit SIMDVec_u(const __m512i & x0, const __m512i & x1) { 
            mVec[0] = x0;
            mVec[1] = x1;
        }
    public:

        constexpr static uint32_t length() { return 32; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec[0] = _mm512_set1_epi32(i);
            mVec[1] = mVec[0];
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
        UME_FORCE_INLINE explicit SIMDVec_u(uint32_t const *p) { 
            load(p); 
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i0,  uint32_t i1,  uint32_t i2,  uint32_t i3,
                         uint32_t i4,  uint32_t i5,  uint32_t i6,  uint32_t i7,
                         uint32_t i8,  uint32_t i9,  uint32_t i10, uint32_t i11,
                         uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15,
                         uint32_t i16, uint32_t i17, uint32_t i18, uint32_t i19,
                         uint32_t i20, uint32_t i21, uint32_t i22, uint32_t i23,
                         uint32_t i24, uint32_t i25, uint32_t i26, uint32_t i27,
                         uint32_t i28, uint32_t i29, uint32_t i30, uint32_t i31)
        {
            mVec[0] = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7,
                                        i8, i9, i10, i11, i12, i13, i14, i15);
            mVec[1] = _mm512_setr_epi32(i16, i17, i18, i19, i20, i21, i22, i23,
                                        i24, i25, i26, i27, i28, i29, i30, i31);
        }
        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            alignas(64) uint32_t raw[16];
            uint32_t t0;
            if (index < 16) {
                _mm512_store_si512((__m512i*)raw, mVec[0]);
                t0 = index;
            }
            else {
                _mm512_store_si512((__m512i*)raw, mVec[1]);
                t0 = index - 16;
            }
            return raw[t0];
        }
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(64) uint32_t raw[16];
            uint32_t t0;
            if (index < 16) {
                _mm512_store_si512((__m512i*)raw, mVec[0]);
                t0 = index;
            }
            else {
                _mm512_store_si512((__m512i*)raw, mVec[1]);
                t0 = index - 16;
            }
            
            raw[t0] = value;

            if (index < 16) {
                mVec[0] = _mm512_load_si512((__m512i*)raw);
            }
            else {
                mVec[1] = _mm512_load_si512((__m512i*)raw);
            }
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mov_epi32(mVec[0], m0, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_epi32(mVec[1], m1, b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec[0] = _mm512_set1_epi32(b);
            mVec[1] = mVec[0];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_mov_epi32(mVec[0], m0, t0);
            mVec[1] = _mm512_mask_mov_epi32(mVec[1], m1, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const * p) {
            mVec[0] = _mm512_loadu_si512(p);
            mVec[1] = _mm512_loadu_si512(p + 16);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<32> const & mask, uint32_t const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_epi32(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_epi32(mVec[1], m1, p + 16);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const * p) {
            mVec[0] = _mm512_load_si512((__m512i*)p);
            mVec[1] = _mm512_load_si512((__m512i*)(p + 16));
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<32> const & mask, uint32_t const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_load_epi32(mVec[0], m0, p);
            mVec[1] = _mm512_mask_load_epi32(mVec[1], m1, p + 16);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t * store(uint32_t * p) const {
            _mm512_storeu_si512(p, mVec[0]);
            _mm512_storeu_si512(p + 16, mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t * store(SIMDVecMask<32> const & mask, uint32_t * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_storeu_epi32(p, m0, mVec[0]);
            _mm512_mask_storeu_epi32(p + 16, m1, mVec[1]);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t * storea(uint32_t * p) {
            _mm512_store_si512((__m512i*)p, mVec[0]);
            _mm512_store_si512((__m512i*)(p + 16), mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t * storea(SIMDVecMask<32> const & mask, uint32_t * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_store_epi32(p,      m0, mVec[0]);
            _mm512_mask_store_epi32(p + 16, m1, mVec[1]);
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mov_epi32(mVec[0], m0, b.mVec[0]);
            __m512i t1 = _mm512_mask_mov_epi32(mVec[1], m1, b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<32> const & mask, uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t1 = _mm512_mask_mov_epi32(mVec[0], m0, t0);
            __m512i t2 = _mm512_mask_mov_epi32(mVec[1], m1, t0);
            return SIMDVec_u(t1, t2);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec[0], t0);
            __m512i t2 = _mm512_add_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec[0] = _mm512_add_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA 
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
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
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sub_epi32(mVec[0], t0);
            __m512i t2 = _mm512_sub_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec[0] = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
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
            __m512i t0 = _mm512_sub_epi32(b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_sub_epi32(b.mVec[1], mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec[0], m0, b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[0]);
            __m512i t1 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, m0, t0, mVec[0]);
            __m512i t2 = _mm512_mask_sub_epi32(t0, m1, t0, mVec[1]);
            return SIMDVec_u(t1, t2);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec[0] = _mm512_sub_epi32(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_epi32(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_epi32(b.mVec[0], m0, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec[0] = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[0]);
            mVec[1] = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u subfroma(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_sub_epi32(t0, m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(t0, m1, t0, mVec[1]);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mullo_epi32(mVec[0], t0);
            __m512i t2 = _mm512_mullo_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec[0] = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mullo_epi32(mVec[0], t0);
            mVec[1] = _mm512_mullo_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], t0);
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
        UME_FORCE_INLINE SIMDVecMask<32> cmpeq(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<32> cmpeq(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<32> cmpne(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<32> cmpne(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpneq_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpneq_epu32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<32> cmpgt(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<32> cmpgt(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpgt_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpgt_epu32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<32> cmplt(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmplt_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<32> cmplt(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmplt_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmplt_epu32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<32> cmpge(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmpge_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<32> cmpge(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpge_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpge_epu32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<32> cmple(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmple_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmple_epu32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<32> cmple(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmple_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmple_epu32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_u const & b) const {
            __mmask16 m0 = _mm512_cmple_epu32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmple_epu32_mask(mVec[1], b.mVec[1]);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epu32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpeq_epu32_mask(mVec[1], t0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            alignas(64) uint32_t raw[32];
            _mm512_store_si512(raw, mVec[0]);
            _mm512_store_si512(raw + 16, mVec[1]);

            for (int i = 0; i < 31; i++) {
                for (int j = i; j < 32; j++) {
                    if (raw[i] == raw[j]) {
                        return false;
                    }
                }
            }

            return true;
        }
        // HADD
        UME_FORCE_INLINE uint32_t hadd() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_add_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  + raw[1]  + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                   raw[8]  + raw[9]  + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            uint32_t t0 = _mm512_reduce_add_epi32(mVec[0]);
            uint32_t t1 = _mm512_reduce_add_epi32(mVec[1]);
            return t0 + t1;
#endif
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = 0;
            if (mask.mMask & 0x00000001) t0 += raw[0];
            if (mask.mMask & 0x00000002) t0 += raw[1];
            if (mask.mMask & 0x00000004) t0 += raw[2];
            if (mask.mMask & 0x00000008) t0 += raw[3];
            if (mask.mMask & 0x00000010) t0 += raw[4];
            if (mask.mMask & 0x00000020) t0 += raw[5];
            if (mask.mMask & 0x00000040) t0 += raw[6];
            if (mask.mMask & 0x00000080) t0 += raw[7];
            if (mask.mMask & 0x00000100) t0 += raw[8];
            if (mask.mMask & 0x00000200) t0 += raw[9];
            if (mask.mMask & 0x00000400) t0 += raw[10];
            if (mask.mMask & 0x00000800) t0 += raw[11];
            if (mask.mMask & 0x00001000) t0 += raw[12];
            if (mask.mMask & 0x00002000) t0 += raw[13];
            if (mask.mMask & 0x00004000) t0 += raw[14];
            if (mask.mMask & 0x00008000) t0 += raw[15];
            if (mask.mMask & 0x00010000) t0 += raw[16];
            if (mask.mMask & 0x00020000) t0 += raw[17];
            if (mask.mMask & 0x00040000) t0 += raw[18];
            if (mask.mMask & 0x00080000) t0 += raw[19];
            if (mask.mMask & 0x00100000) t0 += raw[20];
            if (mask.mMask & 0x00200000) t0 += raw[21];
            if (mask.mMask & 0x00400000) t0 += raw[22];
            if (mask.mMask & 0x00800000) t0 += raw[23];
            if (mask.mMask & 0x01000000) t0 += raw[24];
            if (mask.mMask & 0x02000000) t0 += raw[25];
            if (mask.mMask & 0x04000000) t0 += raw[26];
            if (mask.mMask & 0x08000000) t0 += raw[27];
            if (mask.mMask & 0x10000000) t0 += raw[28];
            if (mask.mMask & 0x20000000) t0 += raw[29];
            if (mask.mMask & 0x40000000) t0 += raw[30];
            if (mask.mMask & 0x80000000) t0 += raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            uint32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 + t1;
#endif
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_add_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b + raw[0]  + raw[1]  + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                       raw[8]  + raw[9]  + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            uint32_t t0 = _mm512_reduce_add_epi32(mVec[0]);
            uint32_t t1 = _mm512_reduce_add_epi32(mVec[1]);
            return t0 + t1 + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<32> const & mask, uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = b;
            if (mask.mMask & 0x00000001) t0 += raw[0];
            if (mask.mMask & 0x00000002) t0 += raw[1];
            if (mask.mMask & 0x00000004) t0 += raw[2];
            if (mask.mMask & 0x00000008) t0 += raw[3];
            if (mask.mMask & 0x00000010) t0 += raw[4];
            if (mask.mMask & 0x00000020) t0 += raw[5];
            if (mask.mMask & 0x00000040) t0 += raw[6];
            if (mask.mMask & 0x00000080) t0 += raw[7];
            if (mask.mMask & 0x00000100) t0 += raw[8];
            if (mask.mMask & 0x00000200) t0 += raw[9];
            if (mask.mMask & 0x00000400) t0 += raw[10];
            if (mask.mMask & 0x00000800) t0 += raw[11];
            if (mask.mMask & 0x00001000) t0 += raw[12];
            if (mask.mMask & 0x00002000) t0 += raw[13];
            if (mask.mMask & 0x00004000) t0 += raw[14];
            if (mask.mMask & 0x00008000) t0 += raw[15];
            if (mask.mMask & 0x00010000) t0 += raw[16];
            if (mask.mMask & 0x00020000) t0 += raw[17];
            if (mask.mMask & 0x00040000) t0 += raw[18];
            if (mask.mMask & 0x00080000) t0 += raw[19];
            if (mask.mMask & 0x00100000) t0 += raw[20];
            if (mask.mMask & 0x00200000) t0 += raw[21];
            if (mask.mMask & 0x00400000) t0 += raw[22];
            if (mask.mMask & 0x00800000) t0 += raw[23];
            if (mask.mMask & 0x01000000) t0 += raw[24];
            if (mask.mMask & 0x02000000) t0 += raw[25];
            if (mask.mMask & 0x04000000) t0 += raw[26];
            if (mask.mMask & 0x08000000) t0 += raw[27];
            if (mask.mMask & 0x10000000) t0 += raw[28];
            if (mask.mMask & 0x20000000) t0 += raw[29];
            if (mask.mMask & 0x40000000) t0 += raw[30];
            if (mask.mMask & 0x80000000) t0 += raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            uint32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 + t1 + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_mullo_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  * raw[1]  * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[8]  * raw[9]  * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            uint32_t t0 = _mm512_reduce_mul_epi32(mVec[0]);
            uint32_t t1 = _mm512_reduce_mul_epi32(mVec[1]);
            return t0 * t1;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = 1;
            if (mask.mMask & 0x00000001) t0 *= raw[0];
            if (mask.mMask & 0x00000002) t0 *= raw[1];
            if (mask.mMask & 0x00000004) t0 *= raw[2];
            if (mask.mMask & 0x00000008) t0 *= raw[3];
            if (mask.mMask & 0x00000010) t0 *= raw[4];
            if (mask.mMask & 0x00000020) t0 *= raw[5];
            if (mask.mMask & 0x00000040) t0 *= raw[6];
            if (mask.mMask & 0x00000080) t0 *= raw[7];
            if (mask.mMask & 0x00000100) t0 *= raw[8];
            if (mask.mMask & 0x00000200) t0 *= raw[9];
            if (mask.mMask & 0x00000400) t0 *= raw[10];
            if (mask.mMask & 0x00000800) t0 *= raw[11];
            if (mask.mMask & 0x00001000) t0 *= raw[12];
            if (mask.mMask & 0x00002000) t0 *= raw[13];
            if (mask.mMask & 0x00004000) t0 *= raw[14];
            if (mask.mMask & 0x00008000) t0 *= raw[15];
            if (mask.mMask & 0x00010000) t0 *= raw[16];
            if (mask.mMask & 0x00020000) t0 *= raw[17];
            if (mask.mMask & 0x00040000) t0 *= raw[18];
            if (mask.mMask & 0x00080000) t0 *= raw[19];
            if (mask.mMask & 0x00100000) t0 *= raw[20];
            if (mask.mMask & 0x00200000) t0 *= raw[21];
            if (mask.mMask & 0x00400000) t0 *= raw[22];
            if (mask.mMask & 0x00800000) t0 *= raw[23];
            if (mask.mMask & 0x01000000) t0 *= raw[24];
            if (mask.mMask & 0x02000000) t0 *= raw[25];
            if (mask.mMask & 0x04000000) t0 *= raw[26];
            if (mask.mMask & 0x08000000) t0 *= raw[27];
            if (mask.mMask & 0x10000000) t0 *= raw[28];
            if (mask.mMask & 0x20000000) t0 *= raw[29];
            if (mask.mMask & 0x40000000) t0 *= raw[30];
            if (mask.mMask & 0x80000000) t0 *= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            uint32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 * t1;
#endif
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_mullo_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b * raw[0]  * raw[1]  * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                       raw[8]  * raw[9]  * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            uint32_t t0 = _mm512_reduce_mul_epi32(mVec[0]);
            uint32_t t1 = _mm512_reduce_mul_epi32(mVec[1]);
            return b * t0 * t1;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<32> const & mask, uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = b;
            if (mask.mMask & 0x00000001) t0 *= raw[0];
            if (mask.mMask & 0x00000002) t0 *= raw[1];
            if (mask.mMask & 0x00000004) t0 *= raw[2];
            if (mask.mMask & 0x00000008) t0 *= raw[3];
            if (mask.mMask & 0x00000010) t0 *= raw[4];
            if (mask.mMask & 0x00000020) t0 *= raw[5];
            if (mask.mMask & 0x00000040) t0 *= raw[6];
            if (mask.mMask & 0x00000080) t0 *= raw[7];
            if (mask.mMask & 0x00000100) t0 *= raw[8];
            if (mask.mMask & 0x00000200) t0 *= raw[9];
            if (mask.mMask & 0x00000400) t0 *= raw[10];
            if (mask.mMask & 0x00000800) t0 *= raw[11];
            if (mask.mMask & 0x00001000) t0 *= raw[12];
            if (mask.mMask & 0x00002000) t0 *= raw[13];
            if (mask.mMask & 0x00004000) t0 *= raw[14];
            if (mask.mMask & 0x00008000) t0 *= raw[15];
            if (mask.mMask & 0x00010000) t0 *= raw[16];
            if (mask.mMask & 0x00020000) t0 *= raw[17];
            if (mask.mMask & 0x00040000) t0 *= raw[18];
            if (mask.mMask & 0x00080000) t0 *= raw[19];
            if (mask.mMask & 0x00100000) t0 *= raw[20];
            if (mask.mMask & 0x00200000) t0 *= raw[21];
            if (mask.mMask & 0x00400000) t0 *= raw[22];
            if (mask.mMask & 0x00800000) t0 *= raw[23];
            if (mask.mMask & 0x01000000) t0 *= raw[24];
            if (mask.mMask & 0x02000000) t0 *= raw[25];
            if (mask.mMask & 0x04000000) t0 *= raw[26];
            if (mask.mMask & 0x08000000) t0 *= raw[27];
            if (mask.mMask & 0x10000000) t0 *= raw[28];
            if (mask.mMask & 0x20000000) t0 *= raw[29];
            if (mask.mMask & 0x40000000) t0 *= raw[30];
            if (mask.mMask & 0x80000000) t0 *= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            uint32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return b * t0 * t1;
#endif
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_add_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_add_epi32(t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<32> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_add_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_add_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_sub_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_sub_epi32(t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<32> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_sub_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_sub_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mullo_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_mullo_epi32(t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<32> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mullo_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_mullo_epi32(t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<32> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_max_epu32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_max_epu32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_max_epu32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_max_epu32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epu32(mVec[0], t0);
            __m512i t2 = _mm512_max_epu32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epu32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_max_epu32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec[0] = _mm512_max_epu32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_epu32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_max_epu32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_epu32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_max_epu32(mVec[0], t0);
            mVec[1] = _mm512_max_epu32(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_max_epu32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_max_epu32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_min_epu32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_min_epu32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_min_epu32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_min_epu32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epu32(mVec[0], t0);
            __m512i t2 = _mm512_min_epu32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epu32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_min_epu32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec[0] = _mm512_min_epu32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_epu32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_min_epu32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_epu32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_min_epu32(mVec[0], t0);
            mVec[1] = _mm512_min_epu32(mVec[1], t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_min_epu32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_min_epu32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_max_epu32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            uint32_t t1 = raw[0] > raw[1] ? raw[0] : raw[1];
            uint32_t t2 = raw[2] > raw[3] ? raw[2] : raw[3];
            uint32_t t3 = raw[4] > raw[5] ? raw[4] : raw[5];
            uint32_t t4 = raw[6] > raw[7] ? raw[6] : raw[7];
            uint32_t t5 = raw[8] > raw[9] ? raw[8] : raw[9];
            uint32_t t6 = raw[10] > raw[11] ? raw[10] : raw[11];
            uint32_t t7 = raw[12] > raw[13] ? raw[12] : raw[13];
            uint32_t t8 = raw[14] > raw[15] ? raw[14] : raw[15];

            uint32_t t9 = t1 > t2 ? t1 : t2;
            uint32_t t10 = t3 > t4 ? t3 : t4;
            uint32_t t11 = t5 > t6 ? t5 : t6;
            uint32_t t12 = t7 > t8 ? t7 : t8;

            uint32_t t13 = t9 > t10 ? t9 : t10;
            uint32_t t14 = t11 > t12 ? t11 : t12;

            return t13 > t14 ? t13 : t14;
#else
            uint32_t t0 = _mm512_reduce_max_epu32(mVec[0]);
            uint32_t t1 = _mm512_reduce_max_epu32(mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }       
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 =  ((mask.mMask & 0x00000001) != 0) ? raw[0] : std::numeric_limits<uint32_t>::min();
            uint32_t t1 = (((mask.mMask & 0x00000002) != 0) && raw[1] > t0) ? raw[1] : t0;
            uint32_t t2 = (((mask.mMask & 0x00000004) != 0) && raw[2] > t1) ? raw[2] : t1;
            uint32_t t3 = (((mask.mMask & 0x00000008) != 0) && raw[3] > t2) ? raw[3] : t2;
            uint32_t t4 = (((mask.mMask & 0x00000010) != 0) && raw[4] > t3) ? raw[4] : t3;
            uint32_t t5 = (((mask.mMask & 0x00000020) != 0) && raw[5] > t4) ? raw[5] : t4;
            uint32_t t6 = (((mask.mMask & 0x00000040) != 0) && raw[6] > t5) ? raw[6] : t5;
            uint32_t t7 = (((mask.mMask & 0x00000080) != 0) && raw[7] > t6) ? raw[7] : t6;
            uint32_t t8 = (((mask.mMask & 0x00000100) != 0) && raw[8] > t7) ? raw[8] : t7;
            uint32_t t9 = (((mask.mMask & 0x00000200) != 0) && raw[9] > t8) ? raw[9] : t8;
            uint32_t t10 = (((mask.mMask & 0x00000400) != 0) && raw[10] > t9) ? raw[10] : t9;
            uint32_t t11 = (((mask.mMask & 0x00000800) != 0) && raw[11] > t10) ? raw[11] : t10;
            uint32_t t12 = (((mask.mMask & 0x00001000) != 0) && raw[12] > t11) ? raw[12] : t11;
            uint32_t t13 = (((mask.mMask & 0x00002000) != 0) && raw[13] > t12) ? raw[13] : t12;
            uint32_t t14 = (((mask.mMask & 0x00004000) != 0) && raw[14] > t13) ? raw[14] : t13;
            uint32_t t15 = (((mask.mMask & 0x00008000) != 0) && raw[15] > t14) ? raw[15] : t14;
            uint32_t t16 = (((mask.mMask & 0x00010000) != 0) && raw[16] > t15) ? raw[16] : t15;
            uint32_t t17 = (((mask.mMask & 0x00020000) != 0) && raw[17] > t16) ? raw[17] : t16;
            uint32_t t18 = (((mask.mMask & 0x00040000) != 0) && raw[18] > t17) ? raw[18] : t17;
            uint32_t t19 = (((mask.mMask & 0x00080000) != 0) && raw[19] > t18) ? raw[19] : t18;
            uint32_t t20 = (((mask.mMask & 0x00100000) != 0) && raw[20] > t19) ? raw[20] : t19;
            uint32_t t21 = (((mask.mMask & 0x00200000) != 0) && raw[21] > t20) ? raw[21] : t20;
            uint32_t t22 = (((mask.mMask & 0x00400000) != 0) && raw[22] > t21) ? raw[22] : t21;
            uint32_t t23 = (((mask.mMask & 0x00800000) != 0) && raw[23] > t22) ? raw[23] : t22;
            uint32_t t24 = (((mask.mMask & 0x01000000) != 0) && raw[24] > t23) ? raw[24] : t23;
            uint32_t t25 = (((mask.mMask & 0x02000000) != 0) && raw[25] > t24) ? raw[25] : t24;
            uint32_t t26 = (((mask.mMask & 0x04000000) != 0) && raw[26] > t25) ? raw[26] : t25;
            uint32_t t27 = (((mask.mMask & 0x08000000) != 0) && raw[27] > t26) ? raw[27] : t26;
            uint32_t t28 = (((mask.mMask & 0x10000000) != 0) && raw[28] > t27) ? raw[28] : t27;
            uint32_t t29 = (((mask.mMask & 0x20000000) != 0) && raw[29] > t28) ? raw[29] : t28;
            uint32_t t30 = (((mask.mMask & 0x40000000) != 0) && raw[30] > t29) ? raw[30] : t29;
            uint32_t t31 = (((mask.mMask & 0x80000000) != 0) && raw[31] > t30) ? raw[31] : t30;
            return t31;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_max_epu32(m0, mVec[0]);
            uint32_t t1 = _mm512_mask_reduce_max_epu32(m1, mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }       
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE uint32_t hmin() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_min_epu32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            uint32_t t1 = raw[0] < raw[1] ? raw[0] : raw[1];
            uint32_t t2 = raw[2] < raw[3] ? raw[2] : raw[3];
            uint32_t t3 = raw[4] < raw[5] ? raw[4] : raw[5];
            uint32_t t4 = raw[6] < raw[7] ? raw[6] : raw[7];
            uint32_t t5 = raw[8] < raw[9] ? raw[8] : raw[9];
            uint32_t t6 = raw[10] < raw[11] ? raw[10] : raw[11];
            uint32_t t7 = raw[12] < raw[13] ? raw[12] : raw[13];
            uint32_t t8 = raw[14] < raw[15] ? raw[14] : raw[15];

            uint32_t t9 = t1 < t2 ? t1 : t2;
            uint32_t t10 = t3 < t4 ? t3 : t4;
            uint32_t t11 = t5 < t6 ? t5 : t6;
            uint32_t t12 = t7 < t8 ? t7 : t8;

            uint32_t t13 = t9 < t10 ? t9 : t10;
            uint32_t t14 = t10 < t12 ? t11 : t12;

            return t13 < t14 ? t13 : t14;
#else
            uint32_t t0 = _mm512_reduce_min_epu32(mVec[0]);
            uint32_t t1 = _mm512_reduce_min_epu32(mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }       
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 =  ((mask.mMask & 0x00000001) != 0) ? raw[0] : std::numeric_limits<uint32_t>::max();
            uint32_t t1 = (((mask.mMask & 0x00000002) != 0) && raw[1] < t0) ? raw[1] : t0;
            uint32_t t2 = (((mask.mMask & 0x00000004) != 0) && raw[2] < t1) ? raw[2] : t1;
            uint32_t t3 = (((mask.mMask & 0x00000008) != 0) && raw[3] < t2) ? raw[3] : t2;
            uint32_t t4 = (((mask.mMask & 0x00000010) != 0) && raw[4] < t3) ? raw[4] : t3;
            uint32_t t5 = (((mask.mMask & 0x00000020) != 0) && raw[5] < t4) ? raw[5] : t4;
            uint32_t t6 = (((mask.mMask & 0x00000040) != 0) && raw[6] < t5) ? raw[6] : t5;
            uint32_t t7 = (((mask.mMask & 0x00000080) != 0) && raw[7] < t6) ? raw[7] : t6;
            uint32_t t8 = (((mask.mMask & 0x00000100) != 0) && raw[8] < t7) ? raw[8] : t7;
            uint32_t t9 = (((mask.mMask & 0x00000200) != 0) && raw[9] < t8) ? raw[9] : t8;
            uint32_t t10 = (((mask.mMask & 0x00000400) != 0) && raw[10] < t9) ? raw[10] : t9;
            uint32_t t11 = (((mask.mMask & 0x00000800) != 0) && raw[11] < t10) ? raw[11] : t10;
            uint32_t t12 = (((mask.mMask & 0x00001000) != 0) && raw[12] < t11) ? raw[12] : t11;
            uint32_t t13 = (((mask.mMask & 0x00002000) != 0) && raw[13] < t12) ? raw[13] : t12;
            uint32_t t14 = (((mask.mMask & 0x00004000) != 0) && raw[14] < t13) ? raw[14] : t13;
            uint32_t t15 = (((mask.mMask & 0x00008000) != 0) && raw[15] < t14) ? raw[15] : t14;
            uint32_t t16 = (((mask.mMask & 0x00010000) != 0) && raw[16] < t15) ? raw[16] : t15;
            uint32_t t17 = (((mask.mMask & 0x00020000) != 0) && raw[17] < t16) ? raw[17] : t16;
            uint32_t t18 = (((mask.mMask & 0x00040000) != 0) && raw[18] < t17) ? raw[18] : t17;
            uint32_t t19 = (((mask.mMask & 0x00080000) != 0) && raw[19] < t18) ? raw[19] : t18;
            uint32_t t20 = (((mask.mMask & 0x00100000) != 0) && raw[20] < t19) ? raw[20] : t19;
            uint32_t t21 = (((mask.mMask & 0x00200000) != 0) && raw[21] < t20) ? raw[21] : t20;
            uint32_t t22 = (((mask.mMask & 0x00400000) != 0) && raw[22] < t21) ? raw[22] : t21;
            uint32_t t23 = (((mask.mMask & 0x00800000) != 0) && raw[23] < t22) ? raw[23] : t22;
            uint32_t t24 = (((mask.mMask & 0x01000000) != 0) && raw[24] < t23) ? raw[24] : t23;
            uint32_t t25 = (((mask.mMask & 0x02000000) != 0) && raw[25] < t24) ? raw[25] : t24;
            uint32_t t26 = (((mask.mMask & 0x04000000) != 0) && raw[26] < t25) ? raw[26] : t25;
            uint32_t t27 = (((mask.mMask & 0x08000000) != 0) && raw[27] < t26) ? raw[27] : t26;
            uint32_t t28 = (((mask.mMask & 0x10000000) != 0) && raw[28] < t27) ? raw[28] : t27;
            uint32_t t29 = (((mask.mMask & 0x20000000) != 0) && raw[29] < t28) ? raw[29] : t28;
            uint32_t t30 = (((mask.mMask & 0x40000000) != 0) && raw[30] < t29) ? raw[30] : t29;
            uint32_t t31 = (((mask.mMask & 0x80000000) != 0) && raw[31] < t30) ? raw[31] : t30;
            return t31;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_min_epu32(m0, mVec[0]);
            uint32_t t1 = _mm512_mask_reduce_min_epu32(m1, mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }       
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_and_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec[0], t0);
            __m512i t2 = _mm512_and_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec[0] = _mm512_and_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_and_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_and_epi32(mVec[0], t0);
            mVec[1] = _mm512_and_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_or_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec[0], t0);
            __m512i t2 = _mm512_or_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec[0] = _mm512_or_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_or_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_or_epi32(mVec[0], t0);
            mVec[1] = _mm512_or_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_xor_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec[0], t0);
            __m512i t2 = _mm512_xor_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec[0] = _mm512_xor_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_xor_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_xor_epi32(mVec[0], t0);
            mVec[1] = _mm512_xor_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_andnot_epi32(mVec[0], t0);
            __m512i t2 = _mm512_andnot_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_andnot_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm512_andnot_epi32(mVec[0], t0);
            mVec[1] = _mm512_andnot_epi32(mVec[1], t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u bnota(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm512_mask_andnot_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_andnot_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_and_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  & raw[1]  & raw[2]  & raw[3]  & raw[4]  & raw[5]  & raw[6]  & raw[7] &
                   raw[8]  & raw[9]  & raw[10] & raw[11] & raw[12] & raw[13] & raw[14] & raw[15];
#else
            uint32_t t0 = _mm512_reduce_and_epi32(mVec[0]);
            t0 &= _mm512_reduce_and_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = 0xFFFFFFFF;
            if (mask.mMask & 0x00000001) t0 &= raw[0];
            if (mask.mMask & 0x00000002) t0 &= raw[1];
            if (mask.mMask & 0x00000004) t0 &= raw[2];
            if (mask.mMask & 0x00000008) t0 &= raw[3];
            if (mask.mMask & 0x00000010) t0 &= raw[4];
            if (mask.mMask & 0x00000020) t0 &= raw[5];
            if (mask.mMask & 0x00000040) t0 &= raw[6];
            if (mask.mMask & 0x00000080) t0 &= raw[7];
            if (mask.mMask & 0x00000100) t0 &= raw[8];
            if (mask.mMask & 0x00000200) t0 &= raw[9];
            if (mask.mMask & 0x00000400) t0 &= raw[10];
            if (mask.mMask & 0x00000800) t0 &= raw[11];
            if (mask.mMask & 0x00001000) t0 &= raw[12];
            if (mask.mMask & 0x00002000) t0 &= raw[13];
            if (mask.mMask & 0x00004000) t0 &= raw[14];
            if (mask.mMask & 0x00008000) t0 &= raw[15];
            if (mask.mMask & 0x00010000) t0 &= raw[16];
            if (mask.mMask & 0x00020000) t0 &= raw[17];
            if (mask.mMask & 0x00040000) t0 &= raw[18];
            if (mask.mMask & 0x00080000) t0 &= raw[19];
            if (mask.mMask & 0x00100000) t0 &= raw[20];
            if (mask.mMask & 0x00200000) t0 &= raw[21];
            if (mask.mMask & 0x00400000) t0 &= raw[22];
            if (mask.mMask & 0x00800000) t0 &= raw[23];
            if (mask.mMask & 0x01000000) t0 &= raw[24];
            if (mask.mMask & 0x02000000) t0 &= raw[25];
            if (mask.mMask & 0x04000000) t0 &= raw[26];
            if (mask.mMask & 0x08000000) t0 &= raw[27];
            if (mask.mMask & 0x10000000) t0 &= raw[28];
            if (mask.mMask & 0x20000000) t0 &= raw[29];
            if (mask.mMask & 0x40000000) t0 &= raw[30];
            if (mask.mMask & 0x80000000) t0 &= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_and_epi32(m0, mVec[0]);
            t0 &= _mm512_mask_reduce_and_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_and_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b & raw[0]  & raw[1]  & raw[2]  & raw[3]  & raw[4]  & raw[5]  & raw[6]  & raw[7] &
                       raw[8]  & raw[9]  & raw[10] & raw[11] & raw[12] & raw[13] & raw[14] & raw[15];
#else
            uint32_t t0 = b;
            t0 &= _mm512_reduce_and_epi32(mVec[0]);
            t0 &= _mm512_reduce_and_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<32> const & mask, uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = b;
            if (mask.mMask & 0x00000001) t0 &= raw[0];
            if (mask.mMask & 0x00000002) t0 &= raw[1];
            if (mask.mMask & 0x00000004) t0 &= raw[2];
            if (mask.mMask & 0x00000008) t0 &= raw[3];
            if (mask.mMask & 0x00000010) t0 &= raw[4];
            if (mask.mMask & 0x00000020) t0 &= raw[5];
            if (mask.mMask & 0x00000040) t0 &= raw[6];
            if (mask.mMask & 0x00000080) t0 &= raw[7];
            if (mask.mMask & 0x00000100) t0 &= raw[8];
            if (mask.mMask & 0x00000200) t0 &= raw[9];
            if (mask.mMask & 0x00000400) t0 &= raw[10];
            if (mask.mMask & 0x00000800) t0 &= raw[11];
            if (mask.mMask & 0x00001000) t0 &= raw[12];
            if (mask.mMask & 0x00002000) t0 &= raw[13];
            if (mask.mMask & 0x00004000) t0 &= raw[14];
            if (mask.mMask & 0x00008000) t0 &= raw[15];
            if (mask.mMask & 0x00010000) t0 &= raw[16];
            if (mask.mMask & 0x00020000) t0 &= raw[17];
            if (mask.mMask & 0x00040000) t0 &= raw[18];
            if (mask.mMask & 0x00080000) t0 &= raw[19];
            if (mask.mMask & 0x00100000) t0 &= raw[20];
            if (mask.mMask & 0x00200000) t0 &= raw[21];
            if (mask.mMask & 0x00400000) t0 &= raw[22];
            if (mask.mMask & 0x00800000) t0 &= raw[23];
            if (mask.mMask & 0x01000000) t0 &= raw[24];
            if (mask.mMask & 0x02000000) t0 &= raw[25];
            if (mask.mMask & 0x04000000) t0 &= raw[26];
            if (mask.mMask & 0x08000000) t0 &= raw[27];
            if (mask.mMask & 0x10000000) t0 &= raw[28];
            if (mask.mMask & 0x20000000) t0 &= raw[29];
            if (mask.mMask & 0x40000000) t0 &= raw[30];
            if (mask.mMask & 0x80000000) t0 &= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = b;
            t0 &= _mm512_mask_reduce_and_epi32(m0, mVec[0]);
            t0 &= _mm512_mask_reduce_and_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_or_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  | raw[1]  | raw[2]  | raw[3]  | raw[4]  | raw[5]  | raw[6]  | raw[7] |
                   raw[8]  | raw[9]  | raw[10] | raw[11] | raw[12] | raw[13] | raw[14] | raw[15];
#else
            uint32_t t0 = _mm512_reduce_or_epi32(mVec[0]);
            t0 |= _mm512_reduce_or_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = 0;
            if (mask.mMask & 0x00000001) t0 |= raw[0];
            if (mask.mMask & 0x00000002) t0 |= raw[1];
            if (mask.mMask & 0x00000004) t0 |= raw[2];
            if (mask.mMask & 0x00000008) t0 |= raw[3];
            if (mask.mMask & 0x00000010) t0 |= raw[4];
            if (mask.mMask & 0x00000020) t0 |= raw[5];
            if (mask.mMask & 0x00000040) t0 |= raw[6];
            if (mask.mMask & 0x00000080) t0 |= raw[7];
            if (mask.mMask & 0x00000100) t0 |= raw[8];
            if (mask.mMask & 0x00000200) t0 |= raw[9];
            if (mask.mMask & 0x00000400) t0 |= raw[10];
            if (mask.mMask & 0x00000800) t0 |= raw[11];
            if (mask.mMask & 0x00001000) t0 |= raw[12];
            if (mask.mMask & 0x00002000) t0 |= raw[13];
            if (mask.mMask & 0x00004000) t0 |= raw[14];
            if (mask.mMask & 0x00008000) t0 |= raw[15];
            if (mask.mMask & 0x00010000) t0 |= raw[16];
            if (mask.mMask & 0x00020000) t0 |= raw[17];
            if (mask.mMask & 0x00040000) t0 |= raw[18];
            if (mask.mMask & 0x00080000) t0 |= raw[19];
            if (mask.mMask & 0x00100000) t0 |= raw[20];
            if (mask.mMask & 0x00200000) t0 |= raw[21];
            if (mask.mMask & 0x00400000) t0 |= raw[22];
            if (mask.mMask & 0x00800000) t0 |= raw[23];
            if (mask.mMask & 0x01000000) t0 |= raw[24];
            if (mask.mMask & 0x02000000) t0 |= raw[25];
            if (mask.mMask & 0x04000000) t0 |= raw[26];
            if (mask.mMask & 0x08000000) t0 |= raw[27];
            if (mask.mMask & 0x10000000) t0 |= raw[28];
            if (mask.mMask & 0x20000000) t0 |= raw[29];
            if (mask.mMask & 0x40000000) t0 |= raw[30];
            if (mask.mMask & 0x80000000) t0 |= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = _mm512_mask_reduce_or_epi32(m0, mVec[0]);
            t0 |= _mm512_mask_reduce_or_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_or_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b | raw[0]  | raw[1]  | raw[2]  | raw[3]  | raw[4]  | raw[5]  | raw[6]  | raw[7] |
                       raw[8]  | raw[9]  | raw[10] | raw[11] | raw[12] | raw[13] | raw[14] | raw[15];
#else
            uint32_t t0 = b;
            t0 |= _mm512_reduce_or_epi32(mVec[0]);
            t0 |= _mm512_reduce_or_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<32> const & mask, uint32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_2)
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = b;
            if (mask.mMask & 0x00000001) t0 |= raw[0];
            if (mask.mMask & 0x00000002) t0 |= raw[1];
            if (mask.mMask & 0x00000004) t0 |= raw[2];
            if (mask.mMask & 0x00000008) t0 |= raw[3];
            if (mask.mMask & 0x00000010) t0 |= raw[4];
            if (mask.mMask & 0x00000020) t0 |= raw[5];
            if (mask.mMask & 0x00000040) t0 |= raw[6];
            if (mask.mMask & 0x00000080) t0 |= raw[7];
            if (mask.mMask & 0x00000100) t0 |= raw[8];
            if (mask.mMask & 0x00000200) t0 |= raw[9];
            if (mask.mMask & 0x00000400) t0 |= raw[10];
            if (mask.mMask & 0x00000800) t0 |= raw[11];
            if (mask.mMask & 0x00001000) t0 |= raw[12];
            if (mask.mMask & 0x00002000) t0 |= raw[13];
            if (mask.mMask & 0x00004000) t0 |= raw[14];
            if (mask.mMask & 0x00008000) t0 |= raw[15];
            if (mask.mMask & 0x00010000) t0 |= raw[16];
            if (mask.mMask & 0x00020000) t0 |= raw[17];
            if (mask.mMask & 0x00040000) t0 |= raw[18];
            if (mask.mMask & 0x00080000) t0 |= raw[19];
            if (mask.mMask & 0x00100000) t0 |= raw[20];
            if (mask.mMask & 0x00200000) t0 |= raw[21];
            if (mask.mMask & 0x00400000) t0 |= raw[22];
            if (mask.mMask & 0x00800000) t0 |= raw[23];
            if (mask.mMask & 0x01000000) t0 |= raw[24];
            if (mask.mMask & 0x02000000) t0 |= raw[25];
            if (mask.mMask & 0x04000000) t0 |= raw[26];
            if (mask.mMask & 0x08000000) t0 |= raw[27];
            if (mask.mMask & 0x10000000) t0 |= raw[28];
            if (mask.mMask & 0x20000000) t0 |= raw[29];
            if (mask.mMask & 0x40000000) t0 |= raw[30];
            if (mask.mMask & 0x80000000) t0 |= raw[31];
            return t0;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            uint32_t t0 = b;
            t0 |= _mm512_mask_reduce_or_epi32(m0, mVec[0]);
            t0 |= _mm512_mask_reduce_or_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_xor_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            uint32_t t1 = 0;
            for (int i = 0; i < 16; i++) {
                t1 ^= raw[i];
            }
            return t1;
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<32> const & mask) const {
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = 0;
            for (int i = 0; i < 32; i++) {
                if ((mask.mMask & (1 << i)) != 0) t0 ^= raw[i];
            }
            return t0;
        }
        // HBXORS
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            alignas(64) uint32_t raw[16];
            __m512i t0 = _mm512_xor_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            uint32_t t1 = 0;
            for (int i = 0; i < 16; i++) {
                t1 ^= raw[i];
            }
            return b ^ t1;
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<32> const & mask, uint32_t b) const {
            alignas(64) uint32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            uint32_t t0 = b;
            for (int i = 0; i < 32; i++) {
                if ((mask.mMask & (1 << i)) != 0) t0 ^= raw[i];
            }
            return t0;
        }
        // GATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(uint32_t const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            mVec[0] = _mm512_i32gather_epi32(t3, (const int *)baseAddr, 4);
            mVec[1] = _mm512_i32gather_epi32(t4, (const int *)baseAddr, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_u & gatheru(SIMDVecMask<32> const & mask, uint32_t const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            mVec[0] = _mm512_mask_i32gather_epi32(mVec[0], mask.mMask & 0x0000FFFF, t3, (const int *)baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_epi32(mVec[1], (mask.mMask & 0xFFFF0000) >> 16, t4, (const int *)baseAddr, 4);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices+16);
            mVec[0] = _mm512_i32gather_epi32(t0, (const int *)baseAddr, 4);
            mVec[1] = _mm512_i32gather_epi32(t1, (const int *)baseAddr, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<32> const & mask, uint32_t const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices+16);
            mVec[0] = _mm512_mask_i32gather_epi32(mVec[0], mask.mMask & 0x0000FFFF, t0, (const int *)baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_epi32(mVec[1], (mask.mMask & 0xFFFF0000) >> 16, t1, (const int *)baseAddr, 4);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = _mm512_i32gather_epi32(indices.mVec[0], (const int *)baseAddr, 4);
            mVec[1] = _mm512_i32gather_epi32(indices.mVec[1], (const int *)baseAddr, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<32> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = _mm512_mask_i32gather_epi32(mVec[0], mask.mMask & 0x0000FFFF, indices.mVec[0], (const int *)baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_epi32(mVec[1], (mask.mMask & 0xFFFF0000) >> 16, indices.mVec[1], (const int *)baseAddr, 4);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE uint32_t* scatteru(uint32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            _mm512_i32scatter_epi32((int *)baseAddr, t3, mVec[0], 4);
            _mm512_i32scatter_epi32((int *)baseAddr, t4, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE uint32_t*  scatteru(SIMDVecMask<32> const & mask, uint32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, mask.mMask & 0x0000FFFF, t3, mVec[0], 4);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, (mask.mMask & 0xFFFF0000) >> 16, t4, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            __m512i t0 = _mm512_loadu_si512((__m512i *) indices);
            __m512i t1 = _mm512_loadu_si512((__m512i *) (indices + 16));
            _mm512_i32scatter_epi32((int *)baseAddr, t0, mVec[0], 4);
            _mm512_i32scatter_epi32((int *)baseAddr, t1, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<32> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), m0, (__m512i *) indices);
            __m512i t1 = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), m1, (__m512i *) (indices + 16));
            _mm512_mask_i32scatter_epi32((int *)baseAddr, m0, t0, mVec[0], 4);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, m1, t1, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            _mm512_i32scatter_epi32((int *)baseAddr, indices.mVec[0], mVec[0], 4);
            _mm512_i32scatter_epi32((int *)baseAddr, indices.mVec[1], mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<32> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_i32scatter_epi32((int *)baseAddr, m0, indices.mVec[0], mVec[0], 4);
            _mm512_mask_i32scatter_epi32((int *)baseAddr, m1, indices.mVec[1], mVec[1], 4);
            return baseAddr;
        }
        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sllv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sllv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (SIMDVec_u const & b) const {
            return lsh(b);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sllv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sllv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sllv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_sllv_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator<< (uint32_t b) const {
            return lsh(b);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sllv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_sllv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_srlv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_srlv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (SIMDVec_u const & b) const {
            return rsh(b);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_srlv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_srlv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_srlv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_srlv_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator>> (uint32_t b) const {
            return rsh(b);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_srlv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_srlv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
        // ROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_rolv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_rolv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_u rol(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_rolv_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_u rol(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVec_u const & b) {
            mVec[0] = _mm512_rolv_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_rolv_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MROLVA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_rolv_epi32(mVec[0], t0);
            mVec[1] = _mm512_rolv_epi32(mVec[1], t0);
            return *this;
        }
        // MROLSA
        UME_FORCE_INLINE SIMDVec_u & rola(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // RORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_rorv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<32> const & mask, SIMDVec_u const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_u ror(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_rorv_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_u ror(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVec_u const & b) {
            mVec[0] = _mm512_rorv_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_rorv_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MRORVA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<32> const & mask, SIMDVec_u const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // RORSA
        UME_FORCE_INLINE SIMDVec_u & rora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_rorv_epi32(mVec[0], t0);
            mVec[1] = _mm512_rorv_epi32(mVec[1], t0);
            return *this;
        }
        // MRORSA
        UME_FORCE_INLINE SIMDVec_u & rora(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }

        // PACK
        UME_FORCE_INLINE SIMDVec_u & pack(SIMDVec_u<uint32_t, 16> const & a, SIMDVec_u<uint32_t, 16> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_u & packlo(SIMDVec_u<uint32_t, 16> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_u & packhi(SIMDVec_u<uint32_t, 16> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_u<uint32_t, 16> & a, SIMDVec_u<uint32_t, 16> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 16> unpacklo() const {
            return SIMDVec_u<uint32_t, 16>(mVec[0]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 16> unpackhi() const {
            return SIMDVec_u<uint32_t, 16>(mVec[1]);
        }

        // PROMOTE
        // - 
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 32>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 32> () const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 32>() const;

    };

}
}

#endif
