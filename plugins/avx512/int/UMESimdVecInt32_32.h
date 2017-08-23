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

#ifndef UME_SIMD_VEC_INT32_32_H_
#define UME_SIMD_VEC_INT32_32_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 32> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 32>,
            SIMDVec_u<uint32_t, 32>,
            int32_t,
            16,
            uint32_t,
            SIMDVecMask<32>,
            SIMDSwizzle<32>> ,
        public SIMDVecPackableInterface<
           SIMDVec_i<int32_t, 32>,
           SIMDVec_i<int32_t, 16>>
    {
        friend class SIMDVec_u<uint32_t, 32>;
        friend class SIMDVec_f<float, 32>;

    private:
        __m512i mVec[2];

        UME_FORCE_INLINE explicit SIMDVec_i(__m512i const & x0, __m512i const & x1) { 
            mVec[0] = x0;
            mVec[1] = x1;
        }
    public:

        constexpr static uint32_t length() { return 32; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};

        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            mVec[0] = _mm512_set1_epi32(i);
            mVec[1] = mVec[0]; 
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const * p) {
            mVec[0] = _mm512_loadu_si512((void *)p);
            mVec[1] = _mm512_loadu_si512((void *)(p + 16));
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i0,  int32_t i1,  int32_t i2,  int32_t i3,
                         int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                         int32_t i8,  int32_t i9,  int32_t i10, int32_t i11,
                         int32_t i12, int32_t i13, int32_t i14, int32_t i15,
                         int32_t i16, int32_t i17, int32_t i18, int32_t i19,
                         int32_t i20, int32_t i21, int32_t i22, int32_t i23,
                         int32_t i24, int32_t i25, int32_t i26, int32_t i27,
                         int32_t i28, int32_t i29, int32_t i30, int32_t i31)
        {
            mVec[0] = _mm512_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7,
                                        i8, i9, i10, i11, i12, i13, i14, i15);
            mVec[1] = _mm512_setr_epi32(i16, i17, i18, i19, i20, i21, i22, i23,
                                        i24, i25, i26, i27, i28, i29, i30, i31);
        }
        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            alignas(64) int32_t raw[16];
            int32_t t0;
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
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            alignas(64) int32_t raw[16];
            int32_t t0;
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
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mov_epi32(mVec[0], m0, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_epi32(mVec[1], m1, b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            mVec[0] = _mm512_set1_epi32(b);
            mVec[1] = mVec[0];
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<32> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const * p) {
            mVec[0] = _mm512_loadu_si512(p);
            mVec[1] = _mm512_loadu_si512(p + 16);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<32> const & mask, int32_t const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_epi32(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_epi32(mVec[1], m1, p + 16);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const * p) {
            mVec[0] = _mm512_load_si512((__m512i*)p);
            mVec[1] = _mm512_load_si512((__m512i*)(p + 16));
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<32> const & mask, int32_t const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_load_epi32(mVec[0], m0, p);
            mVec[1] = _mm512_mask_load_epi32(mVec[1], m1, p + 16);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t * store(int32_t * p) const {
            _mm512_storeu_si512(p, mVec[0]);
            _mm512_storeu_si512(p + 16, mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t * store(SIMDVecMask<32> const & mask, int32_t * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_storeu_epi32(p, m0, mVec[0]);
            _mm512_mask_storeu_epi32(p + 16, m1, mVec[1]);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int32_t * storea(int32_t * p) {
            _mm512_store_si512((__m512i*)p, mVec[0]);
            _mm512_store_si512((__m512i*)(p + 16), mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int32_t * storea(SIMDVecMask<32> const & mask, int32_t * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_store_epi32(p,      m0, mVec[0]);
            _mm512_mask_store_epi32(p + 16, m1, mVec[1]);
            return p;
        }
        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mov_epi32(mVec[0], m0, b.mVec[0]);
            __m512i t1 = _mm512_mask_mov_epi32(mVec[1], m1, b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<32> const & mask, int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t1 = _mm512_mask_mov_epi32(mVec[0], m0, t0);
            __m512i t2 = _mm512_mask_mov_epi32(mVec[1], m1, t0);
            return SIMDVec_i(t1, t2);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec[0], t0);
            __m512i t2 = _mm512_add_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec[0] = _mm512_add_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA 
        UME_FORCE_INLINE SIMDVec_i & adda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<32> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sub_epi32(mVec[0], t0);
            __m512i t2 = _mm512_sub_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec[0] = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<32> const & mask, int32_t b) {
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
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_sub_epi32(b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec[0], m0, b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[0]);
            __m512i t1 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, m0, t0, mVec[0]);
            __m512i t2 = _mm512_mask_sub_epi32(t0, m1, t0, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec[0] = _mm512_sub_epi32(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_epi32(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_epi32(b.mVec[0], m0, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int32_t b) {
            mVec[0] = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[0]);
            mVec[1] = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i subfroma(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_sub_epi32(t0, m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(t0, m1, t0, mVec[1]);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mullo_epi32(mVec[0], t0);
            __m512i t2 = _mm512_mullo_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec[0] = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mullo_epi32(mVec[0], t0);
            mVec[1] = _mm512_mullo_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], t0);
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
        UME_FORCE_INLINE SIMDVecMask<32> cmpeq(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<32> cmpeq(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<32> cmpne(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator!=(SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<32> cmpne(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator!=(int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<32> cmpgt(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpgt_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<32> cmpgt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpgt_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<32> cmplt(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmplt_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<32> cmplt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmplt_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<32> cmpge(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpge_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<32> cmpge(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpge_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<32> cmple(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmple_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<32> cmple(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmple_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        UME_FORCE_INLINE SIMDVecMask<32> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmple_epi32_mask(mVec[1], b.mVec[1]);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(mVec[1], t0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            alignas(64) int32_t raw[32];
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
        UME_FORCE_INLINE int32_t hadd() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_add_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  + raw[1]  + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                   raw[8]  + raw[9]  + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            int32_t t0 = _mm512_reduce_add_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_add_epi32(mVec[1]);
            return t0 + t1;
#endif
        }
        // MHADD
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = 0;
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
            int32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 + t1;
#endif
        }
        // HADDS
        UME_FORCE_INLINE int32_t hadd(int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_add_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b + raw[0]  + raw[1]  + raw[2]  + raw[3]  + raw[4]  + raw[5]  + raw[6]  + raw[7] +
                       raw[8]  + raw[9]  + raw[10] + raw[11] + raw[12] + raw[13] + raw[14] + raw[15];
#else
            int32_t t0 = _mm512_reduce_add_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_add_epi32(mVec[1]);
            return t0 + t1 + b;
#endif
        }
        // MHADDS
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<32> const & mask, int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = b;
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
            int32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 + t1 + b;
#endif
        }
        // HMUL
        UME_FORCE_INLINE int32_t hmul() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_mullo_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  * raw[1]  * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                   raw[8]  * raw[9]  * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            int32_t t0 = _mm512_reduce_mul_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_mul_epi32(mVec[1]);
            return t0 * t1;
#endif
        }
        // MHMUL
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = 1;
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
            int32_t t0 = _mm512_mask_reduce_mul_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_mul_epi32(m1, mVec[1]);
            return t0 * t1;
#endif
        }
        // HMULS
        UME_FORCE_INLINE int32_t hmul(int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_mullo_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b * raw[0]  * raw[1]  * raw[2]  * raw[3]  * raw[4]  * raw[5]  * raw[6]  * raw[7] *
                       raw[8]  * raw[9]  * raw[10] * raw[11] * raw[12] * raw[13] * raw[14] * raw[15];
#else
            int32_t t0 = _mm512_reduce_mul_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_mul_epi32(mVec[1]);
            return b * t0 * t1;
#endif
        }
        // MHMULS
        UME_FORCE_INLINE int32_t hmul(SIMDVecMask<32> const & mask, int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = b;
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
            int32_t t0 = _mm512_mask_reduce_mul_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_mul_epi32(m1, mVec[1]);
            return b * t0 * t1;
#endif
        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_add_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_add_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_add_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_add_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_sub_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_sub_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_sub_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_sub_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mullo_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_mullo_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mullo_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_mullo_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_max_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_max_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epi32(mVec[0], t0);
            __m512i t2 = _mm512_max_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec[0] = _mm512_max_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_max_epi32(mVec[0], t0);
            mVec[1] = _mm512_max_epi32(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_min_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_min_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epi32(mVec[0], t0);
            __m512i t2 = _mm512_min_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec[0] = _mm512_min_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_min_epi32(mVec[0], t0);
            mVec[1] = _mm512_min_epi32(mVec[1], t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE int32_t hmax() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_max_epu32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            int32_t t1 = raw[0] > raw[1] ? raw[0] : raw[1];
            int32_t t2 = raw[2] > raw[3] ? raw[2] : raw[3];
            int32_t t3 = raw[4] > raw[5] ? raw[4] : raw[5];
            int32_t t4 = raw[6] > raw[7] ? raw[6] : raw[7];
            int32_t t5 = raw[8] > raw[9] ? raw[8] : raw[9];
            int32_t t6 = raw[10] > raw[11] ? raw[10] : raw[11];
            int32_t t7 = raw[12] > raw[13] ? raw[12] : raw[13];
            int32_t t8 = raw[14] > raw[15] ? raw[14] : raw[15];

            int32_t t9 = t1 > t2 ? t1 : t2;
            int32_t t10 = t3 > t4 ? t3 : t4;
            int32_t t11 = t5 > t6 ? t5 : t6;
            int32_t t12 = t7 > t8 ? t7 : t8;

            int32_t t13 = t9 > t10 ? t9 : t10;
            int32_t t14 = t11 > t12 ? t11 : t12;

            return t13 > t14 ? t13 : t14;
#else
            int32_t t0 = _mm512_reduce_max_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_max_epi32(mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }       
        // MHMAX
        UME_FORCE_INLINE int32_t hmax(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 =  ((mask.mMask & 0x00000001) != 0) ? raw[0] : std::numeric_limits<int32_t>::min();
            int32_t t1 = (((mask.mMask & 0x00000002) != 0) && raw[1] > t0) ? raw[1] : t0;
            int32_t t2 = (((mask.mMask & 0x00000004) != 0) && raw[2] > t1) ? raw[2] : t1;
            int32_t t3 = (((mask.mMask & 0x00000008) != 0) && raw[3] > t2) ? raw[3] : t2;
            int32_t t4 = (((mask.mMask & 0x00000010) != 0) && raw[4] > t3) ? raw[4] : t3;
            int32_t t5 = (((mask.mMask & 0x00000020) != 0) && raw[5] > t4) ? raw[5] : t4;
            int32_t t6 = (((mask.mMask & 0x00000040) != 0) && raw[6] > t5) ? raw[6] : t5;
            int32_t t7 = (((mask.mMask & 0x00000080) != 0) && raw[7] > t6) ? raw[7] : t6;
            int32_t t8 = (((mask.mMask & 0x00000100) != 0) && raw[8] > t7) ? raw[8] : t7;
            int32_t t9 = (((mask.mMask & 0x00000200) != 0) && raw[9] > t8) ? raw[9] : t8;
            int32_t t10 = (((mask.mMask & 0x00000400) != 0) && raw[10] > t9) ? raw[10] : t9;
            int32_t t11 = (((mask.mMask & 0x00000800) != 0) && raw[11] > t10) ? raw[11] : t10;
            int32_t t12 = (((mask.mMask & 0x00001000) != 0) && raw[12] > t11) ? raw[12] : t11;
            int32_t t13 = (((mask.mMask & 0x00002000) != 0) && raw[13] > t12) ? raw[13] : t12;
            int32_t t14 = (((mask.mMask & 0x00004000) != 0) && raw[14] > t13) ? raw[14] : t13;
            int32_t t15 = (((mask.mMask & 0x00008000) != 0) && raw[15] > t14) ? raw[15] : t14;
            int32_t t16 = (((mask.mMask & 0x00010000) != 0) && raw[16] > t15) ? raw[16] : t15;
            int32_t t17 = (((mask.mMask & 0x00020000) != 0) && raw[17] > t16) ? raw[17] : t16;
            int32_t t18 = (((mask.mMask & 0x00040000) != 0) && raw[18] > t17) ? raw[18] : t17;
            int32_t t19 = (((mask.mMask & 0x00080000) != 0) && raw[19] > t18) ? raw[19] : t18;
            int32_t t20 = (((mask.mMask & 0x00100000) != 0) && raw[20] > t19) ? raw[20] : t19;
            int32_t t21 = (((mask.mMask & 0x00200000) != 0) && raw[21] > t20) ? raw[21] : t20;
            int32_t t22 = (((mask.mMask & 0x00400000) != 0) && raw[22] > t21) ? raw[22] : t21;
            int32_t t23 = (((mask.mMask & 0x00800000) != 0) && raw[23] > t22) ? raw[23] : t22;
            int32_t t24 = (((mask.mMask & 0x01000000) != 0) && raw[24] > t23) ? raw[24] : t23;
            int32_t t25 = (((mask.mMask & 0x02000000) != 0) && raw[25] > t24) ? raw[25] : t24;
            int32_t t26 = (((mask.mMask & 0x04000000) != 0) && raw[26] > t25) ? raw[26] : t25;
            int32_t t27 = (((mask.mMask & 0x08000000) != 0) && raw[27] > t26) ? raw[27] : t26;
            int32_t t28 = (((mask.mMask & 0x10000000) != 0) && raw[28] > t27) ? raw[28] : t27;
            int32_t t29 = (((mask.mMask & 0x20000000) != 0) && raw[29] > t28) ? raw[29] : t28;
            int32_t t30 = (((mask.mMask & 0x40000000) != 0) && raw[30] > t29) ? raw[30] : t29;
            int32_t t31 = (((mask.mMask & 0x80000000) != 0) && raw[31] > t30) ? raw[31] : t30;
            return t31;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_max_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_max_epi32(m1, mVec[1]);
            return t0 > t1 ? t0 : t1;
#endif
        }       
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE int32_t hmin() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_min_epu32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            int32_t t1 = raw[0] < raw[1] ? raw[0] : raw[1];
            int32_t t2 = raw[2] < raw[3] ? raw[2] : raw[3];
            int32_t t3 = raw[4] < raw[5] ? raw[4] : raw[5];
            int32_t t4 = raw[6] < raw[7] ? raw[6] : raw[7];
            int32_t t5 = raw[8] < raw[9] ? raw[8] : raw[9];
            int32_t t6 = raw[10] < raw[11] ? raw[10] : raw[11];
            int32_t t7 = raw[12] < raw[13] ? raw[12] : raw[13];
            int32_t t8 = raw[14] < raw[15] ? raw[14] : raw[15];

            int32_t t9 = t1 < t2 ? t1 : t2;
            int32_t t10 = t3 < t4 ? t3 : t4;
            int32_t t11 = t5 < t6 ? t5 : t6;
            int32_t t12 = t7 < t8 ? t7 : t8;

            int32_t t13 = t9 < t10 ? t9 : t10;
            int32_t t14 = t10 < t12 ? t11 : t12;

            return t13 < t14 ? t13 : t14;
#else
            int32_t t0 = _mm512_reduce_min_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_min_epi32(mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }       
        // MHMIN
        UME_FORCE_INLINE int32_t hmin(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 =  ((mask.mMask & 0x00000001) != 0) ? raw[0] : std::numeric_limits<int32_t>::max();
            int32_t t1 = (((mask.mMask & 0x00000002) != 0) && raw[1] < t0) ? raw[1] : t0;
            int32_t t2 = (((mask.mMask & 0x00000004) != 0) && raw[2] < t1) ? raw[2] : t1;
            int32_t t3 = (((mask.mMask & 0x00000008) != 0) && raw[3] < t2) ? raw[3] : t2;
            int32_t t4 = (((mask.mMask & 0x00000010) != 0) && raw[4] < t3) ? raw[4] : t3;
            int32_t t5 = (((mask.mMask & 0x00000020) != 0) && raw[5] < t4) ? raw[5] : t4;
            int32_t t6 = (((mask.mMask & 0x00000040) != 0) && raw[6] < t5) ? raw[6] : t5;
            int32_t t7 = (((mask.mMask & 0x00000080) != 0) && raw[7] < t6) ? raw[7] : t6;
            int32_t t8 = (((mask.mMask & 0x00000100) != 0) && raw[8] < t7) ? raw[8] : t7;
            int32_t t9 = (((mask.mMask & 0x00000200) != 0) && raw[9] < t8) ? raw[9] : t8;
            int32_t t10 = (((mask.mMask & 0x00000400) != 0) && raw[10] < t9) ? raw[10] : t9;
            int32_t t11 = (((mask.mMask & 0x00000800) != 0) && raw[11] < t10) ? raw[11] : t10;
            int32_t t12 = (((mask.mMask & 0x00001000) != 0) && raw[12] < t11) ? raw[12] : t11;
            int32_t t13 = (((mask.mMask & 0x00002000) != 0) && raw[13] < t12) ? raw[13] : t12;
            int32_t t14 = (((mask.mMask & 0x00004000) != 0) && raw[14] < t13) ? raw[14] : t13;
            int32_t t15 = (((mask.mMask & 0x00008000) != 0) && raw[15] < t14) ? raw[15] : t14;
            int32_t t16 = (((mask.mMask & 0x00010000) != 0) && raw[16] < t15) ? raw[16] : t15;
            int32_t t17 = (((mask.mMask & 0x00020000) != 0) && raw[17] < t16) ? raw[17] : t16;
            int32_t t18 = (((mask.mMask & 0x00040000) != 0) && raw[18] < t17) ? raw[18] : t17;
            int32_t t19 = (((mask.mMask & 0x00080000) != 0) && raw[19] < t18) ? raw[19] : t18;
            int32_t t20 = (((mask.mMask & 0x00100000) != 0) && raw[20] < t19) ? raw[20] : t19;
            int32_t t21 = (((mask.mMask & 0x00200000) != 0) && raw[21] < t20) ? raw[21] : t20;
            int32_t t22 = (((mask.mMask & 0x00400000) != 0) && raw[22] < t21) ? raw[22] : t21;
            int32_t t23 = (((mask.mMask & 0x00800000) != 0) && raw[23] < t22) ? raw[23] : t22;
            int32_t t24 = (((mask.mMask & 0x01000000) != 0) && raw[24] < t23) ? raw[24] : t23;
            int32_t t25 = (((mask.mMask & 0x02000000) != 0) && raw[25] < t24) ? raw[25] : t24;
            int32_t t26 = (((mask.mMask & 0x04000000) != 0) && raw[26] < t25) ? raw[26] : t25;
            int32_t t27 = (((mask.mMask & 0x08000000) != 0) && raw[27] < t26) ? raw[27] : t26;
            int32_t t28 = (((mask.mMask & 0x10000000) != 0) && raw[28] < t27) ? raw[28] : t27;
            int32_t t29 = (((mask.mMask & 0x20000000) != 0) && raw[29] < t28) ? raw[29] : t28;
            int32_t t30 = (((mask.mMask & 0x40000000) != 0) && raw[30] < t29) ? raw[30] : t29;
            int32_t t31 = (((mask.mMask & 0x80000000) != 0) && raw[31] < t30) ? raw[31] : t30;
            return t31;
#else
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_min_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_min_epi32(m1, mVec[1]);
            return t0 < t1 ? t0 : t1;
#endif
        }       
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_and_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec[0], t0);
            __m512i t2 = _mm512_and_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec[0] = _mm512_and_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_and_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_and_epi32(mVec[0], t0);
            mVec[1] = _mm512_and_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (int32_t b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_or_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec[0], t0);
            __m512i t2 = _mm512_or_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec[0] = _mm512_or_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_or_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_or_epi32(mVec[0], t0);
            mVec[1] = _mm512_or_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_xor_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec[0], t0);
            __m512i t2 = _mm512_xor_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec[0] = _mm512_xor_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_xor_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_xor_epi32(mVec[0], t0);
            mVec[1] = _mm512_xor_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_andnot_epi32(mVec[0], t0);
            __m512i t2 = _mm512_andnot_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator! () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_andnot_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm512_andnot_epi32(mVec[0], t0);
            mVec[1] = _mm512_andnot_epi32(mVec[1], t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i bnota(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm512_mask_andnot_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_andnot_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE int32_t hband() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_and_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  & raw[1]  & raw[2]  & raw[3]  & raw[4]  & raw[5]  & raw[6]  & raw[7] &
                   raw[8]  & raw[9]  & raw[10] & raw[11] & raw[12] & raw[13] & raw[14] & raw[15];
#else
            int32_t t0 = _mm512_reduce_and_epi32(mVec[0]);
            t0 &= _mm512_reduce_and_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBAND
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = 0xFFFFFFFF;
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
            int32_t t0 = _mm512_mask_reduce_and_epi32(m0, mVec[0]);
            t0 &= _mm512_mask_reduce_and_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBANDS
        UME_FORCE_INLINE int32_t hband(int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_and_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b & raw[0]  & raw[1]  & raw[2]  & raw[3]  & raw[4]  & raw[5]  & raw[6]  & raw[7] &
                       raw[8]  & raw[9]  & raw[10] & raw[11] & raw[12] & raw[13] & raw[14] & raw[15];
#else
            int32_t t0 = b;
            t0 &= _mm512_reduce_and_epi32(mVec[0]);
            t0 &= _mm512_reduce_and_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBANDS
        UME_FORCE_INLINE int32_t hband(SIMDVecMask<32> const & mask, int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
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
            int32_t t0 = b;
            t0 &= _mm512_mask_reduce_and_epi32(m0, mVec[0]);
            t0 &= _mm512_mask_reduce_and_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBOR
        UME_FORCE_INLINE int32_t hbor() const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_or_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return raw[0]  | raw[1]  | raw[2]  | raw[3]  | raw[4]  | raw[5]  | raw[6]  | raw[7] |
                   raw[8]  | raw[9]  | raw[10] | raw[11] | raw[12] | raw[13] | raw[14] | raw[15];
#else
            int32_t t0 = _mm512_reduce_or_epi32(mVec[0]);
            t0 |= _mm512_reduce_or_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBOR
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<32> const & mask) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = 0;
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
            int32_t t0 = _mm512_mask_reduce_or_epi32(m0, mVec[0]);
            t0 |= _mm512_mask_reduce_or_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBORS
        UME_FORCE_INLINE int32_t hbor(int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_or_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            return b | raw[0]  | raw[1]  | raw[2]  | raw[3]  | raw[4]  | raw[5]  | raw[6]  | raw[7] |
                       raw[8]  | raw[9]  | raw[10] | raw[11] | raw[12] | raw[13] | raw[14] | raw[15];
#else
            int32_t t0 = b;
            t0 |= _mm512_reduce_or_epi32(mVec[0]);
            t0 |= _mm512_reduce_or_epi32(mVec[1]);
            return t0;
#endif
        }
        // MHBORS
        UME_FORCE_INLINE int32_t hbor(SIMDVecMask<32> const & mask, int32_t b) const {
#if defined (WA_GCC_INTR_SUPPORT_6_4)
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = b;
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
            int32_t t0 = b;
            t0 |= _mm512_mask_reduce_or_epi32(m0, mVec[0]);
            t0 |= _mm512_mask_reduce_or_epi32(m1, mVec[1]);
            return t0;
#endif
        }
        // HBXOR
        UME_FORCE_INLINE int32_t hbxor() const {
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_xor_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            int32_t t1 = 0;
            for (int i = 0; i < 16; i++) {
                t1 ^= raw[i];
            }
            return t1;
        }
        // MHBXOR
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<32> const & mask) const {
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = 0;
            for (int i = 0; i < 32; i++) {
                if ((mask.mMask & (1 << i)) != 0) t0 ^= raw[i];
            }
            return t0;
        }
        // HBXORS
        UME_FORCE_INLINE int32_t hbxor(int32_t b) const {
            alignas(64) int32_t raw[16];
            __m512i t0 = _mm512_xor_epi32(mVec[0], mVec[1]);
            _mm512_store_si512((__m512i*)raw, t0);
            int32_t t1 = 0;
            for (int i = 0; i < 16; i++) {
                t1 ^= raw[i];
            }
            return b ^ t1;
        }
        // MHBXORS
        UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<32> const & mask, int32_t b) const {
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = b;
            for (int i = 0; i < 32; i++) {
                if ((mask.mMask & (1 << i)) != 0) t0 ^= raw[i];
            }
            return t0;
        }
        // GATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(int32_t const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            mVec[0] = _mm512_i32gather_epi32(t3, baseAddr, 4);
            mVec[1] = _mm512_i32gather_epi32(t4, baseAddr, 4);
            return *this;
        }
        // MGATHERU
        UME_FORCE_INLINE SIMDVec_i & gatheru(SIMDVecMask<32> const & mask, int32_t const * baseAddr, uint32_t stride) {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            mVec[0] = _mm512_mask_i32gather_epi32(mVec[0], mask.mMask & 0x0000FFFF, t3, baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_epi32(mVec[1], (mask.mMask >> 16) & 0x0000FFFF, t4, baseAddr, 4);
            return *this;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices+16);
            mVec[0] = _mm512_i32gather_epi32(t0, baseAddr, 4);
            mVec[1] = _mm512_i32gather_epi32(t1, baseAddr, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<32> const & mask, int32_t const * baseAddr, uint32_t const * indices) {
            __m512i t0 = _mm512_loadu_si512(indices);
            __m512i t1 = _mm512_loadu_si512(indices+16);
            mVec[0] = _mm512_mask_i32gather_epi32(mVec[0], mask.mMask & 0x0000FFFF, t0, baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_epi32(mVec[1], (mask.mMask >> 16) & 0x0000FFFF, t1, baseAddr, 4);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            mVec[0] = _mm512_i32gather_epi32(indices.mVec[0], baseAddr, 4);
            mVec[1] = _mm512_i32gather_epi32(indices.mVec[1], baseAddr, 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<32> const & mask, int32_t const * baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            mVec[0] = _mm512_mask_i32gather_epi32(mVec[0], mask.mMask & 0x0000FFFF, indices.mVec[0], baseAddr, 4);
            mVec[1] = _mm512_mask_i32gather_epi32(mVec[1], (mask.mMask >> 16) & 0x0000FFFF, indices.mVec[1], baseAddr, 4);
            return *this;
        }
        // SCATTERU
        UME_FORCE_INLINE int32_t* scatteru(int32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            _mm512_i32scatter_epi32(baseAddr, t3, mVec[0], 4);
            _mm512_i32scatter_epi32(baseAddr, t4, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERU
        UME_FORCE_INLINE int32_t*  scatteru(SIMDVecMask<32> const & mask, int32_t* baseAddr, uint32_t stride) const {
            __m512i t0 = _mm512_set1_epi32(stride);
            __m512i t1 = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            __m512i t2 = _mm512_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
            __m512i t3 = _mm512_mullo_epi32(t0, t1);
            __m512i t4 = _mm512_mullo_epi32(t0, t2);
            _mm512_mask_i32scatter_epi32(baseAddr, mask.mMask & 0x0000FFFF, t3, mVec[0], 4);
            _mm512_mask_i32scatter_epi32(baseAddr, (mask.mMask >> 16) & 0x0000FFFF, t4, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERS
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, uint32_t* indices) {
            __m512i t0 = _mm512_loadu_si512((__m512i *) indices);
            __m512i t1 = _mm512_loadu_si512((__m512i *) (indices + 16));
            _mm512_i32scatter_epi32(baseAddr, t0, mVec[0], 4);
            _mm512_i32scatter_epi32(baseAddr, t1, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<32> const & mask, int32_t* baseAddr, uint32_t* indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), m0, (__m512i *) indices);
            __m512i t1 = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), m1, (__m512i *) (indices + 16));
            _mm512_mask_i32scatter_epi32(baseAddr, m0, t0, mVec[0], 4);
            _mm512_mask_i32scatter_epi32(baseAddr, m1, t1, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            _mm512_i32scatter_epi32(baseAddr, indices.mVec[0], mVec[0], 4);
            _mm512_i32scatter_epi32(baseAddr, indices.mVec[1], mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<32> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_i32scatter_epi32(baseAddr, m0, indices.mVec[0], mVec[0], 4);
            _mm512_mask_i32scatter_epi32(baseAddr, m1, indices.mVec[1], mVec[1], 4);
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
            __m512i t0 = _mm512_rolv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_rolv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MROLV
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ROLS
        UME_FORCE_INLINE SIMDVec_i rol(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_rolv_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MROLS
        UME_FORCE_INLINE SIMDVec_i rol(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // ROLVA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVec_u<uint32_t, 32> const & b) {
            mVec[0] = _mm512_rolv_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_rolv_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MROLVA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ROLSA
        UME_FORCE_INLINE SIMDVec_i & rola(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_rolv_epi32(mVec[0], t0);
            mVec[1] = _mm512_rolv_epi32(mVec[1], t0);
            return *this;
        }
        // MROLSA
        UME_FORCE_INLINE SIMDVec_i & rola(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // RORV
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVec_u<uint32_t, 32> const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_rorv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MRORV
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // RORS
        UME_FORCE_INLINE SIMDVec_i ror(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_rorv_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MRORS
        UME_FORCE_INLINE SIMDVec_i ror(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // RORVA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVec_u<uint32_t, 32> const & b) {
            mVec[0] = _mm512_rorv_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_rorv_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MRORVA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // RORSA
        UME_FORCE_INLINE SIMDVec_i & rora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_rorv_epi32(mVec[0], t0);
            mVec[1] = _mm512_rorv_epi32(mVec[1], t0);
            return *this;
        }
        // MRORSA
        UME_FORCE_INLINE SIMDVec_i & rora(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_sub_epi32(t0, mVec[0]);
            __m512i t2 = _mm512_sub_epi32(t0, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_mask_sub_epi32(mVec[0], m0, t0, mVec[0]);
            __m512i t2 = _mm512_mask_sub_epi32(mVec[1], m1, t0, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            __m512i t0 = _mm512_setzero_epi32();
            mVec[0] = _mm512_sub_epi32(t0, mVec[0]);
            mVec[1] = _mm512_sub_epi32(t0, mVec[1]);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_setzero_epi32();
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, t0, mVec[1]);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            __m512i t0 = _mm512_abs_epi32(mVec[0]);
            __m512i t1 = _mm512_abs_epi32(mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_abs_epi32(mVec[0], m0, mVec[0]);
            __m512i t1 = _mm512_mask_abs_epi32(mVec[1], m1, mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            mVec[0] = _mm512_abs_epi32(mVec[0]);
            mVec[1] = _mm512_abs_epi32(mVec[1]);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_abs_epi32(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_abs_epi32(mVec[1], m1, mVec[1]);
            return *this;
        }
        // PACK
        UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int32_t, 16> const & a, SIMDVec_i<int32_t, 16> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int32_t, 16> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_i & packhi(SIMDVec_i<int32_t, 16> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_i<int32_t, 16> & a, SIMDVec_i<int32_t, 16> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_i<int32_t, 16> unpacklo() const {
            return SIMDVec_i<int32_t, 16>(mVec[0]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_i<int32_t, 16> unpackhi() const {
            return SIMDVec_i<int32_t, 16>(mVec[1]);
        }

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 32>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 32> () const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 32>() const;

    };

}
}

#endif
