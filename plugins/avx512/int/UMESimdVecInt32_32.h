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

        inline explicit SIMDVec_i(__m512i const & x0, __m512i const & x1) { 
            mVec[0] = x0;
            mVec[1] = x1;
        }
    public:

        constexpr static uint32_t length() { return 32; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline SIMDVec_i(int32_t i) {
            mVec[0] = _mm512_set1_epi32(i);
            mVec[1] = mVec[0]; 
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const * p) {
            mVec[0] = _mm512_load_epi32((void *)p);
            mVec[1] = _mm512_load_epi32((void *)(p + 16));
        }
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0,  int32_t i1,  int32_t i2,  int32_t i3,
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
        inline int32_t extract(uint32_t index) const {
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
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
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
        inline IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>> operator() (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>> operator[] (SIMDVecMask<32> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<32>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif
        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        inline SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mov_epi32(mVec[0], m0, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_epi32(mVec[1], m1, b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t b) {
            mVec[0] = _mm512_set1_epi32(b);
            mVec[1] = mVec[0];
            return *this;
        }
        inline SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<32> const & mask, int32_t b) {
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
        inline SIMDVec_i & load(int32_t const * p) {
            mVec[0] = _mm512_loadu_si512(p);
            mVec[1] = _mm512_loadu_si512(p + 16);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<32> const & mask, int32_t const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_loadu_epi32(mVec[0], m0, p);
            mVec[1] = _mm512_mask_loadu_epi32(mVec[1], m1, p + 16);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec[0] = _mm512_load_si512((__m512i*)p);
            mVec[1] = _mm512_load_si512((__m512i*)(p + 16));
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<32> const & mask, int32_t const * p) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_load_epi32(mVec[0], m0, p);
            mVec[1] = _mm512_mask_load_epi32(mVec[1], m1, p + 16);
            return *this;
        }
        // STORE
        inline int32_t * store(int32_t * p) const {
            _mm512_storeu_si512(p, mVec[0]);
            _mm512_storeu_si512(p + 16, mVec[1]);
            return p;
        }
        // MSTORE
        inline int32_t * store(SIMDVecMask<32> const & mask, int32_t * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_storeu_epi32(p, m0, mVec[0]);
            _mm512_mask_storeu_epi32(p + 16, m1, mVec[1]);
            return p;
        }
        // STOREA
        inline int32_t * storea(int32_t * p) {
            _mm512_store_si512((__m512i*)p, mVec[0]);
            _mm512_store_si512((__m512i*)(p + 16), mVec[1]);
            return p;
        }
        // MSTOREA
        inline int32_t * storea(SIMDVecMask<32> const & mask, int32_t * p) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_store_epi32(p,      m0, mVec[0]);
            _mm512_mask_store_epi32(p + 16, m1, mVec[1]);
            return p;
        }
        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mov_epi32(mVec[0], m0, b.mVec[0]);
            __m512i t1 = _mm512_mask_mov_epi32(mVec[1], m1, b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<32> const & mask, int32_t b) const {
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
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_add_epi32(mVec[0], t0);
            __m512i t2 = _mm512_add_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec[0] = _mm512_add_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA 
        inline SIMDVec_i & adda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<32> const & mask, int32_t b) {
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
        inline SIMDVec_i postinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<32> const & mask) {
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
        inline SIMDVec_i & prefinc() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_add_epi32(mVec[0], t0);
            mVec[1] = _mm512_add_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_sub_epi32(mVec[0], t0);
            __m512i t2 = _mm512_sub_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec[0] = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<32> const & mask, int32_t b) {
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
        inline SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi32(b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_sub_epi32(b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(b.mVec[0], m0, b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            __m512i t0 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[0]);
            __m512i t1 = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_sub_epi32(t0, m0, t0, mVec[0]);
            __m512i t2 = _mm512_mask_sub_epi32(t0, m1, t0, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec[0] = _mm512_sub_epi32(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_epi32(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_sub_epi32(b.mVec[0], m0, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(b.mVec[1], m1, b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            mVec[0] = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[0]);
            mVec[1] = _mm512_sub_epi32(_mm512_set1_epi32(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i subfroma(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_sub_epi32(t0, m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(t0, m1, t0, mVec[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_i postdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<32> const & mask) {
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
        inline SIMDVec_i & prefdec() {
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_sub_epi32(mVec[0], t0);
            mVec[1] = _mm512_sub_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(1);
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mullo_epi32(mVec[0], t0);
            __m512i t2 = _mm512_mullo_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec[0] = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mullo_epi32(mVec[0], t0);
            mVec[1] = _mm512_mullo_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<32> const & mask, int32_t b) {
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
        inline SIMDVecMask<32> cmpeq(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<32> cmpeq(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<32> cmpne(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator!=(SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<32> cmpne(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpneq_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpneq_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator!=(int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<32> cmpgt(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpgt_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<32> cmpgt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpgt_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpgt_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<32> cmplt(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmplt_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<32> cmplt(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmplt_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmplt_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<32> cmpge(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmpge_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<32> cmpge(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpge_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpge_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<32> cmple(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmple_epi32_mask(mVec[1], b.mVec[1]);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<32> cmple(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmple_epi32_mask(mVec[1], t0);
            __mmask32 m2 = m0 | (m1 << 16);
            SIMDVecMask<32> ret_mask;
            ret_mask.mMask = m2;
            return ret_mask;
        }
        inline SIMDVecMask<32> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            __mmask16 m0 = _mm512_cmple_epi32_mask(mVec[0], b.mVec[0]);
            __mmask16 m1 = _mm512_cmple_epi32_mask(mVec[1], b.mVec[1]);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __mmask16 m0 = _mm512_cmpeq_epi32_mask(mVec[0], t0);
            __mmask16 m1 = _mm512_cmpeq_epi32_mask(mVec[1], t0);
            return (m0 == 0xFFFF) && (m1 == 0xFFFF);
        }
        // UNIQUE
        inline bool unique() const {
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
        inline int32_t hadd() const {
            int32_t t0 = _mm512_reduce_add_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_add_epi32(mVec[1]);
            return t0 + t1;
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 + t1;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            int32_t t0 = _mm512_reduce_add_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_add_epi32(mVec[1]);
            return t0 + t1 + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 + t1 + b;
        }
        // HMUL
        inline int32_t hmul() const {
            int32_t t0 = _mm512_reduce_mul_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_mul_epi32(mVec[1]);
            return t0 * t1;
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return t0 * t1;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            int32_t t0 = _mm512_reduce_mul_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_mul_epi32(mVec[1]);
            return b * t0 * t1;
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_add_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_add_epi32(m1, mVec[1]);
            return b * t0 * t1;
        }
        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_add_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_add_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_add_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_add_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // FMULSUBV
        inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_mullo_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_sub_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_sub_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFMULSUBV
        inline SIMDVec_i fmulsub(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_mullo_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_sub_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_sub_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // FADDMULV
        inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_add_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mullo_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_mullo_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFADDMULV
        inline SIMDVec_i faddmul(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_add_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // FSUBMULV
        inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m512i t0 = _mm512_sub_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi32(mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mullo_epi32(t0, c.mVec[0]);
            __m512i t3 = _mm512_mullo_epi32(t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MFSUBMULV
        inline SIMDVec_i fsubmul(SIMDVecMask<32> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_sub_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            __m512i t2 = _mm512_mask_mullo_epi32(t0, m0, t0, c.mVec[0]);
            __m512i t3 = _mm512_mask_mullo_epi32(t1, m1, t1, c.mVec[1]);
            return SIMDVec_i(t2, t3);
        }
        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_max_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_max_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_max_epi32(mVec[0], t0);
            __m512i t2 = _mm512_max_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec[0] = _mm512_max_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_max_epi32(mVec[0], t0);
            mVec[1] = _mm512_max_epi32(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_max_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_max_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_min_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_min_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_min_epi32(mVec[0], t0);
            __m512i t2 = _mm512_min_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec[0] = _mm512_min_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_min_epi32(mVec[0], t0);
            mVec[1] = _mm512_min_epi32(mVec[1], t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_min_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_min_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HMAX
        inline int32_t hmax() const {
            int32_t t0 = _mm512_reduce_max_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_max_epi32(mVec[1]);
            return t0 > t1 ? t0 : t1;
        }       
        // MHMAX
        inline int32_t hmax(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_max_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_max_epi32(m1, mVec[1]);
            return t0 > t1 ? t0 : t1;
        }       
        // IMAX
        // MIMAX
        // HMIN
        inline int32_t hmin() const {
            int32_t t0 = _mm512_reduce_min_epi32(mVec[0]);
            int32_t t1 = _mm512_reduce_min_epi32(mVec[1]);
            return t0 < t1 ? t0 : t1;
        }       
        // MHMIN
        inline int32_t hmin(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_min_epi32(m0, mVec[0]);
            int32_t t1 = _mm512_mask_reduce_min_epi32(m1, mVec[1]);
            return t0 < t1 ? t0 : t1;
        }       
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_and_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_and_epi32(mVec[0], t0);
            __m512i t2 = _mm512_and_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec[0] = _mm512_and_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_and_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_and_epi32(mVec[0], t0);
            mVec[1] = _mm512_and_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator&= (int32_t b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_and_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_and_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_or_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_or_epi32(mVec[0], t0);
            __m512i t2 = _mm512_or_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec[0] = _mm512_or_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_or_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_or_epi32(mVec[0], t0);
            mVec[1] = _mm512_or_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_or_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_or_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_xor_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<32> const & mask, SIMDVec_i const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_xor_epi32(mVec[0], t0);
            __m512i t2 = _mm512_xor_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec[0] = _mm512_xor_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_xor_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<32> const & mask, SIMDVec_i const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_xor_epi32(mVec[0], t0);
            mVec[1] = _mm512_xor_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<32> const & mask, int32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_xor_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_xor_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_andnot_epi32(mVec[0], t0);
            __m512i t2 = _mm512_andnot_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator! () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            __m512i t1 = _mm512_mask_andnot_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_andnot_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm512_andnot_epi32(mVec[0], t0);
            mVec[1] = _mm512_andnot_epi32(mVec[1], t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i bnota(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm512_mask_andnot_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_andnot_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            int32_t t0 = _mm512_reduce_and_epi32(mVec[0]);
            t0 &= _mm512_reduce_and_epi32(mVec[1]);
            return t0;
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_and_epi32(m0, mVec[0]);
            t0 &= _mm512_mask_reduce_and_epi32(m1, mVec[1]);
            return t0;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            int32_t t0 = b;
            t0 &= _mm512_reduce_and_epi32(mVec[0]);
            t0 &= _mm512_reduce_and_epi32(mVec[1]);
            return t0;
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = b;
            t0 &= _mm512_mask_reduce_and_epi32(m0, mVec[0]);
            t0 &= _mm512_mask_reduce_and_epi32(m1, mVec[1]);
            return t0;
        }
        // HBOR
        inline int32_t hbor() const {
            int32_t t0 = _mm512_reduce_or_epi32(mVec[0]);
            t0 |= _mm512_reduce_or_epi32(mVec[1]);
            return t0;
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = _mm512_mask_reduce_or_epi32(m0, mVec[0]);
            t0 |= _mm512_mask_reduce_or_epi32(m1, mVec[1]);
            return t0;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            int32_t t0 = b;
            t0 |= _mm512_reduce_or_epi32(mVec[0]);
            t0 |= _mm512_reduce_or_epi32(mVec[1]);
            return t0;
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<32> const & mask, int32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            int32_t t0 = b;
            t0 |= _mm512_mask_reduce_or_epi32(m0, mVec[0]);
            t0 |= _mm512_mask_reduce_or_epi32(m1, mVec[1]);
            return t0;
        }
        // HBXOR
        inline int32_t hbxor() const {
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = 0;
            for (int i = 0; i < 32; i++) {
                t0 ^= raw[i];
            }
            return t0;
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
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
        inline int32_t hbxor(int32_t b) const {
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = b;
            for (int i = 0; i < 32; i++) {
                t0 ^= raw[i];
            }
            return t0;
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<32> const & mask, int32_t b) const {
            alignas(64) int32_t raw[32];
            _mm512_store_si512((__m512i*)raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 16), mVec[1]);
            int32_t t0 = b;
            for (int i = 0; i < 32; i++) {
                if ((mask.mMask & (1 << i)) != 0) t0 ^= raw[i];
            }
            return t0;
        }
        // GATHERS
        inline SIMDVec_i & gather(int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t raw[32] = { 
                baseAddr[indices[0]], baseAddr[indices[1]],
                baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]],
                baseAddr[indices[6]], baseAddr[indices[7]],
                baseAddr[indices[8]], baseAddr[indices[9]],
                baseAddr[indices[10]], baseAddr[indices[11]],
                baseAddr[indices[12]], baseAddr[indices[13]],
                baseAddr[indices[14]], baseAddr[indices[15]],
                baseAddr[indices[16]], baseAddr[indices[17]],
                baseAddr[indices[18]], baseAddr[indices[19]],
                baseAddr[indices[20]], baseAddr[indices[21]],
                baseAddr[indices[22]], baseAddr[indices[23]],
                baseAddr[indices[24]], baseAddr[indices[25]],
                baseAddr[indices[26]], baseAddr[indices[27]],
                baseAddr[indices[28]], baseAddr[indices[29]],
                baseAddr[indices[30]], baseAddr[indices[31]]};
            mVec[0] = _mm512_load_si512((__m512i*)raw);
            mVec[1] = _mm512_load_si512((__m512i*)(raw + 16));
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<32> const & mask, int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t raw[32] = {
                baseAddr[indices[0]], baseAddr[indices[1]],
                baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]],
                baseAddr[indices[6]], baseAddr[indices[7]],
                baseAddr[indices[8]], baseAddr[indices[9]],
                baseAddr[indices[10]], baseAddr[indices[11]],
                baseAddr[indices[12]], baseAddr[indices[13]],
                baseAddr[indices[14]], baseAddr[indices[15]],
                baseAddr[indices[16]], baseAddr[indices[17]] ,
                baseAddr[indices[18]], baseAddr[indices[19]] ,
                baseAddr[indices[20]], baseAddr[indices[21]] ,
                baseAddr[indices[22]], baseAddr[indices[23]] ,
                baseAddr[indices[24]], baseAddr[indices[25]] ,
                baseAddr[indices[26]], baseAddr[indices[27]] ,
                baseAddr[indices[28]], baseAddr[indices[29]] ,
                baseAddr[indices[30]], baseAddr[indices[31]] };
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            alignas(64) int32_t rawIndices[32];
            alignas(64) int32_t rawData[32];
            _mm512_store_si512((__m512i*) rawIndices, indices.mVec[0]);
            _mm512_store_si512((__m512i*) (rawIndices + 16), indices.mVec[1]);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            rawData[4] = baseAddr[rawIndices[4]];
            rawData[5] = baseAddr[rawIndices[5]];
            rawData[6] = baseAddr[rawIndices[6]];
            rawData[7] = baseAddr[rawIndices[7]];
            rawData[8] = baseAddr[rawIndices[8]];
            rawData[9] = baseAddr[rawIndices[9]];
            rawData[10] = baseAddr[rawIndices[10]];
            rawData[11] = baseAddr[rawIndices[11]];
            rawData[12] = baseAddr[rawIndices[12]];
            rawData[13] = baseAddr[rawIndices[13]];
            rawData[14] = baseAddr[rawIndices[14]];
            rawData[15] = baseAddr[rawIndices[15]];
            rawData[16] = baseAddr[rawIndices[16]];
            rawData[17] = baseAddr[rawIndices[17]];
            rawData[18] = baseAddr[rawIndices[18]];
            rawData[19] = baseAddr[rawIndices[19]];
            rawData[20] = baseAddr[rawIndices[20]];
            rawData[21] = baseAddr[rawIndices[21]];
            rawData[22] = baseAddr[rawIndices[22]];
            rawData[23] = baseAddr[rawIndices[23]];
            rawData[24] = baseAddr[rawIndices[24]];
            rawData[25] = baseAddr[rawIndices[25]];
            rawData[26] = baseAddr[rawIndices[26]];
            rawData[27] = baseAddr[rawIndices[27]];
            rawData[28] = baseAddr[rawIndices[28]];
            rawData[29] = baseAddr[rawIndices[29]];
            rawData[30] = baseAddr[rawIndices[30]];
            rawData[31] = baseAddr[rawIndices[31]];
            mVec[0] = _mm512_load_si512((__m512i*)rawData);
            mVec[1] = _mm512_load_si512((__m512i*)(rawData + 16));
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<32> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            alignas(64) int32_t rawIndices[32];
            alignas(64) int32_t rawData[32];
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_store_si512((__m512i*) rawIndices, indices.mVec[0]);
            _mm512_store_si512((__m512i*) (rawIndices + 16), indices.mVec[1]);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            rawData[4] = baseAddr[rawIndices[4]];
            rawData[5] = baseAddr[rawIndices[5]];
            rawData[6] = baseAddr[rawIndices[6]];
            rawData[7] = baseAddr[rawIndices[7]];
            rawData[8] = baseAddr[rawIndices[8]];
            rawData[9] = baseAddr[rawIndices[9]];
            rawData[10] = baseAddr[rawIndices[10]];
            rawData[11] = baseAddr[rawIndices[11]];
            rawData[12] = baseAddr[rawIndices[12]];
            rawData[13] = baseAddr[rawIndices[13]];
            rawData[14] = baseAddr[rawIndices[14]];
            rawData[15] = baseAddr[rawIndices[15]];
            rawData[16] = baseAddr[rawIndices[16]];
            rawData[17] = baseAddr[rawIndices[17]];
            rawData[18] = baseAddr[rawIndices[18]];
            rawData[19] = baseAddr[rawIndices[19]];
            rawData[20] = baseAddr[rawIndices[20]];
            rawData[21] = baseAddr[rawIndices[21]];
            rawData[22] = baseAddr[rawIndices[22]];
            rawData[23] = baseAddr[rawIndices[23]];
            rawData[24] = baseAddr[rawIndices[24]];
            rawData[25] = baseAddr[rawIndices[25]];
            rawData[26] = baseAddr[rawIndices[26]];
            rawData[27] = baseAddr[rawIndices[27]];
            rawData[28] = baseAddr[rawIndices[28]];
            rawData[29] = baseAddr[rawIndices[29]];
            rawData[30] = baseAddr[rawIndices[30]];
            rawData[31] = baseAddr[rawIndices[31]];
            mVec[0] = _mm512_mask_load_epi32(mVec[0], m0, rawData);
            mVec[1] = _mm512_mask_load_epi32(mVec[1], m1, rawData + 16);
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint32_t* indices) {
            alignas(64) int32_t rawIndices[32] = { 
                indices[0],  indices[1],  indices[2],  indices[3],
                indices[4],  indices[5],  indices[6],  indices[7],
                indices[8],  indices[9],  indices[10], indices[11],
                indices[12], indices[13], indices[14], indices[15],
                indices[16], indices[17], indices[18], indices[19],
                indices[20], indices[21], indices[22], indices[23],
                indices[24], indices[25], indices[26], indices[27],
                indices[28], indices[29], indices[30], indices[31] };
            __m512i t0 = _mm512_load_si512((__m512i *) rawIndices);
            __m512i t1 = _mm512_load_si512((__m512i *)rawIndices);
            _mm512_i32scatter_epi32(baseAddr, t0, mVec[0], 4);
            _mm512_i32scatter_epi32(baseAddr, t1, mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<32> const & mask, int32_t* baseAddr, uint32_t* indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            alignas(64) int32_t rawIndices[32] = { 
                indices[0], indices[1], indices[2], indices[3],
                indices[4], indices[5], indices[6], indices[7],
                indices[8],  indices[9],  indices[10], indices[11],
                indices[12], indices[13], indices[14], indices[15],
                indices[16], indices[17], indices[18], indices[19],
                indices[20], indices[21], indices[22], indices[23],
                indices[24], indices[25], indices[26], indices[27],
                indices[28], indices[29], indices[30], indices[31] };
            __m512i t0 = _mm512_mask_load_epi32(_mm512_set1_epi32(0), m0, (__m512i *) rawIndices);
            __m512i t1 = _mm512_mask_load_epi32(_mm512_set1_epi32(0), m1, (__m512i *) (rawIndices + 16));
            _mm512_mask_i32scatter_epi32(baseAddr, m0, t0, mVec[0], 4);
            _mm512_mask_i32scatter_epi32(baseAddr, m1, t1, mVec[1], 4);
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            _mm512_i32scatter_epi32(baseAddr, indices.mVec[0], mVec[0], 4);
            _mm512_i32scatter_epi32(baseAddr + 16, indices.mVec[1], mVec[1], 4);
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<32> const & mask, int32_t* baseAddr, SIMDVec_u<uint32_t, 32> const & indices) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            _mm512_mask_i32scatter_epi32(baseAddr,      m0, indices.mVec[0], mVec[0], 4);
            _mm512_mask_i32scatter_epi32(baseAddr + 16, m1, indices.mVec[1], mVec[1], 4);
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
        inline SIMDVec_i rol(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_rolv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_rolv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MROLV
        inline SIMDVec_i rol(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ROLS
        inline SIMDVec_i rol(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rolv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_rolv_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MROLS
        inline SIMDVec_i rol(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // ROLVA
        inline SIMDVec_i & rola(SIMDVec_u<uint32_t, 32> const & b) {
            mVec[0] = _mm512_rolv_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_rolv_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MROLVA
        inline SIMDVec_i & rola(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ROLSA
        inline SIMDVec_i & rola(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_rolv_epi32(mVec[0], t0);
            mVec[1] = _mm512_rolv_epi32(mVec[1], t0);
            return *this;
        }
        // MROLSA
        inline SIMDVec_i & rola(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_rolv_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_rolv_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // RORV
        inline SIMDVec_i ror(SIMDVec_u<uint32_t, 32> const & b) const {
            __m512i t0 = _mm512_rorv_epi32(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_rorv_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MRORV
        inline SIMDVec_i ror(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // RORS
        inline SIMDVec_i ror(uint32_t b) const {
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_rorv_epi32(mVec[0], t0);
            __m512i t2 = _mm512_rorv_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MRORS
        inline SIMDVec_i ror(SIMDVecMask<32> const & mask, uint32_t b) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            __m512i t1 = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], t0);
            __m512i t2 = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // RORVA
        inline SIMDVec_i & rora(SIMDVec_u<uint32_t, 32> const & b) {
            mVec[0] = _mm512_rorv_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_rorv_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MRORVA
        inline SIMDVec_i & rora(SIMDVecMask<32> const & mask, SIMDVec_u<uint32_t, 32> const & b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], b.mVec[1]);
            return *this;
        }
        // RORSA
        inline SIMDVec_i & rora(uint32_t b) {
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_rorv_epi32(mVec[0], t0);
            mVec[1] = _mm512_rorv_epi32(mVec[1], t0);
            return *this;
        }
        // MRORSA
        inline SIMDVec_i & rora(SIMDVecMask<32> const & mask, uint32_t b) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_set1_epi32(b);
            mVec[0] = _mm512_mask_rorv_epi32(mVec[0], m0, mVec[0], t0);
            mVec[1] = _mm512_mask_rorv_epi32(mVec[1], m1, mVec[1], t0);
            return *this;
        }
        // NEG
        inline SIMDVec_i neg() const {
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_sub_epi32(t0, mVec[0]);
            __m512i t2 = _mm512_sub_epi32(t0, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_i neg(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_setzero_epi32();
            __m512i t1 = _mm512_mask_sub_epi32(mVec[0], m0, t0, mVec[0]);
            __m512i t2 = _mm512_mask_sub_epi32(mVec[1], m1, t0, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        // NEGA
        inline SIMDVec_i & nega() {
            __m512i t0 = _mm512_setzero_epi32();
            mVec[0] = _mm512_sub_epi32(t0, mVec[0]);
            mVec[1] = _mm512_sub_epi32(t0, mVec[1]);
            return *this;
        }
        // MNEGA
        inline SIMDVec_i & nega(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_setzero_epi32();
            mVec[0] = _mm512_mask_sub_epi32(mVec[0], m0, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi32(mVec[1], m1, t0, mVec[1]);
            return *this;
        }
        // ABS
        inline SIMDVec_i abs() const {
            __m512i t0 = _mm512_abs_epi32(mVec[0]);
            __m512i t1 = _mm512_abs_epi32(mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<32> const & mask) const {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            __m512i t0 = _mm512_mask_abs_epi32(mVec[0], m0, mVec[0]);
            __m512i t1 = _mm512_mask_abs_epi32(mVec[1], m1, mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ABSA
        inline SIMDVec_i & absa() {
            mVec[0] = _mm512_abs_epi32(mVec[0]);
            mVec[1] = _mm512_abs_epi32(mVec[1]);
            return *this;
        }
        // MABSA
        inline SIMDVec_i & absa(SIMDVecMask<32> const & mask) {
            __mmask16 m0 = mask.mMask & 0x0000FFFF;
            __mmask16 m1 = (mask.mMask & 0xFFFF0000) >> 16;
            mVec[0] = _mm512_mask_abs_epi32(mVec[0], m0, mVec[0]);
            mVec[1] = _mm512_mask_abs_epi32(mVec[1], m1, mVec[1]);
            return *this;
        }
        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 16> const & a, SIMDVec_i<int32_t, 16> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        inline SIMDVec_i & packlo(SIMDVec_i<int32_t, 16> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        inline SIMDVec_i & packhi(SIMDVec_i<int32_t, 16> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 16> & a, SIMDVec_i<int32_t, 16> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 16> unpacklo() const {
            return SIMDVec_i<int32_t, 16>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 16> unpackhi() const {
            return SIMDVec_i<int32_t, 16>(mVec[1]);
        }

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_i<int16_t, 32>() const;

        // UTOI
        inline operator SIMDVec_u<uint32_t, 32> () const;
        // UTOF
        inline operator SIMDVec_f<float, 32>() const;

    };

}
}

#endif
