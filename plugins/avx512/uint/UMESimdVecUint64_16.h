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

#ifndef UME_SIMD_VEC_UINT64_16_H_
#define UME_SIMD_VEC_UINT64_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

#define EXPAND_CALL_BINARY(a_512i, b_512i, binary_op) \
            _mm512_castsi512_si512( \
                binary_op( \
                    _mm512_castsi512_si512(a_512i), \
                    _mm512_castsi512_si512(b_512i)))

#define EXPAND_CALL_BINARY_MASK(a_512i, b_512i, mask8, binary_op) \
            _mm512_castsi512_si512( \
                binary_op( \
                    _mm512_castsi512_si512(a_512i), \
                    mask8, \
                    _mm512_castsi512_si512(a_512i), \
                    _mm512_castsi512_si512(b_512i)))

#define EXPAND_CALL_BINARY_SCALAR_MASK(a_512i, b_64u, mask8, binary_op) \
            _mm512_castsi512_si512( \
                binary_op( \
                    _mm512_castsi512_si512(a_512i), \
                    mask8, \
                    _mm512_castsi512_si512(a_512i), \
                    _mm512__mm512_set1_epi64(b_64u)))

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint64_t, 16> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint64_t, 16>,
            uint64_t,
            16,
            SIMDVecMask<16>,
            SIMDVecSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint64_t, 16>,
            SIMDVec_u<uint64_t, 8>>
    {
    public:
        friend class SIMDVec_i<int64_t, 16>;
        friend class SIMDVec_f<double, 16>;

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
        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint64_t i) {
            mVec[0] = _mm512_set1_epi64(i);
            mVec[1] = _mm512_set1_epi64(i);
        }
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint64_t const *p) {
            mVec[0] = _mm512_loadu_si512((__m512i*)p);
            mVec[1] = _mm512_loadu_si512((__m512i*)(p + 8));
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i0,  uint64_t i1,  uint64_t i2,  uint64_t i3,
                         uint64_t i4,  uint64_t i5,  uint64_t i6,  uint64_t i7,
                         uint64_t i8,  uint64_t i9,  uint64_t i10, uint64_t i11,
                         uint64_t i12, uint64_t i13, uint64_t i14, uint64_t i15) {
            mVec[0] = _mm512_set_epi64(i7, i6, i5, i4, i3, i2, i1, i0);
            mVec[1] = _mm512_set_epi64(i15, i14, i13, i12, i11, i10, i9, i8);
        }

        // EXTRACT
        UME_FORCE_INLINE uint64_t extract(uint32_t index) const {
            alignas(64) uint64_t raw[16];
            _mm512_store_si512((__m512i*) raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 8), mVec[1]);
            return raw[index];
        }
        UME_FORCE_INLINE uint64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint64_t value) {
            alignas(64) uint64_t raw[16];
            _mm512_store_si512((__m512i*) raw, mVec[0]);
            _mm512_store_si512((__m512i*)(raw + 8), mVec[1]);
            raw[index] = value;
            mVec[0] = _mm512_load_si512((__m512i*) raw);
            mVec[1] = _mm512_load_si512((__m512i*)(raw + 8));
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint64_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
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
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], t0, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], t1, b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint64_t b) {
            mVec[0] = _mm512_set1_epi64(b);
            mVec[1] = _mm512_set1_epi64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<16> const & mask, uint64_t b) {
            __m512i t0 = _mm512_set1_epi64(b);
            __mmask8 t1 = mask.mMask & 0x00FF;
            __mmask8 t2 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], t1, t0);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], t2, t0);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint64_t const *p) {
            mVec[0] = _mm512_loadu_si512((const __m512i *) p);
            mVec[1] = _mm512_loadu_si512((const __m512i *) (p + 8));
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<16> const & mask, uint64_t const *p) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_loadu_epi64(mVec[0], t0, p);
            mVec[1] = _mm512_mask_loadu_epi64(mVec[1], t1, p + 8);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint64_t const *p) {
            mVec[0] = _mm512_load_si512((const __m512i *) p);
            mVec[1] = _mm512_load_si512((const __m512i *) (p + 8));
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<16> const & mask, uint64_t const *p) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_load_epi64(mVec[0], t0, p);
            mVec[1] = _mm512_mask_load_epi64(mVec[1], t1, p + 8);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint64_t* store(uint64_t* p) const {
            _mm512_storeu_si512((__m512i *)p, mVec[0]);
            _mm512_storeu_si512((__m512i *)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint64_t* store(SIMDVecMask<16> const & mask, uint64_t* p) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            _mm512_mask_storeu_epi64(p, t0, mVec[0]);
            _mm512_mask_storeu_epi64(p + 8, t1, mVec[1]);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint64_t* storea(uint64_t* p) const {
            _mm512_store_si512((__m512i *)p, mVec[0]);
            _mm512_store_si512((__m512i *) (p + 8), mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint64_t* storea(SIMDVecMask<16> const & mask, uint64_t* p) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            _mm512_mask_store_epi64(p, t0, mVec[0]);
            _mm512_mask_store_epi64(p + 8, t1, mVec[1]);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_mov_epi64(mVec[0], t0, b.mVec[0]);
            __m512i t3 = _mm512_mask_mov_epi64(mVec[1], t1, b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_mov_epi64(mVec[0], t0, _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_mov_epi64(mVec[1], t1, _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_add_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_add_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_add_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint64_t b) const {
            __m512i t0 = _mm512_add_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_add_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_add_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_add_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec[0] = _mm512_add_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_add_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint64_t b) {
            mVec[0] = _mm512_add_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_add_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_add_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_add_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
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
            __m512i t0 = _mm512_set1_epi64(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_add_epi64(mVec[0], t0);
            mVec[1] = _mm512_add_epi64(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = mVec[0];
            __m512i t1 = mVec[1];
            __mmask8 t2 = mask.mMask & 0x00FF;
            __mmask8 t3 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_add_epi64(mVec[0], t2, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_add_epi64(mVec[1], t3, mVec[1], _mm512_set1_epi64(1));
            return SIMDVec_u(t0, t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m512i t0 = _mm512_set1_epi64(1);
            mVec[0] = _mm512_add_epi64(mVec[0], t0);
            mVec[1] = _mm512_add_epi64(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<16> const & mask) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_add_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_add_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(1));
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_sub_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_sub_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_sub_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint64_t b) const {
            __m512i t0 = _mm512_sub_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_sub_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_sub_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_sub_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec[0] = _mm512_sub_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint64_t b) {
            mVec[0] = _mm512_sub_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_sub_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
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
            __m512i t0 = _mm512_sub_epi64(b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_sub_epi64(b.mVec[1], mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_sub_epi64(b.mVec[0], t0, b.mVec[0], mVec[0]);
            __m512i t3 = _mm512_mask_sub_epi64(b.mVec[1], t1, b.mVec[1], mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint64_t b) const {
            __m512i t0 = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[0]);
            __m512i t1 = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_set1_epi64(b);
            __m512i t3 = _mm512_mask_sub_epi64(t2, t0, t2, mVec[0]);
            __m512i t4 = _mm512_mask_sub_epi64(t2, t1, t2, mVec[1]);
            return SIMDVec_u(t3, t4);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec[0] = _mm512_sub_epi64(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_epi64(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_sub_epi64(b.mVec[0], t0, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(b.mVec[1], t1, b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint64_t b) {
            mVec[0] = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[0]);
            mVec[1] = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_set1_epi64(b);
            mVec[0] = _mm512_mask_sub_epi64(t2, t0, t2, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(t2, t1, t2, mVec[1]);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m512i t0 = _mm512_set1_epi64(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_sub_epi64(mVec[0], t0);
            mVec[1] = _mm512_sub_epi64(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = mVec[0];
            __m512i t1 = mVec[1];
            __mmask8 t2 = mask.mMask & 0x00FF;
            __mmask8 t3 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], t2, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], t3, mVec[1], _mm512_set1_epi64(1));
            return SIMDVec_u(t0, t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            mVec[0] = _mm512_sub_epi64(mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_sub_epi64(mVec[1], _mm512_set1_epi64(1));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<16> const & mask) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(1));
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
#if defined(__AVX512DQ__)
    #if defined(__AVX512VL__)
            __m256i t0 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t3 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t4 = _mm256_mullo_epi64(t0, t2);
            __m256i t5 = _mm256_mullo_epi64(t1, t3);

            __m256i t6 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t9 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t10 = _mm256_mullo_epi64(t6, t8);
            __m256i t11 = _mm256_mullo_epi64(t7, t9);
            __m512i t12 = _mm512_inserti64x4(mVec[0], t4, 0);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t5, 0);
            t12 = _mm512_inserti64x4(t12, t10, 1);
            t13 = _mm512_inserti64x4(t13, t11, 1);
            return SIMDVec_u(t12, t13);
    #else
            __m512i t0 = _mm512_mullo_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
    #endif
#else
            alignas(64) uint64_t raw[16];
            alignas(64) uint64_t raw_b[16];
            alignas(64) uint64_t res[16];
            _mm512_store_si512((__m512i *)raw, mVec[0]);
            _mm512_store_si512((__m512i *)(raw + 8), mVec[1]);

            _mm512_store_si512((__m512i *)raw_b, b.mVec[0]);
            _mm512_store_si512((__m512i *)(raw_b + 8), b.mVec[1]);

            res[0] = raw[0] * raw_b[0];
            res[1] = raw[1] * raw_b[1];
            res[2] = raw[2] * raw_b[2];
            res[3] = raw[3] * raw_b[3];
            res[4] = raw[4] * raw_b[4];
            res[5] = raw[5] * raw_b[5];
            res[6] = raw[6] * raw_b[6];
            res[7] = raw[7] * raw_b[7];
            __m512i t0 = _mm512_load_si512((const __m512i *)res);

            res[8] = raw[8] * raw_b[8];
            res[9] = raw[9] * raw_b[9];
            res[10] = raw[10] * raw_b[10];
            res[11] = raw[11] * raw_b[11];
            res[12] = raw[12] * raw_b[12];
            res[13] = raw[13] * raw_b[13];
            res[14] = raw[14] * raw_b[14];
            res[15] = raw[15] * raw_b[15];
            __m512i t1 = _mm512_load_si512((const __m512i *)(res + 8));

            return SIMDVec_u(t0, t1);
#endif
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t0 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t3 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t4 = _mm256_mask_mullo_epi64(t0, mask.mMask & 0x000F, t0, t2);
            __m256i t5 = _mm256_mask_mullo_epi64(t1, (mask.mMask & 0x00F0) >> 4, t1, t3);

            __m256i t6 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t9 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t10 = _mm256_mask_mullo_epi64(t6, (mask.mMask & 0x0F00) >> 8, t6, t8);
            __m256i t11 = _mm256_mask_mullo_epi64(t7, (mask.mMask & 0xF000) >> 12, t7, t9);
            __m512i t12 = _mm512_inserti64x4(mVec[0], t4, 0);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t5, 0);
            t12 = _mm512_inserti64x4(t12, t10, 1);
            t13 = _mm512_inserti64x4(t13, t11, 1);
            return SIMDVec_u(t12, t13);
#else
            __m512i t0 = _mm512_mask_mullo_epi64(mVec[0], mask.mMask & 0x00FF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi64(mVec[1], (mask.mMask & 0xFF00) >> 8, mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
#endif
#else
            alignas(64) uint64_t raw[16];
            alignas(64) uint64_t raw_b[16];
            alignas(64) uint64_t res[16];
            _mm512_store_si512((__m512i *)raw, mVec[0]);
            _mm512_store_si512((__m512i *)(raw + 8), mVec[1]);

            _mm512_store_si512((__m512i *)raw_b, b.mVec[0]);
            _mm512_store_si512((__m512i *)(raw_b + 8), b.mVec[1]);

            res[0] = mask.mMask & 0x0001 ? raw[0] * raw_b[0] : raw[0];
            res[1] = mask.mMask & 0x0002 ? raw[1] * raw_b[1] : raw[1];
            res[2] = mask.mMask & 0x0004 ? raw[2] * raw_b[2] : raw[2];
            res[3] = mask.mMask & 0x0008 ? raw[3] * raw_b[3] : raw[3];
            res[4] = mask.mMask & 0x0010 ? raw[4] * raw_b[4] : raw[4];
            res[5] = mask.mMask & 0x0020 ? raw[5] * raw_b[5] : raw[5];
            res[6] = mask.mMask & 0x0040 ? raw[6] * raw_b[6] : raw[6];
            res[7] = mask.mMask & 0x0080 ? raw[7] * raw_b[7] : raw[7];
            __m512i t0 = _mm512_load_si512((const __m512i *)res);

            res[8] = mask.mMask & 0x0100 ? raw[8] * raw_b[8] : raw[8];
            res[9] = mask.mMask & 0x0200 ? raw[9] * raw_b[9] : raw[9];
            res[10] = mask.mMask & 0x0400 ? raw[10] * raw_b[10] : raw[10];
            res[11] = mask.mMask & 0x0800 ? raw[11] * raw_b[11] : raw[11];
            res[12] = mask.mMask & 0x1000 ? raw[12] * raw_b[12] : raw[12];
            res[13] = mask.mMask & 0x2000 ? raw[13] * raw_b[13] : raw[13];
            res[14] = mask.mMask & 0x4000 ? raw[14] * raw_b[14] : raw[14];
            res[15] = mask.mMask & 0x8000 ? raw[15] * raw_b[15] : raw[15];
            __m512i t1 = _mm512_load_si512((const __m512i *)(res + 8));

            return SIMDVec_u(t0, t1);
#endif
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)

            __m256i t0 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mullo_epi64(t0, t2);
            __m256i t4 = _mm256_mullo_epi64(t1, t2);
            __m256i t5 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t6 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t7 = _mm256_mullo_epi64(t5, t2);
            __m256i t8 = _mm256_mullo_epi64(t6, t2);
            __m512i t9 = _mm512_inserti64x4(mVec[0], t3, 0);
            __m512i t10 = _mm512_inserti64x4(mVec[1], t3, 0);
            t9 = _mm512_inserti64x4(t9, t7, 1);
            t10 = _mm512_inserti64x4(t10, t8, 1);

            return SIMDVec_u(t9, t10);
#else
            __m512i t0 = _mm512_mullo_epi64(mVec[0], _mm256_set1_epi64(b));
            __m512i t1 = _mm512_mullo_epi64(mVec[1], _mm256_set1_epi64(b));
            return SIMDVec_u(t0, t1);
#endif
#else
            alignas(64) uint64_t raw[16];
            alignas(64) uint64_t res[16];
            _mm512_store_si512((__m512i *)raw, mVec[0]);
            _mm512_store_si512((__m512i *)(raw + 8), mVec[1]);

            res[0] = raw[0] * b;
            res[1] = raw[1] * b;
            res[2] = raw[2] * b;
            res[3] = raw[3] * b;
            res[4] = raw[4] * b;
            res[5] = raw[5] * b;
            res[6] = raw[6] * b;
            res[7] = raw[7] * b;
            __m512i t0 = _mm512_load_si512((const __m512i *)res);

            res[8] = raw[8] * b;
            res[9] = raw[9] * b;
            res[10] = raw[10] * b;
            res[11] = raw[11] * b;
            res[12] = raw[12] * b;
            res[13] = raw[13] * b;
            res[14] = raw[14] * b;
            res[15] = raw[15] * b;
            __m512i t1 = _mm512_load_si512((const __m512i *)(res + 8));

            return SIMDVec_u(t0, t1);
#endif
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<16> const & mask, uint64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)

            __m256i t0 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mask_mullo_epi64(t0, mask.mMask & 0x000F, t0, t2);
            __m256i t4 = _mm256_mask_mullo_epi64(t1, (mask.mMask & 0x0F00 >> 8), t1, t2);
            __m256i t5 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t6 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t7 = _mm256_mask_mullo_epi64(t5, (mask.mMask & 0x00F0 >> 4), t1, t2);
            __m256i t8 = _mm256_mask_mullo_epi64(t6, (mask.mMask & 0xF000 >> 12), t1, t2);
            __m512i t9 = _mm512_inserti64x4(mVec[0], t3, 0);
            __m512i t10 = _mm512_inserti64x4(mVec[1], t3, 0);
            t9 = _mm512_inserti64x4(t9, t7, 1);
            t10 = _mm512_inserti64x4(t10, t8, 1);

            return SIMDVec_u(t9, t10);
#else
            __m512i t0 = _mm512_mask_mullo_epi64(mVec[0], (mask.mMask & 0x00FF), mVec[0], _mm256_set1_epi64(b));
            __m512i t1 = _mm512_mask_mullo_epi64(mVec[1], (mask.mMask & 0xFF00), mVec[1], _mm256_set1_epi64(b));
            return SIMDVec_u(t0, t1);
#endif
#else
            alignas(64) uint64_t raw[16];
            alignas(64) uint64_t res[16];
            _mm512_store_si512((__m512i *)raw, mVec[0]);
            _mm512_store_si512((__m512i *)(raw + 8), mVec[1]);

            res[0] = mask.mMask & 0x0001 ? raw[0] * b : raw[0];
            res[1] = mask.mMask & 0x0002 ? raw[1] * b : raw[1];
            res[2] = mask.mMask & 0x0004 ? raw[2] * b : raw[2];
            res[3] = mask.mMask & 0x0008 ? raw[3] * b : raw[3];
            res[4] = mask.mMask & 0x0010 ? raw[4] * b : raw[4];
            res[5] = mask.mMask & 0x0020 ? raw[5] * b : raw[5];
            res[6] = mask.mMask & 0x0040 ? raw[6] * b : raw[6];
            res[7] = mask.mMask & 0x0080 ? raw[7] * b : raw[7];
            __m512i t0 = _mm512_load_si512((const __m512i *)res);

            res[8] = mask.mMask & 0x0100 ? raw[8] * b : raw[8];
            res[9] = mask.mMask & 0x0200 ? raw[9] * b : raw[9];
            res[10] = mask.mMask & 0x0400 ? raw[10] * b : raw[10];
            res[11] = mask.mMask & 0x0800 ? raw[11] * b : raw[11];
            res[12] = mask.mMask & 0x1000 ? raw[12] * b : raw[12];
            res[13] = mask.mMask & 0x2000 ? raw[13] * b : raw[13];
            res[14] = mask.mMask & 0x4000 ? raw[14] * b : raw[14];
            res[15] = mask.mMask & 0x8000 ? raw[15] * b : raw[15];
            __m512i t1 = _mm512_load_si512((const __m512i *)(res + 8));

            return SIMDVec_u(t0, t1);
#endif
        }
        // MULVA
        // MMULVA
        // MULSA
        // MMULSA
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
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmpeq_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmpeq_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq (uint64_t b) const {
            __mmask8 t0 = _mm512_cmpeq_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmpeq_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (uint64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpne (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmpneq_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmpneq_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<16> cmpne (uint64_t b) const {
            __mmask8 t0 = _mm512_cmpneq_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmpneq_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (uint64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmpgt_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmpgt_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt (uint64_t b) const {
            __mmask8 t0 = _mm512_cmpgt_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmpgt_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (uint64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<16> cmplt (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmplt_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmplt_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<16> cmplt (uint64_t b) const {
            __mmask8 t0 = _mm512_cmplt_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmplt_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (uint64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpge (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmpge_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmpge_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<16> cmpge (uint64_t b) const {
            __mmask8 t0 = _mm512_cmpge_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmpge_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (uint64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<16> cmple (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmple_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmple_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<16> cmple (uint64_t b) const {
            __mmask8 t0 = _mm512_cmple_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmple_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 t2 = (t1 << 8) | t0;
            return SIMDVecMask<16>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (uint64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_u const & b) const {
            __mmask8 t0 = _mm512_cmpeq_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 t1 = _mm512_cmpeq_epi64_mask(mVec[1], b.mVec[1]);
            return t0 == 0xF && t1 == 0xF;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint64_t b) const {
            __mmask8 t0 = _mm512_cmpeq_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 t1 = _mm512_cmpeq_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            return t0 == 0xF && t1 == 0xF;
        }
        // UNIQUE
        // HADD
        UME_FORCE_INLINE uint64_t hadd() const {
            uint64_t retval = _mm512_reduce_add_epi64(mVec[0]);
            retval += _mm512_reduce_add_epi64(mVec[1]);
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE uint64_t hadd(SIMDVecMask<16> const & mask) const {
            uint64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_epi64((mask.mMask & 0xFF00) >> 8, mVec[1]);
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE uint64_t hadd(uint64_t b) const {
            uint64_t retval = _mm512_reduce_add_epi64(mVec[0]);
            return retval + b;
        }
        // MHADDS
        UME_FORCE_INLINE uint64_t hadd(SIMDVecMask<16> const & mask, uint64_t b) const {
            uint64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask, mVec[0]);
            return retval + b;
        }
        // HMUL
        UME_FORCE_INLINE uint64_t hmul() const {
            uint64_t retval = _mm512_reduce_mul_epi64(mVec[0]);
            retval *= _mm512_reduce_mul_epi64(mVec[1]);
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE uint64_t hmul(SIMDVecMask<16> const & mask) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            uint64_t retval = _mm512_mask_reduce_mul_epi64(t0, mVec[0]);
            retval *= _mm512_mask_reduce_mul_epi64(t1, mVec[1]);
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE uint64_t hmul(uint64_t b) const {
            uint64_t retval = _mm512_reduce_mul_epi64(mVec[0]);
            retval *= _mm512_reduce_mul_epi64(mVec[1]);
            return retval * b;
        }
        // MHMULS
        UME_FORCE_INLINE uint64_t hmul(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            uint64_t retval = _mm512_mask_reduce_mul_epi64(t0, mVec[0]);
            retval *= _mm512_mask_reduce_mul_epi64(t1, mVec[1]);
            return retval * b;
        }

        // FMULADDV
        /*UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mVec[0] * b.mVec[0] + c.mVec[0];
            uint64_t t1 = mVec[1] * b.mVec[1] + c.mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // MFMULADDV
        /*UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] + c.mVec[0]) : mVec[0];
            uint64_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] + c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // FMULSUBV
        /*UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mVec[0] * b.mVec[0] - c.mVec[0];
            uint64_t t1 = mVec[1] * b.mVec[1] - c.mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // MFMULSUBV
        /*UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask[0] ? (mVec[0] * b.mVec[0] - c.mVec[0]) : mVec[0];
            uint64_t t1 = mask.mMask[1] ? (mVec[1] * b.mVec[1] - c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // FADDMULV
        /*UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = (mVec[0] + b.mVec[0]) * c.mVec[0];
            uint64_t t1 = (mVec[1] + b.mVec[1]) * c.mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // MFADDMULV
        /*UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask[0] ? ((mVec[0] + b.mVec[0]) * c.mVec[0]) : mVec[0];
            uint64_t t1 = mask.mMask[1] ? ((mVec[1] + b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // FSUBMULV
        /*UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = (mVec[0] - b.mVec[0]) * c.mVec[0];
            uint64_t t1 = (mVec[1] - b.mVec[1]) * c.mVec[1];
            return SIMDVec_u(t0, t1);
        }*/
        // MFSUBMULV
        /*UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64_t t0 = mask.mMask[0] ? ((mVec[0] - b.mVec[0]) * c.mVec[0]) : mVec[0];
            uint64_t t1 = mask.mMask[1] ? ((mVec[1] - b.mVec[1]) * c.mVec[1]) : mVec[1];
            return SIMDVec_u(t0, t1);
        }*/

        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_max_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_max_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_max_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_max_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint64_t b) const {
            __m512i t0 = _mm512_max_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_max_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_max_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_max_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec[0] = _mm512_max_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_max_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint64_t b) {
            mVec[0] = _mm512_max_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_max_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_max_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_max_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_min_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_min_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_min_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_min_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint64_t b) const {
            __m512i t0 = _mm512_min_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_min_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_min_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_min_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec[0] = _mm512_min_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_min_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint64_t b) {
            mVec[0] = _mm512_min_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_min_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_min_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_min_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // HMAX
        /*UME_FORCE_INLINE uint64_t hmax () const {
            return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
        }*/
        // MHMAX
        /*UME_FORCE_INLINE uint64_t hmax(SIMDVecMask<16> const & mask) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<uint64_t>::min();
            uint64_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
            return t1;
        }*/
        // IMAX
        /*UME_FORCE_INLINE uint64_t imax() const {
            return mVec[0] > mVec[1] ? 0 : 1;
        }*/
        // MIMAX
        /*UME_FORCE_INLINE uint64_t imax(SIMDVecMask<16> const & mask) const {
            uint64_t i0 = 0xFFFFFFFFFFFFFFFF;
            uint64_t t0 = std::numeric_limits<uint64_t>::min();
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
        /*UME_FORCE_INLINE uint64_t hmin() const {
            return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
        }*/
        // MHMIN
        /*UME_FORCE_INLINE uint64_t hmin(SIMDVecMask<16> const & mask) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<uint64_t>::max();
            uint64_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
            return t1;
        }*/
        // IMIN
        /*UME_FORCE_INLINE uint64_t imin() const {
            return mVec[0] < mVec[1] ? 0 : 1;
        }*/
        // MIMIN
        /*UME_FORCE_INLINE uint64_t imin(SIMDVecMask<16> const & mask) const {
            uint64_t i0 = 0xFFFFFFFFFFFFFFFF;
            uint64_t t0 = std::numeric_limits<uint64_t>::max();
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
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_and_si512(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_and_si512(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_and_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_and_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint64_t b) const {
            __m512i t0 = _mm512_and_si512(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_and_si512(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_and_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_and_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec[0] = _mm512_and_si512(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_and_si512(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_and_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_and_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint64_t b) {
            mVec[0] = _mm512_and_si512(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_and_si512(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_and_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_and_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_or_si512(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_or_si512(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_or_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_or_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint64_t b) const {
            __m512i t0 = _mm512_or_si512(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_or_si512(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_or_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_or_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec[0] = _mm512_or_si512(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_or_si512(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_or_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_or_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint64_t b) {
            mVec[0] = _mm512_or_si512(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_or_si512(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_or_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_or_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m512i t0 = _mm512_xor_si512(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_xor_si512(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_xor_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            __m512i t3 = _mm512_mask_xor_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return SIMDVec_u(t2, t3);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint64_t b) const {
            __m512i t0 = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<16> const & mask, uint64_t b) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_mask_xor_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            __m512i t3 = _mm512_mask_xor_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_u(t2, t3);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec[0] = _mm512_xor_si512(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_xor_si512(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_xor_epi64(mVec[0], t0, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_xor_epi64(mVec[1], t1, mVec[1], b.mVec[1]);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint64_t b) {
            mVec[0] = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<16> const & mask, uint64_t b) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            mVec[0] = _mm512_mask_xor_epi64(mVec[0], t0, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_xor_epi64(mVec[1], t1, mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m512i t0 = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            __m512i t1 = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<16> const & mask) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            __m512i t3 = _mm512_mask_xor_epi64(mVec[0], t0, mVec[0], t2);
            __m512i t4 = _mm512_mask_xor_epi64(mVec[1], t1, mVec[1], t2);
            return SIMDVec_u(t3, t4);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            mVec[0] = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            mVec[1] = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota(SIMDVecMask<16> const & mask) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            mVec[0] = _mm512_mask_xor_epi64(mVec[0], t0, mVec[0], t2);
            mVec[1] = _mm512_mask_xor_epi64(mVec[1], t1, mVec[1], t2);
            return *this;
        }
        // HBAND
        /*UME_FORCE_INLINE uint64_t hband() const {
            return mVec[0] & mVec[1];
        }*/
        // MHBAND
        /*UME_FORCE_INLINE uint64_t hband(SIMDVecMask<16> const & mask) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFFFFFFFFFF;
            uint64_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
            return t1;
        }*/
        // HBANDS
        /*UME_FORCE_INLINE uint64_t hband(uint64_t b) const {
            return mVec[0] & mVec[1] & b;
        }*/
        // MHBANDS
        /*UME_FORCE_INLINE uint64_t hband(SIMDVecMask<16> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] & b: b;
            uint64_t t1 = mask.mMask[1] ? mVec[1] & t0: t0;
            return t1;
        }*/
        // HBOR
        /*UME_FORCE_INLINE uint64_t hbor() const {
            return mVec[0] | mVec[1];
        }*/
        // MHBOR
        /*UME_FORCE_INLINE uint64_t hbor(SIMDVecMask<16> const & mask) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint64_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            return t1;
        }*/
        // HBORS
        /*UME_FORCE_INLINE uint64_t hbor(uint64_t b) const {
            return mVec[0] | mVec[1] | b;
        }*/
        // MHBORS
        /*UME_FORCE_INLINE uint64_t hbor(SIMDVecMask<16> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] | b : b;
            uint64_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
            return t1;
        }*/
        // HBXOR
        /*UME_FORCE_INLINE uint64_t hbxor() const {
            return mVec[0] ^ mVec[1];
        }*/
        // MHBXOR
        /*UME_FORCE_INLINE uint64_t hbxor(SIMDVecMask<16> const & mask) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] : 0;
            uint64_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            return t0;
        }*/
        // HBXORS
        /*UME_FORCE_INLINE uint64_t hbxor(uint64_t b) const {
            return mVec[0] ^ mVec[1] ^ b;
        }*/
        // MHBXORS
        /*UME_FORCE_INLINE uint64_t hbxor(SIMDVecMask<16> const & mask, uint64_t b) const {
            uint64_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
            uint64_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
            return t1;
        }*/

        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint64_t * baseAddr, uint64_t* indices) {
            __m512i t0 = _mm512_load_si512((__m512i *)indices);
            __m512i t1 =_mm512_load_si512((__m512i *)(indices + 8));
            mVec[0] = _mm512_i64gather_epi64(t0, (__int64 const*)baseAddr, 8);
            mVec[1] = _mm512_i64gather_epi64(t1, (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint64_t* baseAddr, uint64_t* indices) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_load_si512((__m512i *)indices);
            __m512i t3 = _mm512_load_si512((__m512i *)(indices + 8));
            __m512i t4 = _mm512_i64gather_epi64(t2, (__int64 const*)baseAddr, 8);
            __m512i t5 = _mm512_i64gather_epi64(t3, (__int64 const*)baseAddr, 8);
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], t0, t4);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], t1, t5);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint64_t * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = _mm512_i64gather_epi64(indices.mVec[0], (__int64 const*)baseAddr, 8);
            mVec[1] = _mm512_i64gather_epi64(indices.mVec[1], (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint64_t* baseAddr, SIMDVec_u const & indices) {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_i64gather_epi64(indices.mVec[0], (__int64 const*)baseAddr, 8);
            __m512i t3 = _mm512_i64gather_epi64(indices.mVec[1], (__int64 const*)baseAddr, 8);
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], t0, t2);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], t1, t3);
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE uint64_t* scatter(uint64_t* baseAddr, uint64_t* indices) const {
            __m512i t0 = _mm512_load_si512((__m512i *)indices);
            __m512i t1 = _mm512_load_si512((__m512i *)(indices + 8));
            _mm512_i64scatter_epi64(baseAddr, t0, mVec[0], 8);
            _mm512_i64scatter_epi64(baseAddr, t1, mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint64_t* scatter(SIMDVecMask<16> const & mask, uint64_t* baseAddr, uint64_t* indices) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            __m512i t2 = _mm512_load_si512((__m512i *)indices);
            __m512i t3 = _mm512_load_si512((__m512i *)(indices + 8));
            _mm512_mask_i64scatter_epi64(baseAddr, t0, t2, mVec[0], 8);
            _mm512_mask_i64scatter_epi64(baseAddr, t1, t3, mVec[1], 8);
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint64_t* scatter(uint64_t* baseAddr, SIMDVec_u const & indices) const {
            _mm512_i64scatter_epi64(baseAddr, indices.mVec[0], mVec[0], 8);
            _mm512_i64scatter_epi64(baseAddr, indices.mVec[1], mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint64_t* scatter(SIMDVecMask<16> const & mask, uint64_t* baseAddr, SIMDVec_u const & indices) const {
            __mmask8 t0 = mask.mMask & 0x00FF;
            __mmask8 t1 = (mask.mMask & 0xFF00) >> 8;
            _mm512_mask_i64scatter_epi64(baseAddr, t0, indices.mVec[0], mVec[0], 8);
            _mm512_mask_i64scatter_epi64(baseAddr, t1, indices.mVec[1], mVec[1], 8);
            return baseAddr;
        }

        //// LSHV
        //UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
        //    uint64_t t0 = mVec[0] << b.mVec[0];
        //    uint64_t t1 = mVec[1] << b.mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// MLSHV
        //UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
        //    uint64_t t0 = mask.mMask[0] ? mVec[0] << b.mVec[0] : mVec[0];
        //    uint64_t t1 = mask.mMask[1] ? mVec[1] << b.mVec[1] : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// LSHS
        //UME_FORCE_INLINE SIMDVec_u lsh(uint64_t b) const {
        //    uint64_t t0 = mVec[0] << b;
        //    uint64_t t1 = mVec[1] << b;
        //    return SIMDVec_u(t0, t1);
        //}
        //// MLSHS
        //UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<16> const & mask, uint64_t b) const {
        //    uint64_t t0 = mask.mMask[0] ? mVec[0] << b : mVec[0];
        //    uint64_t t1 = mask.mMask[1] ? mVec[1] << b : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// LSHVA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
        //    mVec[0] = mVec[0] << b.mVec[0];
        //    mVec[1] = mVec[1] << b.mVec[1];
        //    return *this;
        //}
        //// MLSHVA
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
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
        //UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<16> const & mask, uint64_t b) {
        //    if(mask.mMask[0]) mVec[0] = mVec[0] << b;
        //    if(mask.mMask[1]) mVec[1] = mVec[1] << b;
        //    return *this;
        //}
        //// RSHV
        //UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
        //    uint64_t t0 = mVec[0] >> b.mVec[0];
        //    uint64_t t1 = mVec[1] >> b.mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// MRSHV
        //UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
        //    uint64_t t0 = mask.mMask[0] ? mVec[0] >> b.mVec[0] : mVec[0];
        //    uint64_t t1 = mask.mMask[1] ? mVec[1] >> b.mVec[1] : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// RSHS
        //UME_FORCE_INLINE SIMDVec_u rsh(uint64_t b) const {
        //    uint64_t t0 = mVec[0] >> b;
        //    uint64_t t1 = mVec[1] >> b;
        //    return SIMDVec_u(t0, t1);
        //}
        //// MRSHS
        //UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<16> const & mask, uint64_t b) const {
        //    uint64_t t0 = mask.mMask[0] ? mVec[0] >> b : mVec[0];
        //    uint64_t t1 = mask.mMask[1] ? mVec[1] >> b : mVec[1];
        //    return SIMDVec_u(t0, t1);
        //}
        //// RSHVA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
        //    mVec[0] = mVec[0] >> b.mVec[0];
        //    mVec[1] = mVec[1] >> b.mVec[1];
        //    return *this;
        //}
        //// MRSHVA
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
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
        //UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<16> const & mask, uint64_t b) {
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
        // PACKLO
        // PACKHI
        // UNPACK
        void unpack(SIMDVec_u<uint64_t, 8> & a, SIMDVec_u<uint64_t, 8> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        SIMDVec_u<uint64_t, 8> unpacklo() const {
            return SIMDVec_u<uint64_t, 8> (mVec[0]);
        }
        // UNPACKHI
        SIMDVec_u<uint64_t, 8> unpackhi() const {
            return SIMDVec_u<uint64_t, 8> (mVec[1]);
        }

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 16>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 16>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 16>() const;
    };

#undef _mm512_set1_epi64
#undef EXPAND_CALL_BINARY
#undef EXPAND_CALL_BINARY_MASK
#undef EXPAND_CALL_BINARY_SCALAR_MASK

}
}

#endif
