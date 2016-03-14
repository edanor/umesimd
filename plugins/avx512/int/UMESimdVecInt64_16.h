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

#ifndef UME_SIMD_VEC_INT64_16_H_
#define UME_SIMD_VEC_INT64_16_H_

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
    class SIMDVec_i<int64_t, 16> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 16>,
            SIMDVec_u<uint64_t, 16>,
            int64_t,
            16,
            uint64_t,
            SIMDVecMask<16>,
            SIMDVecSwizzle<16>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 16>,
            SIMDVec_i<int64_t, 8>>
    {
    public:
        friend class SIMDVec_u<uint64_t, 16>;
        friend class SIMDVec_f<double, 16>;

    private:
        __m512i mVec[2];

        inline explicit SIMDVec_i(__m512i & x0, __m512i & x1) {
            mVec[0] = x0;
            mVec[1] = x1;
        }
    public:
        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 64; }

        // ZERO-CONSTR
        inline SIMDVec_i() {}
        // SET-CONSTR
        inline explicit SIMDVec_i(int64_t i) {
            mVec[0] = _mm512_set1_epi64(i);
            mVec[1] = _mm512_set1_epi64(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int64_t const *p) {
            mVec[0] = _mm512_loadu_si512(p);
            mVec[1] = _mm512_loadu_si512(p + 8);
        }
        // FULL-CONSTR
        inline SIMDVec_i(int64_t i0,  int64_t i1,  int64_t i2,  int64_t i3,
                         int64_t i4,  int64_t i5,  int64_t i6,  int64_t i7,
                         int64_t i8,  int64_t i9,  int64_t i10, int64_t i11,
                         int64_t i12, int64_t i13, int64_t i14, int64_t i15) {
            mVec[0] = _mm512_set_epi64(i7, i6, i5, i4, i3, i2, i1, i0);
            mVec[1] = _mm512_set_epi64(i15, i14, i13, i12, i11, i10, i9, i8);
        }

        // EXTRACT
        inline int64_t extract(uint32_t index) const {
            alignas(64) int64_t raw[8];
            if (index < 8) {
                _mm512_store_si512(raw, mVec[0]);
                return raw[index];
            }
            else {
                _mm512_store_si512(raw, mVec[1]);
                return raw[index - 8];
            }
        }
        inline int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int64_t value) {
            alignas(64) int64_t raw[8];
            if (index < 8) {
                _mm512_store_si512(raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm512_load_si512(raw);
            }
            else {
                _mm512_store_si512(raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm512_load_si512(raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
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
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], mask.mMask & 0xFF, b.mVec[0]);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int64_t b) {
            mVec[0] = _mm512_set1_epi64(b);
            mVec[1] = _mm512_set1_epi64(b);
            return *this;
        }
        inline SIMDVec_i & operator= (int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, int64_t b) {
            __m512i t1 = _mm512_set1_epi64(b);
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], mask.mMask & 0xFF, t1);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), t1);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_i & load(int64_t const *p) {
            mVec[0] = _mm512_loadu_si512((const __m512i *)p);
            mVec[1] = _mm512_loadu_si512((const __m512i *)p + 8);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<16> const & mask, int64_t const *p) {
            mVec[0] = _mm512_mask_loadu_epi64(mVec[0], mask.mMask & 0xFF, p);
            mVec[1] = _mm512_mask_loadu_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), p + 8);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int64_t const *p) {
            mVec[0] = _mm512_load_si512((const __m512i *) p);
            mVec[1] = _mm512_load_si512((const __m512i *) p + 8);
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<16> const & mask, int64_t const *p) {
            mVec[0] = _mm512_mask_load_epi64(mVec[0], mask.mMask & 0xFF, p);
            mVec[1] = _mm512_mask_load_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), p + 8);
            return *this;
        }
        // STORE
        inline int64_t* store(int64_t* p) const {
            _mm512_storeu_si512((__m512i *)p, mVec[0]);
            _mm512_storeu_si512((__m512i *)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        inline int64_t* store(SIMDVecMask<16> const & mask, int64_t* p) const {
            _mm512_mask_storeu_epi64(p, mask.mMask & 0xFF, mVec[0]);
            _mm512_mask_storeu_epi64(p + 8, ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return p;
        }
        // STOREA
        inline int64_t* storea(int64_t* p) const {
            _mm512_store_si512((__m512i *)p, mVec[0]);
            _mm512_store_si512((__m512i *)(p + 8), mVec[1]);
            return p;
        }
        // MSTOREA
        inline int64_t* storea(SIMDVecMask<16> const & mask, int64_t* p) const {
            _mm512_mask_store_epi64(p, mask.mMask & 0xFF, mVec[0]);
            _mm512_mask_store_epi64((p + 8), ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return p;
        }

        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_mov_epi64(mVec[0], mask.mMask & 0xFF, b.mVec[0]);
            __m512i t1 = _mm512_mask_mov_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_mov_epi64(mVec[0], mask.mMask & 0xFF, _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_mov_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_add_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_add_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_add_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_add_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ADDS
        inline SIMDVec_i add(int64_t b) const {
            __m512i t0 = _mm512_add_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_add_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_add_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_add_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec[0] = _mm512_add_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_add_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_add_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_add_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int64_t b) {
            mVec[0] = _mm512_add_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_add_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        inline SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_add_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_add_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
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
            __m512i t0 = _mm512_set1_epi64(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_add_epi64(mVec[0], t0);
            mVec[1] = _mm512_add_epi64(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<16> const & mask) {
            __m512i t0 = mVec[0];
            __m512i t1 = mVec[1];
            mVec[0] = _mm512_mask_add_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_add_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(1));
            return SIMDVec_i(t0, t1);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            __m512i t0 = _mm512_set1_epi64(1);
            mVec[0] = _mm512_add_epi64(mVec[0], t0);
            mVec[1] = _mm512_add_epi64(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_add_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_add_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(1));
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_sub_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_sub_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // SUBS
        inline SIMDVec_i sub(int64_t b) const {
            __m512i t0 = _mm512_sub_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_sub_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec[0] = _mm512_sub_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_sub_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int64_t b) {
            mVec[0] = _mm512_sub_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_sub_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        inline SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
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
            __m512i t0 = _mm512_sub_epi64(b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_sub_epi64(b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_sub_epi64(b.mVec[0], mask.mMask & 0xFF, b.mVec[0], mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi64(b.mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int64_t b) const {
            __m512i t0 = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[0]);
            __m512i t1 = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_set1_epi64(b);
            __m512i t1 = _mm512_mask_sub_epi64(t0, mask.mMask & 0xFF, t0, mVec[0]);
            __m512i t2 = _mm512_mask_sub_epi64(t0, ((mask.mMask & 0xFF00) >> 8), t1, mVec[1]);
            return SIMDVec_i(t1, t2);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec[0] = _mm512_sub_epi64(b.mVec[0], mVec[0]);
            mVec[1] = _mm512_sub_epi64(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_sub_epi64(b.mVec[0], mask.mMask & 0xFF, b.mVec[0], mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(b.mVec[1], ((mask.mMask & 0xFF00) >> 8), b.mVec[1], mVec[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int64_t b) {
            mVec[0] = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[0]);
            mVec[1] = _mm512_sub_epi64(_mm512_set1_epi64(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, int64_t b) {
            __m512i t0 = _mm512_set1_epi64(b);
            mVec[0] = _mm512_mask_sub_epi64(t0, mask.mMask & 0xFF, t0, mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(t0, ((mask.mMask & 0xFF00) >> 8), t0, mVec[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_i postdec() {
            __m512i t0 = _mm512_set1_epi64(1);
            __m512i t1 = mVec[0];
            __m512i t2 = mVec[1];
            mVec[0] = _mm512_sub_epi64(mVec[0], t0);
            mVec[1] = _mm512_sub_epi64(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<16> const & mask) {
            __m512i t0 = mVec[0];
            __m512i t1 = mVec[1];
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(1));
            return SIMDVec_i(t0, t1);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            mVec[0] = _mm512_sub_epi64(mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_sub_epi64(mVec[1], _mm512_set1_epi64(1));
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(1));
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(1));
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
#if defined(__AVX512DQ__)
    #if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
            __m256i t6 = _mm256_mullo_epi64(t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            t0 = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t9 = _mm256_mullo_epi64(t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            __m256i t12 = _mm256_mullo_epi64(t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            t13 = _mm512_inserti64x4(t13, t12, 1);
            return SIMDVec_i(t0, t13);
    #else
            __m512i t0 = _mm512_mullo_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mullo_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
    #endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
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


            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = t31 * t35;
            int64_t t40 = t32 * t36;
            int64_t t41 = t33 * t37;
            int64_t t42 = t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = t45 * t49;
            int64_t t54 = t46 * t50;
            int64_t t55 = t47 * t51;
            int64_t t56 = t48 * t52;
            __m512i t57 = _mm512_set_epi64(t56, t55, t54, t53, t52, t51, t50, t49);
            return SIMDVec_i(t0, t57);
#endif
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
            __m256i t6 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            t0 = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t9 = _mm256_mask_mullo_epi64(t7, (mask.mMask & 0xF00) >> 8, t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            __m256i t12 = _mm256_mask_mullo_epi64(t10, (mask.mMask & 0xF000) >> 12, t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            t13 = _mm512_inserti64x4(t13, t12, 1);
            return SIMDVec_i(t0, t13);
#else
            __m512i t0 = _mm512_mask_mullo_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_mullo_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
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

            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = ((mask.mMask & 0x100) == 0) ? t31 : t31 * t35;
            int64_t t40 = ((mask.mMask & 0x200) == 0) ? t32 : t32 * t36;
            int64_t t41 = ((mask.mMask & 0x400) == 0) ? t33 : t33 * t37;
            int64_t t42 = ((mask.mMask & 0x800) == 0) ? t34 : t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = ((mask.mMask & 0x1000) == 0) ? t45 : t45 * t49;
            int64_t t54 = ((mask.mMask & 0x2000) == 0) ? t46 : t46 * t50;
            int64_t t55 = ((mask.mMask & 0x4000) == 0) ? t47 : t47 * t51;
            int64_t t56 = ((mask.mMask & 0x8000) == 0) ? t48 : t48 * t52;
            __m512i t57 = _mm512_set_epi64(t56, t55, t54, t53, t42, t41, t40, t39);
            return SIMDVec_i(t0, t57);
#endif
        }
        // MULS
        inline SIMDVec_i mul(int64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm256_set1_epi64x(b);
            __m256i t6 = _mm256_mullo_epi64(t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            t0 = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm256_set1_epi64x(b);
            __m256i t9 = _mm256_mullo_epi64(t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm256_set1_epi64x(b);
            __m256i t12 = _mm256_mullo_epi64(t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            t13 = _mm512_inserti64x4(t13, t12, 1);
            return SIMDVec_i(t0, t13);
#else
            __m512i t0 = _mm512_mullo_epi64(mVec[0], _mm256_set1_epi64x(b));
            __m512i t1 = _mm512_mullo_epi64(mVec[1], _mm256_set1_epi64x(b));
            return SIMDVec_i(t0, t1);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm256_set1_epi64x(b);
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


            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm256_set1_epi64x(b);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = t31 * t35;
            int64_t t40 = t32 * t36;
            int64_t t41 = t33 * t37;
            int64_t t42 = t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm256_set1_epi64x(b);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = t45 * t49;
            int64_t t54 = t46 * t50;
            int64_t t55 = t47 * t51;
            int64_t t56 = t48 * t52;
            __m512i t57 = _mm512_set_epi64(t56, t55, t54, t53, t42, t41, t40, t39);
            return SIMDVec_i(t0, t57);
#endif
        }
        inline SIMDVec_i operator* (int64_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, int64_t b) const {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm256_set1_epi64x(b);
            __m256i t6 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            t0 = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm256_set1_epi64x(b);
            __m256i t9 = _mm256_mask_mullo_epi64(t7, (mask.mMask & 0xF00) >> 8, t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm256_set1_epi64x(b);
            __m256i t12 = _mm256_mask_mullo_epi64(t10, (mask.mMask & 0xF000) >> 12, t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            t13 = _mm512_inserti64x4(t13, t12, 1);
            return SIMDVec_i(t0, t13);
#else
            __m512i t0 = _mm512_mask_mullo_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm256_set1_epi64x(b));
            __m512i t1 = _mm512_mask_mullo_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm256_set1_epi64x(b));
            return SIMDVec_i(t0, t1);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm256_set1_epi64x(b);
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

            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm256_set1_epi64x(b);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = ((mask.mMask & 0x100) == 0) ? t31 : t31 * t35;
            int64_t t40 = ((mask.mMask & 0x200) == 0) ? t32 : t32 * t36;
            int64_t t41 = ((mask.mMask & 0x400) == 0) ? t33 : t33 * t37;
            int64_t t42 = ((mask.mMask & 0x800) == 0) ? t34 : t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm256_set1_epi64x(b);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = ((mask.mMask & 0x1000) == 0) ? t45 : t45 * t49;
            int64_t t54 = ((mask.mMask & 0x2000) == 0) ? t46 : t46 * t50;
            int64_t t55 = ((mask.mMask & 0x4000) == 0) ? t47 : t47 * t51;
            int64_t t56 = ((mask.mMask & 0x8000) == 0) ? t48 : t48 * t52;
            __m512i t57 = _mm512_set_epi64(t56, t55, t54, t53, t42, t41, t40, t39);
            return SIMDVec_i(t0, t57);
#endif
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
            __m256i t6 = _mm256_mullo_epi64(t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            mVec[0] = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t9 = _mm256_mullo_epi64(t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            __m256i t12 = _mm256_mullo_epi64(t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            mVec[1] = _mm512_inserti64x4(t13, t12, 1);
#else
            mVec[0] = _mm512_mullo_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mullo_epi64(mVec[1], b.mVec[1]);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
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
            mVec[0] = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);


            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = t31 * t35;
            int64_t t40 = t32 * t36;
            int64_t t41 = t33 * t37;
            int64_t t42 = t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = t45 * t49;
            int64_t t54 = t46 * t50;
            int64_t t55 = t47 * t51;
            int64_t t56 = t48 * t52;
            mVec[1] = _mm512_set_epi64(t56, t55, t54, t53, t52, t51, t50, t49);
#endif
            return *this;
        }
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
            __m256i t6 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            mVec[0] = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            __m256i t9 = _mm256_mask_mullo_epi64(t7, (mask.mMask & 0xF00) >> 8, t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            __m256i t12 = _mm256_mask_mullo_epi64(t10, (mask.mMask & 0xF000) >> 12, t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            mVec[1] = _mm512_inserti64x4(t13, t12, 1);
#else
            mVec[0] = _mm512_mask_mullo_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_mullo_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm512_extracti64x4_epi64(b.mVec[0], 0);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm512_extracti64x4_epi64(b.mVec[0], 1);
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
            mVec[0] = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);

            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm512_extracti64x4_epi64(b.mVec[1], 0);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = ((mask.mMask & 0x100) == 0) ? t31 : t31 * t35;
            int64_t t40 = ((mask.mMask & 0x200) == 0) ? t32 : t32 * t36;
            int64_t t41 = ((mask.mMask & 0x400) == 0) ? t33 : t33 * t37;
            int64_t t42 = ((mask.mMask & 0x800) == 0) ? t34 : t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm512_extracti64x4_epi64(b.mVec[1], 1);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = ((mask.mMask & 0x1000) == 0) ? t45 : t45 * t49;
            int64_t t54 = ((mask.mMask & 0x2000) == 0) ? t46 : t46 * t50;
            int64_t t55 = ((mask.mMask & 0x4000) == 0) ? t47 : t47 * t51;
            int64_t t56 = ((mask.mMask & 0x8000) == 0) ? t48 : t48 * t52;
            mVec[1] = _mm512_set_epi64(t56, t55, t54, t53, t42, t41, t40, t39);
#endif
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mullo_epi64(t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm256_set1_epi64x(b);
            __m256i t6 = _mm256_mullo_epi64(t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            mVec[0] = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm256_set1_epi64x(b);
            __m256i t9 = _mm256_mullo_epi64(t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm256_set1_epi64x(b);
            __m256i t12 = _mm256_mullo_epi64(t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            mVec[1] = _mm512_inserti64x4(t13, t12, 1);
#else
            mVec[0] = _mm512_mullo_epi64(mVec[0], _mm256_set1_epi64x(b));
            mVec[1]  = _mm512_mullo_epi64(mVec[1], _mm256_set1_epi64x(b));
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm256_set1_epi64x(b);
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
            mVec[0] = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);

            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm256_set1_epi64x(b);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = t31 * t35;
            int64_t t40 = t32 * t36;
            int64_t t41 = t33 * t37;
            int64_t t42 = t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm256_set1_epi64x(b);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = t45 * t49;
            int64_t t54 = t46 * t50;
            int64_t t55 = t47 * t51;
            int64_t t56 = t48 * t52;
            mVec[1] = _mm512_set_epi64(t56, t55, t54, t53, t52, t51, t50, t49);
#endif
            return *this;
        }
        inline SIMDVec_i & operator*= (int64_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, int64_t b) {
#if defined(__AVX512DQ__)
#if defined(__AVX512VL__)
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
            __m256i t3 = _mm256_mask_mullo_epi64(t1, mask.mMask & 0xF, t1, t2);
            __m256i t4 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t5 = _mm256_set1_epi64x(b);
            __m256i t6 = _mm256_mask_mullo_epi64(t4, (mask.mMask & 0xF0) >> 4, t4, t5);
            __m512i t0 = _mm512_inserti64x4(mVec[0], t3, 0);
            mVec[0] = _mm512_inserti64x4(t0, t6, 1);

            __m256i t7 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t8 = _mm256_set1_epi64x(b);
            __m256i t9 = _mm256_mask_mullo_epi64(t7, (mask.mMask & 0xF00) >> 8, t7, t8);
            __m256i t10 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t11 = _mm256_set1_epi64x(b);
            __m256i t12 = _mm256_mask_mullo_epi64(t10, (mask.mMask & 0xF000) >> 12, t10, t11);
            __m512i t13 = _mm512_inserti64x4(mVec[1], t9, 0);
            mVec[1] = _mm512_inserti64x4(t13, t12, 1);
#else
            mVec[0] = _mm512_mask_mullo_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm256_set1_epi64x(b));
            mVec[1] = _mm512_mask_mullo_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm256_set1_epi64x(b));
#endif
#else
            __m256i t1 = _mm512_extracti64x4_epi64(mVec[0], 0);
            __m256i t2 = _mm256_set1_epi64x(b);
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
            __m256i t15 = _mm512_extracti64x4_epi64(mVec[0], 1);
            __m256i t16 = _mm256_set1_epi64x(b);
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
            mVec[0] = _mm512_set_epi64(t28, t27, t26, t25, t14, t13, t12, t11);

            __m256i t29 = _mm512_extracti64x4_epi64(mVec[1], 0);
            __m256i t30 = _mm256_set1_epi64x(b);
            int64_t t31 = _mm256_extract_epi64(t29, 0);
            int64_t t32 = _mm256_extract_epi64(t29, 1);
            int64_t t33 = _mm256_extract_epi64(t29, 2);
            int64_t t34 = _mm256_extract_epi64(t29, 3);
            int64_t t35 = _mm256_extract_epi64(t30, 0);
            int64_t t36 = _mm256_extract_epi64(t30, 1);
            int64_t t37 = _mm256_extract_epi64(t30, 2);
            int64_t t38 = _mm256_extract_epi64(t30, 3);
            int64_t t39 = ((mask.mMask & 0x100) == 0) ? t31 : t31 * t35;
            int64_t t40 = ((mask.mMask & 0x200) == 0) ? t32 : t32 * t36;
            int64_t t41 = ((mask.mMask & 0x400) == 0) ? t33 : t33 * t37;
            int64_t t42 = ((mask.mMask & 0x800) == 0) ? t34 : t34 * t38;
            __m256i t43 = _mm512_extracti64x4_epi64(mVec[1], 1);
            __m256i t44 = _mm256_set1_epi64x(b);
            int64_t t45 = _mm256_extract_epi64(t43, 0);
            int64_t t46 = _mm256_extract_epi64(t43, 1);
            int64_t t47 = _mm256_extract_epi64(t43, 2);
            int64_t t48 = _mm256_extract_epi64(t43, 3);
            int64_t t49 = _mm256_extract_epi64(t44, 0);
            int64_t t50 = _mm256_extract_epi64(t44, 1);
            int64_t t51 = _mm256_extract_epi64(t44, 2);
            int64_t t52 = _mm256_extract_epi64(t44, 3);
            int64_t t53 = ((mask.mMask & 0x1000) == 0) ? t45 : t45 * t49;
            int64_t t54 = ((mask.mMask & 0x2000) == 0) ? t46 : t46 * t50;
            int64_t t55 = ((mask.mMask & 0x4000) == 0) ? t47 : t47 * t51;
            int64_t t56 = ((mask.mMask & 0x8000) == 0) ? t48 : t48 * t52;
            mVec[1] = _mm512_set_epi64(t56, t55, t54, t53, t42, t41, t40, t39);
#endif
            return *this;
        }
        // DIVV
        /*inline SIMDVec_i div(SIMDVec_i const & b) const {
            int64_t t0 = mVec[0][0] / b.mVec[0][0];
            int64_t t1 = mVec[0][1] / b.mVec[0][1];
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }*/
        // MDIVV
        /*inline SIMDVec_i div(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] / b.mVec[0][0] : mVec[0][0];
            int64_t t1 = mask.mMask[1] ? mVec[0][1] / b.mVec[0][1] : mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // DIVS
        /*inline SIMDVec_i div(int64_t b) const {
            int64_t t0 = mVec[0][0] / b;
            int64_t t1 = mVec[0][1] / b;
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator/ (int64_t b) const {
            return div(b);
        }*/
        // MDIVS
        /*inline SIMDVec_i div(SIMDVecMask<16> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] / b : mVec[0][0];
            int64_t t1 = mask.mMask[1] ? mVec[0][1] / b : mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // DIVVA
        /*inline SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec[0][0] /= b.mVec[0][0];
            mVec[0][1] /= b.mVec[0][1];
            return *this;
        }
        inline SIMDVec_i operator/= (SIMDVec_i const & b) {
            return diva(b);
        }*/
        // MDIVVA
        /*inline SIMDVec_i & diva(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0][0] = mask.mMask[0] ? mVec[0][0] / b.mVec[0][0] : mVec[0][0];
            mVec[0][1] = mask.mMask[1] ? mVec[0][1] / b.mVec[0][1] : mVec[0][1];
            return *this;
        }*/
        // DIVSA
        /*inline SIMDVec_i & diva(int64_t b) {
            mVec[0][0] /= b;
            mVec[0][1] /= b;
            return *this;
        }
        inline SIMDVec_i operator/= (int64_t b) {
            return diva(b);
        }*/
        // MDIVSA
        /*inline SIMDVec_i & diva(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0][0] = mask.mMask[0] ? mVec[0][0] / b : mVec[0][0];
            mVec[0][1] = mask.mMask[1] ? mVec[0][1] / b : mVec[0][1];
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
        inline SIMDVecMask<16> cmpeq (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmpeq_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq (int64_t b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmpeq_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator== (int64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpneq_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmpneq_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne (int64_t b) const {
            __mmask8 m0 = _mm512_cmpneq_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmpneq_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator!= (int64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpgt_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmpgt_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt (int64_t b) const {
            __mmask8 m0 = _mm512_cmpgt_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmpgt_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator> (int64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmplt_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmplt_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt (int64_t b) const {
            __mmask8 m0 = _mm512_cmplt_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmplt_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator< (int64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpge_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmpge_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge (int64_t b) const {
            __mmask8 m0 = _mm512_cmpge_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmpge_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator>= (int64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmple_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmple_epi64_mask(mVec[1], b.mVec[1]);
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple (int64_t b) const {
            __mmask8 m0 = _mm512_cmple_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmple_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            __mmask16 m2 = __mmask16(m0) | (__mmask16(m1) << 8);
            return SIMDVecMask<16>(m2);
        }
        inline SIMDVecMask<16> operator<= (int64_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe (SIMDVec_i const & b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec[0], b.mVec[0]);
            __mmask8 m1 = _mm512_cmpeq_epi64_mask(mVec[1], b.mVec[1]);
            return (m0 & m1) == 0xFF;
        }
        // CMPES
        inline bool cmpe(int64_t b) const {
            __mmask8 m0 = _mm512_cmpeq_epi64_mask(mVec[0], _mm512_set1_epi64(b));
            __mmask8 m1 = _mm512_cmpeq_epi64_mask(mVec[1], _mm512_set1_epi64(b));
            return (m0 & m1) == 0xFF;
        }
        // UNIQUE
        // HADD
        inline int64_t hadd() const {
            int64_t retval = _mm512_reduce_add_epi64(mVec[0]);
            retval += _mm512_reduce_add_epi64(mVec[1]);
            return retval;
        }
        // MHADD
        inline int64_t hadd(SIMDVecMask<16> const & mask) const {
            int64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_epi64(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval;
        }
        // HADDS
        inline int64_t hadd(int64_t b) const {
            int64_t retval = _mm512_reduce_add_epi64(mVec[0]);
            retval += _mm512_reduce_add_epi64(mVec[1]);
            return retval + b;
        }
        // MHADDS
        inline int64_t hadd(SIMDVecMask<16> const & mask, int64_t b) const {
            int64_t retval = _mm512_mask_reduce_add_epi64(mask.mMask & 0xFF, mVec[0]);
            retval += _mm512_mask_reduce_add_epi64(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval + b;
        }
        // HMUL
        inline int64_t hmul() const {
            int64_t retval = _mm512_reduce_mul_epi64(mVec[0]);
            retval *= _mm512_reduce_mul_epi64(mVec[1]);
            return retval;
        }
        // MHMUL
        inline int64_t hmul(SIMDVecMask<16> const & mask) const {
            int64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask & 0xFF, mVec[0]);
            retval *= _mm512_mask_reduce_mul_epi64(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval;
        }
        // HMULS
        inline int64_t hmul(int64_t b) const {
            int64_t retval = _mm512_reduce_mul_epi64(mVec[0]);
            retval *= _mm512_reduce_mul_epi64(mVec[1]);
            return retval * b;
        }
        // MHMULS
        inline int64_t hmul(SIMDVecMask<16> const & mask, int64_t b) const {
            int64_t retval = _mm512_mask_reduce_mul_epi64(mask.mMask & 0xFF, mVec[0]);
            retval *= _mm512_mask_reduce_mul_epi64(((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return retval * b;
        }

        // FMULADDV
        /*inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mVec[0][0] * b.mVec[0][0] + c.mVec[0][0];
            int64_t t1 = mVec[0][1] * b.mVec[0][1] + c.mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // MFMULADDV
        /*inline SIMDVec_i fmuladd(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask[0] ? (mVec[0][0] * b.mVec[0][0] + c.mVec[0][0]) : mVec[0][0];
            int64_t t1 = mask.mMask[1] ? (mVec[0][1] * b.mVec[0][1] + c.mVec[0][1]) : mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // FMULSUBV
        /*inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mVec[0][0] * b.mVec[0][0] - c.mVec[0][0];
            int64_t t1 = mVec[0][1] * b.mVec[0][1] - c.mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // MFMULSUBV
        /*inline SIMDVec_i fmulsub(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask[0] ? (mVec[0][0] * b.mVec[0][0] - c.mVec[0][0]) : mVec[0][0];
            int64_t t1 = mask.mMask[1] ? (mVec[0][1] * b.mVec[0][1] - c.mVec[0][1]) : mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // FADDMULV
        /*inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = (mVec[0][0] + b.mVec[0][0]) * c.mVec[0][0];
            int64_t t1 = (mVec[0][1] + b.mVec[0][1]) * c.mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // MFADDMULV
        /*inline SIMDVec_i faddmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask[0] ? ((mVec[0][0] + b.mVec[0][0]) * c.mVec[0][0]) : mVec[0][0];
            int64_t t1 = mask.mMask[1] ? ((mVec[0][1] + b.mVec[0][1]) * c.mVec[0][1]) : mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // FSUBMULV
        /*inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = (mVec[0][0] - b.mVec[0][0]) * c.mVec[0][0];
            int64_t t1 = (mVec[0][1] - b.mVec[0][1]) * c.mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/
        // MFSUBMULV
        /*inline SIMDVec_i fsubmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64_t t0 = mask.mMask[0] ? ((mVec[0][0] - b.mVec[0][0]) * c.mVec[0][0]) : mVec[0][0];
            int64_t t1 = mask.mMask[1] ? ((mVec[0][1] - b.mVec[0][1]) * c.mVec[0][1]) : mVec[0][1];
            return SIMDVec_i(t0, t1);
        }*/

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_max_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_max_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_max_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_max_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MAXS
        inline SIMDVec_i max(int64_t b) const {
            __m512i t0 = _mm512_max_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_max_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_max_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_max_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec[0] = _mm512_max_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_max_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_max_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_max_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int64_t b) {
            mVec[0] = _mm512_max_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_max_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_max_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_max_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_min_epi64(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_min_epi64(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_min_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_min_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MINS
        inline SIMDVec_i min(int64_t b) const {
            __m512i t0 = _mm512_min_epi64(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_min_epi64(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_min_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_min_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec[0] = _mm512_min_epi64(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_min_epi64(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_min_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_min_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int64_t b) {
            mVec[0] = _mm512_min_epi64(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_min_epi64(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_min_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_min_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // HMAX
        /*inline int64_t hmax () const {
            return mVec[0][0] > mVec[0][1] ? mVec[0][0] : mVec[0][1];
        }*/
        // MHMAX
        /*inline int64_t hmax(SIMDVecMask<16> const & mask) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] : std::numeric_limits<int64_t>::min();
            int64_t t1 = (mask.mMask[1] && mVec[0][1] > t0) ? mVec[0][1] : t0;
            return t1;
        }*/
        // IMAX
        /*inline int64_t imax() const {
            return mVec[0][0] > mVec[0][1] ? 0 : 1;
        }*/
        // MIMAX
        /*inline int64_t imax(SIMDVecMask<16> const & mask) const {
            int64_t i0 = 0xFFFFFFFFFFFFFFFF;
            int64_t t0 = std::numeric_limits<int64_t>::min();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0][0];
            }
            if(mask.mMask[1] == true && mVec[0][1] > t0) {
                i0 = 1;
            }
            return i0;
        }*/
        // HMIN
        /*inline int64_t hmin() const {
            return mVec[0][0] < mVec[0][1] ? mVec[0][0] : mVec[0][1];
        }*/
        // MHMIN
        /*inline int64_t hmin(SIMDVecMask<16> const & mask) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] : std::numeric_limits<int64_t>::max();
            int64_t t1 = (mask.mMask[1] && mVec[0][1] < t0) ? mVec[0][1] : t0;
            return t1;
        }*/
        // IMIN
        /*inline int64_t imin() const {
            return mVec[0][0] < mVec[0][1] ? 0 : 1;
        }*/
        // MIMIN
        /*inline int64_t imin(SIMDVecMask<16> const & mask) const {
            int64_t i0 = 0xFFFFFFFFFFFFFFFF;
            int64_t t0 = std::numeric_limits<int64_t>::max();
            if(mask.mMask[0] == true) {
                i0 = 0;
                t0 = mVec[0][0];
            }
            if(mask.mMask[1] == true && mVec[0][1] < t0) {
                i0 = 1;
            }
            return i0;
        }*/

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_and_si512(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_and_si512(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_and_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_and_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BANDS
        inline SIMDVec_i band(int64_t b) const {
            __m512i t0 = _mm512_and_si512(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_and_si512(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator& (int64_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_and_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_and_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec[0] = _mm512_and_si512(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_and_si512(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_and_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_and_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int64_t b) {
            mVec[0] = _mm512_and_si512(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_and_si512(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        inline SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_and_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_and_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_or_si512(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_or_si512(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_or_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_or_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BORS
        inline SIMDVec_i bor(int64_t b) const {
            __m512i t0 = _mm512_or_si512(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_or_si512(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator| (int64_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_or_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_or_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec[0] = _mm512_or_si512(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_or_si512(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_or_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_or_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int64_t b) {
            mVec[0] = _mm512_or_si512(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_or_si512(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        inline SIMDVec_i & operator|= (int64_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_or_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_or_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m512i t0 = _mm512_xor_si512(mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_xor_si512(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m512i t0 = _mm512_mask_xor_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            __m512i t1 = _mm512_mask_xor_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // BXORS
        inline SIMDVec_i bxor(int64_t b) const {
            __m512i t0 = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator^ (int64_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<16> const & mask, int64_t b) const {
            __m512i t0 = _mm512_mask_xor_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            __m512i t1 = _mm512_mask_xor_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return SIMDVec_i(t0, t1);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec[0] = _mm512_xor_si512(mVec[0], b.mVec[0]);
            mVec[1] = _mm512_xor_si512(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            mVec[0] = _mm512_mask_xor_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], b.mVec[0]);
            mVec[1] = _mm512_mask_xor_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], b.mVec[1]);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int64_t b) {
            mVec[0] = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        inline SIMDVec_i & operator^= (int64_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<16> const & mask, int64_t b) {
            mVec[0] = _mm512_mask_xor_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], _mm512_set1_epi64(b));
            mVec[1] = _mm512_mask_xor_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], _mm512_set1_epi64(b));
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m512i t0 = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            __m512i t1 = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            __m512i t1 = _mm512_mask_xor_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            __m512i t2 = _mm512_mask_xor_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            mVec[0] = _mm512_xor_si512(mVec[0], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            mVec[1] = _mm512_xor_si512(mVec[1], _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF));
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i & bnota(SIMDVecMask<16> const & mask) {
            __m512i t0 = _mm512_set1_epi64(0xFFFFFFFFFFFFFFFF);
            mVec[0] = _mm512_mask_xor_epi64(mVec[0], mask.mMask & 0xFF, mVec[0], t0);
            mVec[1] = _mm512_mask_xor_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1], t0);
            return *this;
        }
        // HBAND
        /*inline int64_t hband() const {
            return mVec[0][0] & mVec[0][1];
        }*/
        // MHBAND
        /*inline int64_t hband(SIMDVecMask<16> const & mask) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] : 0xFFFFFFFFFFFFFFFF;
            int64_t t1 = mask.mMask[1] ? mVec[0][1] & t0 : t0;
            return t1;
        }*/
        // HBANDS
        /*inline int64_t hband(int64_t b) const {
            return mVec[0][0] & mVec[0][1] & b;
        }*/
        // MHBANDS
        /*inline int64_t hband(SIMDVecMask<16> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] & b: b;
            int64_t t1 = mask.mMask[1] ? mVec[0][1] & t0: t0;
            return t1;
        }*/
        // HBOR
        /*inline int64_t hbor() const {
            return mVec[0][0] | mVec[0][1];
        }*/
        // MHBOR
        /*inline int64_t hbor(SIMDVecMask<16> const & mask) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] : 0;
            int64_t t1 = mask.mMask[1] ? mVec[0][1] | t0 : t0;
            return t1;
        }*/
        // HBORS
        /*inline int64_t hbor(int64_t b) const {
            return mVec[0][0] | mVec[0][1] | b;
        }*/
        // MHBORS
        /*inline int64_t hbor(SIMDVecMask<16> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] | b : b;
            int64_t t1 = mask.mMask[1] ? mVec[0][1] | t0 : t0;
            return t1;
        }*/
        // HBXOR
        /*inline int64_t hbxor() const {
            return mVec[0][0] ^ mVec[0][1];
        }*/
        // MHBXOR
        /*inline int64_t hbxor(SIMDVecMask<16> const & mask) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] : 0;
            int64_t t1 = mask.mMask[1] ? mVec[0][1] ^ t0 : t0;
            return t0;
        }*/
        // HBXORS
        /*inline int64_t hbxor(int64_t b) const {
            return mVec[0][0] ^ mVec[0][1] ^ b;
        }*/
        // MHBXORS
        /*inline int64_t hbxor(SIMDVecMask<16> const & mask, int64_t b) const {
            int64_t t0 = mask.mMask[0] ? mVec[0][0] ^ b : b;
            int64_t t1 = mask.mMask[1] ? mVec[0][1] ^ t0 : t0;
            return t1;
        }*/

        // GATHERS
        inline SIMDVec_i & gather(int64_t * baseAddr, int64_t* indices) {
            __m512i t0 = _mm512_load_si512((__m512i *)indices);
            __m512i t1 = _mm512_load_si512((__m512i *)(indices + 8));
            mVec[0] = _mm512_i64gather_epi64(t0, (__int64 const*)baseAddr, 8);
            mVec[1] = _mm512_i64gather_epi64(t1, (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<16> const & mask, int64_t* baseAddr, int64_t* indices) {
            __m512i t0 = _mm512_load_si512((__m512i *)indices);
            __m512i t1 = _mm512_load_si512((__m512i *)(indices + 8));
            __m512i t2 = _mm512_i64gather_epi64(t0, (__int64 const*)baseAddr, 8);
            __m512i t3 = _mm512_i64gather_epi64(t1, (__int64 const*)baseAddr, 8);
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], mask.mMask & 0xFF, t2);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), t3);
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int64_t * baseAddr, SIMDVec_i const & indices) {
            mVec[0] = _mm512_i64gather_epi64(indices.mVec[0], (__int64 const*)baseAddr, 8);
            mVec[1] = _mm512_i64gather_epi64(indices.mVec[1], (__int64 const*)baseAddr, 8);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<16> const & mask, int64_t* baseAddr, SIMDVec_i const & indices) {
            __m512i t0 = _mm512_i64gather_epi64(indices.mVec[0], (__int64 const*)baseAddr, 8);
            __m512i t1 = _mm512_i64gather_epi64(indices.mVec[1], (__int64 const*)baseAddr, 8);
            mVec[0] = _mm512_mask_mov_epi64(mVec[0], mask.mMask & 0xFF, t0);
            mVec[1] = _mm512_mask_mov_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), t1);
            return *this;
        }
        // SCATTERS
        inline int64_t* scatter(int64_t* baseAddr, int64_t* indices) const {
            __m512i t0 = _mm512_load_si512((__m512i *)indices);
            __m512i t1 = _mm512_load_si512((__m512i *)(indices + 8));
            _mm512_i64scatter_epi64(baseAddr, t0, mVec[0], 8);
            _mm512_i64scatter_epi64(baseAddr, t1, mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERS
        inline int64_t* scatter(SIMDVecMask<16> const & mask, int64_t* baseAddr, int64_t* indices) const {
            __m512i t0 = _mm512_load_si512((__m512i *)indices);
            __m512i t1 = _mm512_load_si512((__m512i *)(indices + 8));
            _mm512_mask_i64scatter_epi64(baseAddr, mask.mMask & 0xFF, t0, mVec[0], 8);
            _mm512_mask_i64scatter_epi64(baseAddr, ((mask.mMask & 0xFF00) >> 8), t1, mVec[1], 8);
            return baseAddr;
        }
        // SCATTERV
        inline int64_t* scatter(int64_t* baseAddr, SIMDVec_i const & indices) const {
            _mm512_i64scatter_epi64(baseAddr, indices.mVec[0], mVec[0], 8);
            _mm512_i64scatter_epi64(baseAddr, indices.mVec[1], mVec[1], 8);
            return baseAddr;
        }
        // MSCATTERV
        inline int64_t* scatter(SIMDVecMask<16> const & mask, int64_t* baseAddr, SIMDVec_i const & indices) const {
            _mm512_mask_i64scatter_epi64(baseAddr, mask.mMask & 0xFF, indices.mVec[0], mVec[0], 8);
            _mm512_mask_i64scatter_epi64(baseAddr, ((mask.mMask & 0xFF00) >> 8), indices.mVec[1], mVec[1], 8);
            return baseAddr;
        }

        //// LSHV
        //inline SIMDVec_i lsh(SIMDVec_i const & b) const {
        //    int64_t t0 = mVec[0][0] << b.mVec[0][0];
        //    int64_t t1 = mVec[0][1] << b.mVec[0][1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// MLSHV
        //inline SIMDVec_i lsh(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0][0] << b.mVec[0][0] : mVec[0][0];
        //    int64_t t1 = mask.mMask[1] ? mVec[0][1] << b.mVec[0][1] : mVec[0][1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// LSHS
        //inline SIMDVec_i lsh(int64_t b) const {
        //    int64_t t0 = mVec[0][0] << b;
        //    int64_t t1 = mVec[0][1] << b;
        //    return SIMDVec_i(t0, t1);
        //}
        //// MLSHS
        //inline SIMDVec_i lsh(SIMDVecMask<16> const & mask, int64_t b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0][0] << b : mVec[0][0];
        //    int64_t t1 = mask.mMask[1] ? mVec[0][1] << b : mVec[0][1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// LSHVA
        //inline SIMDVec_i & lsha(SIMDVec_i const & b) {
        //    mVec[0][0] = mVec[0][0] << b.mVec[0][0];
        //    mVec[0][1] = mVec[0][1] << b.mVec[0][1];
        //    return *this;
        //}
        //// MLSHVA
        //inline SIMDVec_i & lsha(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
        //    if(mask.mMask[0]) mVec[0][0] = mVec[0][0] << b.mVec[0][0];
        //    if(mask.mMask[1]) mVec[0][1] = mVec[0][1] << b.mVec[0][1];
        //    return *this;
        //}
        //// LSHSA
        //inline SIMDVec_i & lsha(int64_t b) {
        //    mVec[0][0] = mVec[0][0] << b;
        //    mVec[0][1] = mVec[0][1] << b;
        //    return *this;
        //}
        //// MLSHSA
        //inline SIMDVec_i & lsha(SIMDVecMask<16> const & mask, int64_t b) {
        //    if(mask.mMask[0]) mVec[0][0] = mVec[0][0] << b;
        //    if(mask.mMask[1]) mVec[0][1] = mVec[0][1] << b;
        //    return *this;
        //}
        //// RSHV
        //inline SIMDVec_i rsh(SIMDVec_i const & b) const {
        //    int64_t t0 = mVec[0][0] >> b.mVec[0][0];
        //    int64_t t1 = mVec[0][1] >> b.mVec[0][1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// MRSHV
        //inline SIMDVec_i rsh(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0][0] >> b.mVec[0][0] : mVec[0][0];
        //    int64_t t1 = mask.mMask[1] ? mVec[0][1] >> b.mVec[0][1] : mVec[0][1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// RSHS
        //inline SIMDVec_i rsh(int64_t b) const {
        //    int64_t t0 = mVec[0][0] >> b;
        //    int64_t t1 = mVec[0][1] >> b;
        //    return SIMDVec_i(t0, t1);
        //}
        //// MRSHS
        //inline SIMDVec_i rsh(SIMDVecMask<16> const & mask, int64_t b) const {
        //    int64_t t0 = mask.mMask[0] ? mVec[0][0] >> b : mVec[0][0];
        //    int64_t t1 = mask.mMask[1] ? mVec[0][1] >> b : mVec[0][1];
        //    return SIMDVec_i(t0, t1);
        //}
        //// RSHVA
        //inline SIMDVec_i & rsha(SIMDVec_i const & b) {
        //    mVec[0][0] = mVec[0][0] >> b.mVec[0][0];
        //    mVec[0][1] = mVec[0][1] >> b.mVec[0][1];
        //    return *this;
        //}
        //// MRSHVA
        //inline SIMDVec_i & rsha(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
        //    if (mask.mMask[0]) mVec[0][0] = mVec[0][0] >> b.mVec[0][0];
        //    if (mask.mMask[1]) mVec[0][1] = mVec[0][1] >> b.mVec[0][1];
        //    return *this;
        //}
        //// RSHSA
        //inline SIMDVec_i & rsha(int64_t b) {
        //    mVec[0][0] = mVec[0][0] >> b;
        //    mVec[0][1] = mVec[0][1] >> b;
        //    return *this;
        //}
        //// MRSHSA
        //inline SIMDVec_i & rsha(SIMDVecMask<16> const & mask, int64_t b) {
        //    if (mask.mMask[0]) mVec[0][0] = mVec[0][0] >> b;
        //    if (mask.mMask[1]) mVec[0][1] = mVec[0][1] >> b;
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
        inline SIMDVec_i neg() const {
            __m512i t0 = _mm512_sub_epi64(_mm512_setzero_si512(), mVec[0]);
            __m512i t1 = _mm512_sub_epi64(_mm512_setzero_si512(), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        inline SIMDVec_i neg(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, _mm512_setzero_si512(), mVec[0]);
            __m512i t1 = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_setzero_si512(), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // NEGA
        inline SIMDVec_i & nega() {
            mVec[0] = _mm512_sub_epi64(_mm512_setzero_si512(), mVec[0]);
            mVec[1] = _mm512_sub_epi64(_mm512_setzero_si512(), mVec[1]);
            return *this;
        }
        // MNEGA
        inline SIMDVec_i & nega(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_sub_epi64(mVec[0], mask.mMask & 0xFF, _mm512_setzero_si512(), mVec[0]);
            mVec[1] = _mm512_mask_sub_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), _mm512_setzero_si512(), mVec[1]);
            return *this;
        }
        // ABS
        inline SIMDVec_i abs() const {
            __m512i t0 = _mm512_abs_epi64(mVec[0]);
            __m512i t1 = _mm512_abs_epi64(mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<16> const & mask) const {
            __m512i t0 = _mm512_mask_abs_epi64(mVec[0], mask.mMask & 0xFF, mVec[0]);
            __m512i t1 = _mm512_mask_abs_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // ABSA
        inline SIMDVec_i & absa() {
            mVec[0] = _mm512_abs_epi64(mVec[0]);
            mVec[1] = _mm512_abs_epi64(mVec[1]);
            return *this;
        }
        // MABSA
        inline SIMDVec_i & absa(SIMDVecMask<16> const & mask) {
            mVec[0] = _mm512_mask_abs_epi64(mVec[0], mask.mMask & 0xFF, mVec[0]);
            mVec[1] = _mm512_mask_abs_epi64(mVec[1], ((mask.mMask & 0xFF00) >> 8), mVec[1]);
            return *this;
        }

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        void unpack(SIMDVec_i<int64_t, 8> & a, SIMDVec_i<int64_t, 8> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        SIMDVec_i<int64_t, 8> unpacklo() const {
            return SIMDVec_i<int64_t, 8> (mVec[0]);
        }
        // UNPACKHI
        SIMDVec_i<int64_t, 8> unpackhi() const {
            return SIMDVec_i<int64_t, 8>(mVec[1]);
        }

        // PROMOTE
        // -
        // DEGRADE
        inline operator SIMDVec_i<int32_t, 16>() const;

        // ITOU
        inline operator SIMDVec_u<uint64_t, 16>() const;
        // ITOF
        inline operator SIMDVec_f<double, 16>() const;
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
