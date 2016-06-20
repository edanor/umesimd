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

#ifndef UME_SIMD_VEC_INT32_16_H_
#define UME_SIMD_VEC_INT32_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 16> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 16>,
            SIMDVec_u<uint32_t, 16>,
            int32_t,
            16,
            uint32_t,
            SIMDVecMask<16>,
            SIMDSwizzle<16>>,
        public SIMDVecPackableInterface<
            SIMDVec_i<int32_t, 16>,
            SIMDVec_i<int32_t, 8>>
    {
    public:
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __m256i mVec[2];

        inline explicit SIMDVec_i(__m256i & x0, __m256i & x1) { mVec[0] = x0; mVec[1] = x1; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_i() {}
        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec[0] = _mm256_set1_epi32(i);
            mVec[1] = _mm256_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const *p) { load(p); }
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0,  int32_t i1,  int32_t i2,  int32_t i3,
                         int32_t i4,  int32_t i5,  int32_t i6,  int32_t i7,
                         int32_t i8,  int32_t i9,  int32_t i10, int32_t i11,
                         int32_t i12, int32_t i13, int32_t i14, int32_t i15)
        {
            mVec[0] = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVec[1] = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            alignas(32) int32_t raw[8];
            uint32_t raw_index;
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mVec[0]);
                raw_index = index;
            }
            else {
                _mm256_store_si256((__m256i*)raw, mVec[1]);
                raw_index = index - 8;
            }
            return raw[raw_index];
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            alignas(32) int32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mVec[0]);
                raw[index] = value;
                mVec[0] = _mm256_load_si256((__m256i*)raw);
            }
            else
            {
                _mm256_store_si256((__m256i*)raw, mVec[1]);
                raw[index - 8] = value;
                mVec[1] = _mm256_load_si256((__m256i*)raw);
            }
            return *this;
        }
        inline IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_i &>(*this));
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
            mVec[0] = _mm256_blendv_epi8(mVec[0], b.mVec[0], mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], b.mVec[1], mask.mMask[1]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t b) {
            mVec[0] = _mm256_set1_epi32(b);
            mVec[1] = _mm256_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_i & load(int32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i*)p);
            mVec[1] = _mm256_loadu_si256((__m256i*)(p + 8));
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<16> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec[0] = _mm256_load_si256((__m256i *)p);
            mVec[1] = _mm256_load_si256((__m256i *)(p + 8));
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<16> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SLOAD
        inline SIMDVec_i & sload(int32_t const * p) {
            mVec[0] = _mm256_stream_load_si256((__m256i*)p);
            mVec[1] = _mm256_stream_load_si256((__m256i*)(p + 8));
            return *this;
        }
        // MSLOAD
        inline SIMDVec_i & sload(SIMDVecMask<16> const & mask, int32_t const * p) {
            __m256i t0 = _mm256_stream_load_si256((__m256i*)p);
            __m256i t1 = _mm256_stream_load_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // STORE
        inline int32_t* store(int32_t * p) const {
            _mm256_storeu_si256((__m256i*)p, mVec[0]);
            _mm256_storeu_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        inline int32_t* store(SIMDVecMask<16> const & mask, int32_t * p) const {
            _mm256_maskstore_epi32(p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_epi32(p + 8, mask.mMask[1], mVec[1]);
            return p;
        }
        // STOREA
        inline int32_t* storea(int32_t * p) const {
            _mm256_store_si256((__m256i*)p, mVec[0]);
            _mm256_store_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTOREA
        inline int32_t* storea(SIMDVecMask<16> const & mask, int32_t * p) const {
            _mm256_maskstore_epi32(p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_epi32(p + 8, mask.mMask[1], mVec[1]);
            return p;
        }
        // SSTORE
        inline int32_t* sstore(int32_t* p) const {
            _mm256_stream_si256((__m256i*)p, mVec[0]);
            _mm256_stream_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSSTORE
        inline int32_t* sstore(SIMDVecMask<16> const & mask, int32_t* p) const {
            __m256i t0 = _mm256_stream_load_si256((__m256i*)p);
            __m256i t1 = _mm256_stream_load_si256((__m256i*)(p + 8));
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t3 = _mm256_blendv_epi8(t1, mVec[1], mask.mMask[1]);
            _mm256_stream_si256((__m256i*)p, t2);
            _mm256_stream_si256((__m256i*)(p + 8), t3);
            return p;
        }
        // BLENDV
        // BLENDS
        // SWIZZLE 
        // SWIZZLEA

        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = _mm256_set1_epi32(b);
            __m256i t3 = _mm256_add_epi32(mVec[1], t2);
            return SIMDVec_i(t1, t3);
        }
        inline SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_add_epi32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec[0] = _mm256_add_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // ADDSA
        inline SIMDVec_i & adda(int32_t b) {
            mVec[0] = _mm256_add_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_add_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_add_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], _mm256_set1_epi32(b));
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
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
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            mVec[0] = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = mVec[1];
            mVec[1] = _mm256_add_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            __m256i t2 = _mm256_add_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t2, mask.mMask[0]);
            __m256i t3 = mVec[1];
            __m256i t4 = _mm256_add_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec[0] = _mm256_add_epi32(mVec[0], t0);
            mVec[1] = _mm256_add_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec[0] = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            mVec[0] = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
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
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            __m256i t1 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(b.mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(b.mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[0]);
            __m256i t1 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, t1, mask.mMask[0]);
            __m256i t3 = _mm256_sub_epi32(t0, mVec[1]);
            __m256i t4 = _mm256_blendv_epi8(t0, t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec[0] = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            mVec[1] = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            mVec[0] = _mm256_blendv_epi8(b.mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            mVec[1] = _mm256_blendv_epi8(b.mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            mVec[0] = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[0]);
            mVec[1] = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i subfroma(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec[0]);
            mVec[0] = _mm256_blendv_epi8(t0, t1, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(t0, mVec[1]);
            mVec[1] = _mm256_blendv_epi8(t0, t2, mask.mMask[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_i postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            mVec[0] = _mm256_sub_epi32(mVec[0], t0);
            __m256i t2 = mVec[1];
            mVec[1] = _mm256_sub_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            __m256i t2 = _mm256_sub_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t2, mask.mMask[0]);
            __m256i t3 = mVec[1];
            __m256i t4 = _mm256_sub_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec[0] = _mm256_sub_epi32(mVec[0], t0);
            mVec[1] = _mm256_sub_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_sub_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec[0] = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            mVec[0] = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // DIVV
        inline SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        // DIVS
        inline SIMDVec_i operator/ (int32_t b) const {
            return div(b);
        }
        // MDIVS
        // DIVVA
        inline SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        inline SIMDVec_i & operator/= (int32_t b) {
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
        inline SIMDVecMask<16> cmpeq(SIMDVec_i const & b) const {
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator==(SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t0);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_xor_si256(m0, t0);
            __m256i m2 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            __m256i m3 = _mm256_xor_si256(m2, t0);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i m1 = _mm256_xor_si256(m0, t1);
            __m256i m2 = _mm256_cmpeq_epi32(mVec[1], t0);
            __m256i m3 = _mm256_xor_si256(m2, t1);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator!= (int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_i const & b) const {
            __m256i m0 = _mm256_cmpgt_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_cmpgt_epi32(mVec[1], b.mVec[1]);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpgt_epi32(mVec[0], t0);
            __m256i m1 = _mm256_cmpgt_epi32(mVec[1], t0);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_i const & b) const {
            __m256i m0 = _mm256_cmpgt_epi32(b.mVec[0], mVec[0]);
            __m256i m1 = _mm256_cmpgt_epi32(b.mVec[1], mVec[1]);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpgt_epi32(t0, mVec[0]);
            __m256i m1 = _mm256_cmpgt_epi32(t0, mVec[1]);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator< (int32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_max_epi32(mVec[0], b.mVec[0]);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i t1 = _mm256_max_epi32(mVec[1], b.mVec[1]);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(int32_t b) const {
            __m256i t0 = _mm256_max_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i t1 = _mm256_max_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_max_epi32(mVec[0], b.mVec[0]);
            __m256i m0 = _mm256_cmpeq_epi32(b.mVec[0], t0);
            __m256i t1 = _mm256_max_epi32(mVec[1], b.mVec[1]);
            __m256i m1 = _mm256_cmpeq_epi32(b.mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epi32(mVec[0],t0);
            __m256i m0 = _mm256_cmpeq_epi32(t0, t1);
            __m256i t2 = _mm256_max_epi32(mVec[1], t0);
            __m256i m1 = _mm256_cmpeq_epi32(t0, t2);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            alignas(32) int32_t raw[16];
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            _mm256_store_si256((__m256i*)raw, m0);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            _mm256_store_si256((__m256i*)(raw + 8), m1);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0) &&
                   (raw[8] != 0) && (raw[9] != 0) && (raw[10] != 0) && (raw[11] != 0) &&
                   (raw[12] != 0) && (raw[13] != 0) && (raw[14] != 0) && (raw[15] != 0);
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            alignas(32) int32_t raw[16];
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            _mm256_store_si256((__m256i*)raw, m0);
            __m256i t1 = _mm256_set1_epi32(b);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            _mm256_store_si256((__m256i*)(raw + 8), m1);
            return (raw[0] != 0) && (raw[1] != 0) && (raw[2] != 0) && (raw[3] !=0) &&
                   (raw[4] != 0) && (raw[5] != 0) && (raw[6] != 0) && (raw[7] !=0) &&
                   (raw[8] != 0) && (raw[9] != 0) && (raw[10] != 0) && (raw[11] != 0) &&
                   (raw[12] != 0) && (raw[13] != 0) && (raw[14] != 0) && (raw[15] != 0);
        }
        // UNIQUE
        inline bool unique() const {
            alignas(32) int32_t raw[16];
            _mm256_store_si256((__m256i *)raw, mVec[0]);
            _mm256_store_si256((__m256i *)(raw + 8), mVec[1]);
            for (unsigned int i = 0; i < 15; i++) {
                for (unsigned int j = i + 1; j < 16; j++) {
                    if (raw[i] == raw[j]) {
                        return false;
                    }
                }
            }
            return true;
        }
        // HADD
        inline int32_t hadd() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_add_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            int32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval;
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_add_epi32(t1, t2);
            __m256i t4 = _mm256_hadd_epi32(t3, t0);
            __m256i t5 = _mm256_hadd_epi32(t4, t0);
            int32_t retval = _mm256_extract_epi32(t5, 0);
            retval += _mm256_extract_epi32(t5, 4);
            return retval;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_add_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            int32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval + b;
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_add_epi32(t1, t2);
            __m256i t4 = _mm256_hadd_epi32(t3, t0);
            __m256i t5 = _mm256_hadd_epi32(t4, t0);
            int32_t retval = _mm256_extract_epi32(t5, 0);
            retval += _mm256_extract_epi32(t5, 4);
            return retval + b;
        }
        // HMUL
        inline int32_t hmul() const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_mullo_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_mullo_epi32(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_mullo_epi32(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_mullo_epi32(t7, t8);
            int32_t retval  = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_mullo_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_mullo_epi32(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval * b;
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_mullo_epi32(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_mullo_epi32(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval * b;
        }

        // FMULADDV
        inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_add_epi32(t2, c.mVec[1]);
            return SIMDVec_i(t1, t3);
        }
        // MFMULADDV
        inline SIMDVec_i fmuladd(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_add_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_i(t2, t5);
        }
        // FMULSUBV
        inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_sub_epi32(t2, c.mVec[1]);
            return SIMDVec_i(t1, t3);
        }
        // MFMULSUBV
        inline SIMDVec_i fmulsub(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_sub_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_i(t2, t5);
        }
        // FADDMULV
        inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_mullo_epi32(t2, c.mVec[1]);
            return SIMDVec_i(t1, t3);
        }
        // MFADDMULV
        inline SIMDVec_i faddmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_mullo_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_i(t2, t5);
        }
        // FSUBMULV
        inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_mullo_epi32(t2, c.mVec[1]);
            return SIMDVec_i(t1, t3);
        }
        // MFSUBMULV
        inline SIMDVec_i fsubmul(SIMDVecMask<16> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_mullo_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_i(t2, t5);
        }

        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_max_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_max_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_max_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_max_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epi32(mVec[0], t0);
            __m256i t2 = _mm256_max_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epi32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_max_epi32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec[0] = _mm256_max_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_max_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_max_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_max_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_max_epi32(mVec[0], t0);
            mVec[1] = _mm256_max_epi32(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_max_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_min_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_min_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_min_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_min_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epi32(mVec[0], t0);
            __m256i t2 = _mm256_min_epi32(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epi32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_min_epi32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec[0] = _mm256_min_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_min_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_min_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_min_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_min_epi32(mVec[0], t0);
            mVec[1] = _mm256_min_epi32(mVec[1], t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_min_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HMAX
        inline int32_t hmax() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<int32_t>::min());
            __m256i t1 = _mm256_max_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_max_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_max_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_max_epi32(t6, t5);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHMAX
        inline int32_t hmax(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<int32_t>::min());
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
            __m256i t3 = _mm256_max_epi32(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_max_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_max_epi32(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_max_epi32(t8, t7);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        inline int32_t hmin() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<int32_t>::max());
            __m256i t1 = _mm256_min_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_min_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_min_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_min_epi32(t6, t5);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHMIN
        inline int32_t hmin(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<int32_t>::max());
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
            __m256i t3 = _mm256_min_epi32(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_min_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_min_epi32(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_min_epi32(t8, t7);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_and_si256(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_and_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            __m256i t2 = _mm256_and_si256(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_and_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec[0] = _mm256_and_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_and_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_and_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_and_si256(mVec[0], t0);
            mVec[1] = _mm256_and_si256(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_and_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_or_si256(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_or_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            __m256i t2 = _mm256_or_si256(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_or_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec[0] = _mm256_or_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_or_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_or_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_or_si256(mVec[0], t0);
            mVec[1] = _mm256_or_si256(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_or_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            return SIMDVec_i(t0, t1);
        }
        inline SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<16> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_i(t1, t3);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec[0] = _mm256_xor_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_xor_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<16> const & mask, SIMDVec_i const & b) {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_xor_si256(mVec[0], t0);
            __m256i t1 = _mm256_set1_epi32(b);
            mVec[1] = _mm256_xor_si256(mVec[1], t1);
            return *this;
        }
        inline SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<16> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            return SIMDVec_i(t1, t2);
        }
        inline SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_i(t2, t4);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm256_xor_si256(mVec[0], t0);
            mVec[1] = _mm256_xor_si256(mVec[1], t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i bnota(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_and_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_and_si256(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_and_si256(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_and_si256(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_and_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_and_si256(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval & b;
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_and_si256(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_and_si256(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval & b;
        }
        // HBOR
        inline int32_t hbor() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_or_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_or_si256(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_or_si256(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_or_si256(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_or_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_or_si256(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval | b;
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_or_si256(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_or_si256(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval | b;
        }
        // HBXOR
        inline int32_t hbxor() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_xor_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_xor_si256(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_xor_si256(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_xor_si256(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HBXORS
        inline int32_t hbxor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_xor_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_xor_si256(t5, t6);
            int32_t retval = _mm256_extract_epi32(t7, 0);
            return retval ^ b;
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<16> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_xor_si256(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_xor_si256(t7, t8);
            int32_t retval = _mm256_extract_epi32(t9, 0);
            return retval ^ b;
        }
        // GATHERS
        inline SIMDVec_i & gather(int32_t* baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec[0] = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(indices + 8));
            mVec[1] = _mm256_i32gather_epi32((const int *)baseAddr, t1, 4);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<16> const & mask, int32_t* baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_loadu_si256((__m256i*)(indices + 8));
            __m256i t3 = _mm256_i32gather_epi32((const int *)baseAddr, t2, 4);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t* baseAddr, SIMDVec_i const & indices) {
            mVec[0] = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[0], 4);
            mVec[1] = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[1], 4);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<16> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
            __m256i t0 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[0], 4);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[1], 4);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint32_t* indices) const {
            alignas(32) int32_t raw[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            for (int i = 0; i < 16; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<16> const & mask, int32_t* baseAddr, uint32_t* indices) const {
            alignas(32) int32_t raw[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 16; i++) { if (rawMask[i] == SIMDVecMask<16>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) const {
            alignas(32) int32_t raw[16];
            alignas(32) uint32_t rawIndices[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawIndices + 8), indices.mVec[1]);
            for (int i = 0; i < 16; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<16> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) const {
            alignas(32) int32_t raw[16];
            alignas(32) uint32_t rawIndices[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawIndices + 8), indices.mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 16; i++) {
                if (rawMask[i] == SIMDVecMask<16>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
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
        inline SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        // NEGA
        // MNEGA
        // ABS
        inline SIMDVec_i abs() const {
            __m128i t0 = _mm256_extractf128_si256(mVec[0], 0);
            __m128i t1 = _mm256_extractf128_si256(mVec[0], 1);
            __m256i t2 = _mm256_setzero_si256();
            t2 = _mm256_insertf128_si256(t2, _mm_abs_epi32(t0), 0);
            t2 = _mm256_insertf128_si256(t2, _mm_abs_epi32(t1), 1);
            __m128i t3 = _mm256_extractf128_si256(mVec[1], 0);
            __m128i t4 = _mm256_extractf128_si256(mVec[1], 1);
            __m256i t5 = _mm256_setzero_si256();
            t5 = _mm256_insertf128_si256(t5, _mm_abs_epi32(t3), 0);
            t5 = _mm256_insertf128_si256(t5, _mm_abs_epi32(t4), 1);
            return SIMDVec_i(t2, t5);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<16> const & mask) const {
            __m128i t0 = _mm256_extractf128_si256(mVec[0], 0);
            __m128i t1 = _mm256_extractf128_si256(mVec[0], 1);
            __m128i t2 = _mm256_extractf128_si256(mask.mMask[0], 0);
            __m128i t3 = _mm256_extractf128_si256(mask.mMask[0], 1);
            __m128i t4 = _mm_blendv_epi8(t0, _mm_abs_epi32(t0), t2);
            __m128i t5 = _mm_blendv_epi8(t1, _mm_abs_epi32(t1), t3);
            __m256i t6 = _mm256_setzero_si256();
            t6 = _mm256_insertf128_si256(t6, t4, 0);
            t6 = _mm256_insertf128_si256(t6, t5, 1);
            __m128i t7 = _mm256_extractf128_si256(mVec[1], 0);
            __m128i t8 = _mm256_extractf128_si256(mVec[1], 1);
            __m128i t9 = _mm256_extractf128_si256(mask.mMask[1], 0);
            __m128i t10 = _mm256_extractf128_si256(mask.mMask[1], 1);
            __m128i t11 = _mm_blendv_epi8(t7, _mm_abs_epi32(t7), t9);
            __m128i t12 = _mm_blendv_epi8(t8, _mm_abs_epi32(t8), t10);
            __m256i t13 = _mm256_setzero_si256();
            t13 = _mm256_insertf128_si256(t13, t11, 0);
            t13 = _mm256_insertf128_si256(t13, t12, 1);
            return SIMDVec_i(t6, t13);
        }
        // ABSA
        // MABSA

        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 8> const & a, SIMDVec_i<int32_t, 8> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        inline SIMDVec_i & packlo(SIMDVec_i<int32_t, 8> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        inline SIMDVec_i & packhi(SIMDVec_i<int32_t, 8> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 8> & a, SIMDVec_i<int32_t, 8> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 8> unpacklo() const {
            return SIMDVec_i<int32_t, 8>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 8> unpackhi() const {
            return SIMDVec_i<int32_t, 8>(mVec[1]);
        }

        // PROMOTE
        inline operator SIMDVec_i<int64_t, 16>() const;
        // DEGRADE
        inline operator SIMDVec_i<int16_t, 16>() const;

        // ITOU
        inline operator SIMDVec_u<uint32_t, 16>() const;
        // ITOF
        inline operator SIMDVec_f<float, 16>() const;
    };

}
}

#endif
