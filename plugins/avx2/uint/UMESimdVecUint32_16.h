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

#ifndef UME_SIMD_VEC_UINT32_16_H_
#define UME_SIMD_VEC_UINT32_16_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 16> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint32_t, 16>,
            uint32_t,
            16,
            SIMDVecMask<16>,
            SIMDSwizzle<16>>,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint32_t, 16>,
            SIMDVec_u<uint32_t, 8>>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 16>;
        friend class SIMDVec_f<float, 16>;

        friend class SIMDVec_u<uint32_t, 32>;
    private:
        __m256i mVec[2];

        UME_FORCE_INLINE explicit SIMDVec_u(__m256i & x0, __m256i & x1) { mVec[0] = x0; mVec[1] = x1; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint32_t i) {
            mVec[0] = _mm256_set1_epi32(i);
            mVec[1] = _mm256_set1_epi32(i);
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
        UME_FORCE_INLINE SIMDVec_u(uint32_t i0,  uint32_t i1,  uint32_t i2,  uint32_t i3,
                         uint32_t i4,  uint32_t i5,  uint32_t i6,  uint32_t i7,
                         uint32_t i8,  uint32_t i9,  uint32_t i10, uint32_t i11,
                         uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15)
        {
            mVec[0] = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVec[1] = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        UME_FORCE_INLINE uint32_t extract(uint32_t index) const {
            alignas(32) uint32_t raw[8];
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
        UME_FORCE_INLINE uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint32_t value) {
            alignas(32) uint32_t raw[8];
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
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
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
            mVec[0] = _mm256_blendv_epi8(mVec[0], b.mVec[0], mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], b.mVec[1], mask.mMask[1]);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint32_t b) {
            mVec[0] = _mm256_set1_epi32(b);
            mVec[1] = _mm256_set1_epi32(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i*)p);
            mVec[1] = _mm256_loadu_si256((__m256i*)(p + 8));
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<16> const & mask, uint32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint32_t const * p) {
            mVec[0] = _mm256_load_si256((__m256i *)p);
            mVec[1] = _mm256_load_si256((__m256i *)(p + 8));
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<16> const & mask, uint32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint32_t* store(uint32_t * p) const {
            _mm256_storeu_si256((__m256i*)p, mVec[0]);
            _mm256_storeu_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint32_t* store(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm256_maskstore_epi32((int*)p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_epi32((int*)(p + 8), mask.mMask[1], mVec[1]);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint32_t* storea(uint32_t * p) const {
            _mm256_store_si256((__m256i*)p, mVec[0]);
            _mm256_store_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint32_t* storea(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm256_maskstore_epi32((int*)p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_epi32((int*)(p + 8), mask.mMask[1], mVec[1]);
            return p;
        }
        // BLENDV
        // BLENDS
        // SWIZZLE 
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = _mm256_set1_epi32(b);
            __m256i t3 = _mm256_add_epi32(mVec[1], t2);
            return SIMDVec_u(t1, t3);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_add_epi32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec[0] = _mm256_add_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint32_t b) {
            mVec[0] = _mm256_add_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_add_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<16> const & mask, uint32_t b) {
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
        UME_FORCE_INLINE SIMDVec_u postinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            mVec[0] = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = mVec[1];
            mVec[1] = _mm256_add_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            __m256i t2 = _mm256_add_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t2, mask.mMask[0]);
            __m256i t3 = mVec[1];
            __m256i t4 = _mm256_add_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec[0] = _mm256_add_epi32(mVec[0], t0);
            mVec[1] = _mm256_add_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec[0] = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint32_t b) {
            mVec[0] = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<16> const & mask, uint32_t b) {
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
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            __m256i t1 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(b.mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(b.mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[0]);
            __m256i t1 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, t1, mask.mMask[0]);
            __m256i t3 = _mm256_sub_epi32(t0, mVec[1]);
            __m256i t4 = _mm256_blendv_epi8(t0, t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec[0] = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            mVec[1] = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            mVec[0] = _mm256_blendv_epi8(b.mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            mVec[1] = _mm256_blendv_epi8(b.mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(uint32_t b) {
            mVec[0] = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[0]);
            mVec[1] = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u subfroma(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec[0]);
            mVec[0] = _mm256_blendv_epi8(t0, t1, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(t0, mVec[1]);
            mVec[1] = _mm256_blendv_epi8(t0, t2, mask.mMask[1]);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            mVec[0] = _mm256_sub_epi32(mVec[0], t0);
            __m256i t2 = mVec[1];
            mVec[1] = _mm256_sub_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            __m256i t2 = _mm256_sub_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t2, mask.mMask[0]);
            __m256i t3 = mVec[1];
            __m256i t4 = _mm256_sub_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec[0] = _mm256_sub_epi32(mVec[0], t0);
            mVec[1] = _mm256_sub_epi32(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_sub_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec[0] = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint32_t b) {
            mVec[0] = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
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
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(SIMDVec_u const & b) const {
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<16> cmpeq(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t0);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_xor_si256(m0, t0);
            __m256i m2 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            __m256i m3 = _mm256_xor_si256(m2, t0);
            return SIMDVecMask<16>(m1, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<16> cmpne(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i m1 = _mm256_xor_si256(m0, t1);
            __m256i m2 = _mm256_cmpeq_epi32(mVec[1], t0);
            __m256i m3 = _mm256_xor_si256(m2, t1);
            return SIMDVecMask<16>(m1, m3);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(b.mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t1, t2);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_xor_si256(b.mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t3, t4);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<16> cmpgt(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_set1_epi32(b ^ 0x80000000);
            __m256i t2 = _mm256_xor_si256(mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t3, t1);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(b.mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_xor_si256(b.mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t4, t3);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<16> cmplt(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t2 = _mm256_set1_epi32(b ^ 0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t2, t3);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<16> cmpge(uint32_t b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], _mm256_set1_epi32(b));
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i t1 = _mm256_max_epu32(mVec[1], _mm256_set1_epi32(b));
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<16> cmple(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i m0 = _mm256_cmpeq_epi32(b.mVec[0], t0);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            __m256i m1 = _mm256_cmpeq_epi32(b.mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<16> cmple(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0],t0);
            __m256i m0 = _mm256_cmpeq_epi32(t0, t1);
            __m256i t2 = _mm256_max_epu32(mVec[1], t1);
            __m256i m1 = _mm256_cmpeq_epi32(t0, t2);
            return SIMDVecMask<16>(m0, m1);
        }
        UME_FORCE_INLINE SIMDVecMask<16> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_u const & b) const {
            alignas(32) uint32_t raw[16];
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
        UME_FORCE_INLINE bool cmpe(uint32_t b) const {
            alignas(32) uint32_t raw[16];
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
        UME_FORCE_INLINE bool unique() const {
            alignas(32) uint32_t raw[16];
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
        UME_FORCE_INLINE uint32_t hadd() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_add_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            uint32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval;
        }
        // MHADD
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_add_epi32(t1, t2);
            __m256i t4 = _mm256_hadd_epi32(t3, t0);
            __m256i t5 = _mm256_hadd_epi32(t4, t0);
            uint32_t retval = _mm256_extract_epi32(t5, 0);
            retval += _mm256_extract_epi32(t5, 4);
            return retval;
        }
        // HADDS
        UME_FORCE_INLINE uint32_t hadd(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_add_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            uint32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval + b;
        }
        // MHADDS
        UME_FORCE_INLINE uint32_t hadd(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_add_epi32(t1, t2);
            __m256i t4 = _mm256_hadd_epi32(t3, t0);
            __m256i t5 = _mm256_hadd_epi32(t4, t0);
            uint32_t retval = _mm256_extract_epi32(t5, 0);
            retval += _mm256_extract_epi32(t5, 4);
            return retval + b;
        }
        // HMUL
        UME_FORCE_INLINE uint32_t hmul() const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_mullo_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_mullo_epi32(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHMUL
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<16> const & mask) const {
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
            uint32_t retval  = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HMULS
        UME_FORCE_INLINE uint32_t hmul(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_mullo_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_mullo_epi32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_mullo_epi32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_mullo_epi32(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval * b;
        }
        // MHMULS
        UME_FORCE_INLINE uint32_t hmul(SIMDVecMask<16> const & mask, uint32_t b) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval * b;
        }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_add_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_add_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_sub_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_sub_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_mullo_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_mullo_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_mullo_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_mullo_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0], t0);
            __m256i t2 = _mm256_max_epu32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_max_epu32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec[0] = _mm256_max_epu32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_max_epu32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_max_epu32(mVec[0], t0);
            mVec[1] = _mm256_max_epu32(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_max_epu32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_min_epu32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_min_epu32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec[0], t0);
            __m256i t2 = _mm256_min_epu32(mVec[1], t1);
            return SIMDVec_u(t1, t2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_min_epu32(mVec[1], t1);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec[0] = _mm256_min_epu32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_min_epu32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_min_epu32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_min_epu32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_u & mina(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_min_epu32(mVec[0], t0);
            mVec[1] = _mm256_min_epu32(mVec[1], t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_min_epu32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HMAX
        UME_FORCE_INLINE uint32_t hmax() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::lowest());
            __m256i t1 = _mm256_max_epu32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_max_epu32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_max_epu32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_max_epu32(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHMAX
        UME_FORCE_INLINE uint32_t hmax(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::lowest());
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_max_epu32(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_max_epu32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_max_epu32(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_max_epu32(t8, t7);
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // IMAX
        // MIMAX
        // HMIN
        UME_FORCE_INLINE uint32_t hmin() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
            __m256i t1 = _mm256_min_epu32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_min_epu32(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_min_epu32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_min_epu32(t6, t5);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHMIN
        UME_FORCE_INLINE uint32_t hmin(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
            __m256i t1 = _mm256_blendv_epi8(t0, mVec[0], mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, mVec[1], mask.mMask[1]);
            __m256i t3 = _mm256_min_epu32(t1, t2);
            __m256i t4 = _mm256_permute2f128_si256(t3, t0, 1);
            __m256i t5 = _mm256_min_epu32(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0xB);
            __m256i t7 = _mm256_min_epu32(t5, t6);
            __m256i t8 = _mm256_shuffle_epi32(t7, 0x1);
            __m256i t9 = _mm256_min_epu32(t8, t7);
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_and_si256(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_and_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            __m256i t2 = _mm256_and_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_and_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec[0] = _mm256_and_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_and_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_and_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_and_si256(mVec[0], t0);
            mVec[1] = _mm256_and_si256(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_and_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_or_si256(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_or_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            __m256i t2 = _mm256_or_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_or_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec[0] = _mm256_or_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_or_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_or_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_or_si256(mVec[0], t0);
            mVec[1] = _mm256_or_si256(mVec[1], t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_or_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec[0] = _mm256_xor_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_xor_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_xor_si256(mVec[0], t0);
            __m256i t1 = _mm256_set1_epi32(b);
            mVec[1] = _mm256_xor_si256(mVec[1], t1);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm256_xor_si256(mVec[0], t0);
            mVec[1] = _mm256_xor_si256(mVec[1], t0);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u bnota(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HBAND
        UME_FORCE_INLINE uint32_t hband() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_and_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_and_si256(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHBAND
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<16> const & mask) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HBANDS
        UME_FORCE_INLINE uint32_t hband(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_and_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_and_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_and_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_and_si256(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval & b;
        }
        // MHBANDS
        UME_FORCE_INLINE uint32_t hband(SIMDVecMask<16> const & mask, uint32_t b) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval & b;
        }
        // HBOR
        UME_FORCE_INLINE uint32_t hbor() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_or_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_or_si256(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHBOR
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<16> const & mask) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HBORS
        UME_FORCE_INLINE uint32_t hbor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_or_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_or_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_or_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_or_si256(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval | b;
        }
        // MHBORS
        UME_FORCE_INLINE uint32_t hbor(SIMDVecMask<16> const & mask, uint32_t b) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval | b;
        }
        // HBXOR
        UME_FORCE_INLINE uint32_t hbxor() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_xor_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_xor_si256(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval;
        }
        // MHBXOR
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<16> const & mask) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval;
        }
        // HBXORS
        UME_FORCE_INLINE uint32_t hbxor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_xor_si256(mVec[0], mVec[1]);
            __m256i t2 = _mm256_permute2f128_si256(t1, t0, 1);
            __m256i t3 = _mm256_xor_si256(t1, t2);
            __m256i t4 = _mm256_shuffle_epi32(t3, 0xB);
            __m256i t5 = _mm256_xor_si256(t3, t4);
            __m256i t6 = _mm256_shuffle_epi32(t5, 0x1);
            __m256i t7 = _mm256_xor_si256(t5, t6);
            uint32_t retval = _mm256_extract_epi32(t7, 0);
            return retval ^ b;
        }
        // MHBXORS
        UME_FORCE_INLINE uint32_t hbxor(SIMDVecMask<16> const & mask, uint32_t b) const {
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
            uint32_t retval = _mm256_extract_epi32(t9, 0);
            return retval ^ b;
        }
        // GATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec[0] = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(indices + 8));
            mVec[1] = _mm256_i32gather_epi32((const int *)baseAddr, t1, 4);
            return *this;
        }
        // MGATHERS
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t const * baseAddr, uint32_t const * indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_loadu_si256((__m256i*)(indices + 8));
            __m256i t3 = _mm256_i32gather_epi32((const int *)baseAddr, t2, 4);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return *this;
        }
        // GATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[0], 4);
            mVec[1] = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[1], 4);
            return *this;
        }
        // MGATHERV
        UME_FORCE_INLINE SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
            __m256i t0 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[0], 4);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[1], 4);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SCATTERS
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            alignas(32) uint32_t raw[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            for (int i = 0; i < 16; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            alignas(32) uint32_t raw[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 16; i++) { if (rawMask[i] == SIMDVecMask<16>::TRUE_VAL()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        UME_FORCE_INLINE uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
            alignas(32) uint32_t raw[16];
            alignas(32) uint32_t rawIndices[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawIndices + 8), indices.mVec[1]);
            for (int i = 0; i < 16; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        UME_FORCE_INLINE uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
            alignas(32) uint32_t raw[16];
            alignas(32) uint32_t rawIndices[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawIndices + 8), indices.mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 16; i++) {
                if (rawMask[i] == SIMDVecMask<16>::TRUE_VAL())
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

        // PACK
        UME_FORCE_INLINE SIMDVec_u & pack(SIMDVec_u<uint32_t, 8> const & a, SIMDVec_u<uint32_t, 8> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        UME_FORCE_INLINE SIMDVec_u & packlo(SIMDVec_u<uint32_t, 8> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        UME_FORCE_INLINE SIMDVec_u & packhi(SIMDVec_u<uint32_t, 8> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        UME_FORCE_INLINE void unpack(SIMDVec_u<uint32_t, 8> & a, SIMDVec_u<uint32_t, 8> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 8> unpacklo() const {
            return SIMDVec_u<uint32_t, 8>(mVec[0]);
        }
        // UNPACKHI
        UME_FORCE_INLINE SIMDVec_u<uint32_t, 8> unpackhi() const {
            return SIMDVec_u<uint32_t, 8>(mVec[1]);
        }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 16>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint16_t, 16>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 16>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 16>() const;

    };

}
}

#endif
