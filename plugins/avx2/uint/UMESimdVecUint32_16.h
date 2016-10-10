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

        inline explicit SIMDVec_u(__m256i & x0, __m256i & x1) { mVec[0] = x0; mVec[1] = x1; }
    public:

        constexpr static uint32_t length() { return 16; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_u() {}
        // SET-CONSTR
        inline SIMDVec_u(uint32_t i) {
            mVec[0] = _mm256_set1_epi32(i);
            mVec[1] = _mm256_set1_epi32(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        inline SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_same<T, int>::value && 
                                    !std::is_same<T, uint32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint32_t>(i)) {}
        // LOAD-CONSTR
        inline explicit SIMDVec_u(uint32_t const *p) { load(p); }
        // FULL-CONSTR
        inline SIMDVec_u(uint32_t i0,  uint32_t i1,  uint32_t i2,  uint32_t i3,
                         uint32_t i4,  uint32_t i5,  uint32_t i6,  uint32_t i7,
                         uint32_t i8,  uint32_t i9,  uint32_t i10, uint32_t i11,
                         uint32_t i12, uint32_t i13, uint32_t i14, uint32_t i15)
        {
            mVec[0] = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
            mVec[1] = _mm256_setr_epi32(i8, i9, i10, i11, i12, i13, i14, i15);
        }
        // EXTRACT
        inline uint32_t extract(uint32_t index) const {
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
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
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
        inline IntermediateIndex<SIMDVec_u, uint32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint32_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator() (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        inline IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>> operator[] (SIMDVecMask<16> const & mask) {
            return IntermediateMask<SIMDVec_u, uint32_t, SIMDVecMask<16>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        inline SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec[0] = b.mVec[0];
            mVec[1] = b.mVec[1];
            return *this;
        }
        inline SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        inline SIMDVec_u & assign(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            mVec[0] = _mm256_blendv_epi8(mVec[0], b.mVec[0], mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], b.mVec[1], mask.mMask[1]);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_u & assign(uint32_t b) {
            mVec[0] = _mm256_set1_epi32(b);
            mVec[1] = _mm256_set1_epi32(b);
            return *this;
        }
        inline SIMDVec_u & operator= (uint32_t b) {
            return assign(b);
        }
        // MASSIGNS
        inline SIMDVec_u & assign(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        inline SIMDVec_u & load(uint32_t const * p) {
            mVec[0] = _mm256_loadu_si256((__m256i*)p);
            mVec[1] = _mm256_loadu_si256((__m256i*)(p + 8));
            return *this;
        }
        // MLOAD
        inline SIMDVec_u & load(SIMDVecMask<16> const & mask, uint32_t const * p) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)p);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // LOADA
        inline SIMDVec_u & loada(uint32_t const * p) {
            mVec[0] = _mm256_load_si256((__m256i *)p);
            mVec[1] = _mm256_load_si256((__m256i *)(p + 8));
            return *this;
        }
        // MLOADA
        inline SIMDVec_u & loada(SIMDVecMask<16> const & mask, uint32_t const * p) {
            __m256i t0 = _mm256_load_si256((__m256i*)p);
            __m256i t1 = _mm256_load_si256((__m256i*)(p + 8));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // STORE
        inline uint32_t* store(uint32_t * p) const {
            _mm256_storeu_si256((__m256i*)p, mVec[0]);
            _mm256_storeu_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTORE
        inline uint32_t* store(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm256_maskstore_epi32((int*)p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_epi32((int*)(p + 8), mask.mMask[1], mVec[1]);
            return p;
        }
        // STOREA
        inline uint32_t* storea(uint32_t * p) const {
            _mm256_store_si256((__m256i*)p, mVec[0]);
            _mm256_store_si256((__m256i*)(p + 8), mVec[1]);
            return p;
        }
        // MSTOREA
        inline uint32_t* storea(SIMDVecMask<16> const & mask, uint32_t * p) const {
            _mm256_maskstore_epi32((int*)p, mask.mMask[0], mVec[0]);
            _mm256_maskstore_epi32((int*)(p + 8), mask.mMask[1], mVec[1]);
            return p;
        }
        // BLENDV
        // BLENDS
        // SWIZZLE 
        // SWIZZLEA

        // ADDV
        inline SIMDVec_u add(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_u add(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // ADDS
        inline SIMDVec_u add(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = _mm256_set1_epi32(b);
            __m256i t3 = _mm256_add_epi32(mVec[1], t2);
            return SIMDVec_u(t1, t3);
        }
        inline SIMDVec_u operator+ (uint32_t b) const {
            return add(b);
        }
        // MADDS
        inline SIMDVec_u add(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_add_epi32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // ADDVA
        inline SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec[0] = _mm256_add_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_add_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // ADDSA
        inline SIMDVec_u & adda(uint32_t b) {
            mVec[0] = _mm256_add_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_add_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator+= (uint32_t b) {
            return adda(b);
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<16> const & mask, uint32_t b) {
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
        inline SIMDVec_u postinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            mVec[0] = _mm256_add_epi32(mVec[0], t0);
            __m256i t2 = mVec[1];
            mVec[1] = _mm256_add_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        inline SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        inline SIMDVec_u postinc(SIMDVecMask<16> const & mask) {
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
        inline SIMDVec_u & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec[0] = _mm256_add_epi32(mVec[0], t0);
            mVec[1] = _mm256_add_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        inline SIMDVec_u & prefinc(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_add_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // SUBV
        inline SIMDVec_u sub(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        inline SIMDVec_u sub(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // SUBS
        inline SIMDVec_u sub(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator- (uint32_t b) const {
            return sub(b);
        }
        // MSUBS
        inline SIMDVec_u sub(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // SUBVA
        inline SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec[0] = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        inline SIMDVec_u & suba(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBSA
        inline SIMDVec_u & suba(uint32_t b) {
            mVec[0] = _mm256_sub_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_sub_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator-= (uint32_t b) {
            return suba(b);
        }
        // MSUBSA
        inline SIMDVec_u & suba(SIMDVecMask<16> const & mask, uint32_t b) {
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
        inline SIMDVec_u subfrom(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            __m256i t1 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMV
        inline SIMDVec_u subfrom(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(b.mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(b.mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // SUBFROMS
        inline SIMDVec_u subfrom(uint32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[0]);
            __m256i t1 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MSUBFROMS
        inline SIMDVec_u subfrom(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(t0, t1, mask.mMask[0]);
            __m256i t3 = _mm256_sub_epi32(t0, mVec[1]);
            __m256i t4 = _mm256_blendv_epi8(t0, t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // SUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec[0] = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            mVec[1] = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_u & subfroma(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_sub_epi32(b.mVec[0], mVec[0]);
            mVec[0] = _mm256_blendv_epi8(b.mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_sub_epi32(b.mVec[1], mVec[1]);
            mVec[1] = _mm256_blendv_epi8(b.mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_u & subfroma(uint32_t b) {
            mVec[0] = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[0]);
            mVec[1] = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec[1]);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_u subfroma(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_sub_epi32(t0, mVec[0]);
            mVec[0] = _mm256_blendv_epi8(t0, t1, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(t0, mVec[1]);
            mVec[1] = _mm256_blendv_epi8(t0, t2, mask.mMask[1]);
            return *this;
        }
        // POSTDEC
        inline SIMDVec_u postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec[0];
            mVec[0] = _mm256_sub_epi32(mVec[0], t0);
            __m256i t2 = mVec[1];
            mVec[1] = _mm256_sub_epi32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        inline SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        inline SIMDVec_u postdec(SIMDVecMask<16> const & mask) {
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
        inline SIMDVec_u & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec[0] = _mm256_sub_epi32(mVec[0], t0);
            mVec[1] = _mm256_sub_epi32(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        inline SIMDVec_u & prefdec(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = _mm256_sub_epi32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MULV
        inline SIMDVec_u mul(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MULS
        inline SIMDVec_u mul(uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator* (uint32_t b) const {
            return mul(b);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MULVA
        inline SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec[0] = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        inline SIMDVec_u & mula(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MULSA
        inline SIMDVec_u & mula(uint32_t b) {
            mVec[0] = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[1] = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            return *this;
        }
        inline SIMDVec_u & operator*= (uint32_t b) {
            return mula(b);
        }
        // MMULSA
        inline SIMDVec_u & mula(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], _mm256_set1_epi32(b));
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_mullo_epi32(mVec[1], _mm256_set1_epi32(b));
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // DIVV
        inline SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        // DIVS
        inline SIMDVec_u operator/ (uint32_t b) const {
            return div(b);
        }
        // MDIVS
        // DIVVA
        inline SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        // DIVSA
        inline SIMDVec_u & operator/= (uint32_t b) {
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
        inline SIMDVecMask<16> cmpeq(SIMDVec_u const & b) const {
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator==(SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        inline SIMDVecMask<16> cmpeq(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t0);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator== (uint32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        inline SIMDVecMask<16> cmpne(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], b.mVec[0]);
            __m256i m1 = _mm256_xor_si256(m0, t0);
            __m256i m2 = _mm256_cmpeq_epi32(mVec[1], b.mVec[1]);
            __m256i m3 = _mm256_xor_si256(m2, t0);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        inline SIMDVecMask<16> cmpne(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i m1 = _mm256_xor_si256(m0, t1);
            __m256i m2 = _mm256_cmpeq_epi32(mVec[1], t0);
            __m256i m3 = _mm256_xor_si256(m2, t1);
            return SIMDVecMask<16>(m1, m3);
        }
        inline SIMDVecMask<16> operator!= (uint32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        inline SIMDVecMask<16> cmpgt(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(b.mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t1, t2);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_xor_si256(b.mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t3, t4);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        inline SIMDVecMask<16> cmpgt(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_set1_epi32(b ^ 0x80000000);
            __m256i t2 = _mm256_xor_si256(mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t3, t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator> (uint32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        inline SIMDVecMask<16> cmplt(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(b.mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_xor_si256(b.mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t4, t3);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        inline SIMDVecMask<16> cmplt(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0x80000000);
            __m256i t2 = _mm256_set1_epi32(b ^ 0x80000000);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i m0 = _mm256_cmpgt_epi32(t2, t1);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i m1 = _mm256_cmpgt_epi32(t2, t3);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator< (uint32_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        inline SIMDVecMask<16> cmpge(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        inline SIMDVecMask<16> cmpge(uint32_t b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], _mm256_set1_epi32(b));
            __m256i m0 = _mm256_cmpeq_epi32(mVec[0], t0);
            __m256i t1 = _mm256_max_epu32(mVec[1], _mm256_set1_epi32(b));
            __m256i m1 = _mm256_cmpeq_epi32(mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator>= (uint32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        inline SIMDVecMask<16> cmple(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i m0 = _mm256_cmpeq_epi32(b.mVec[0], t0);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            __m256i m1 = _mm256_cmpeq_epi32(b.mVec[1], t1);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        inline SIMDVecMask<16> cmple(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0],t0);
            __m256i m0 = _mm256_cmpeq_epi32(t0, t1);
            __m256i t2 = _mm256_max_epu32(mVec[1], t1);
            __m256i m1 = _mm256_cmpeq_epi32(t0, t2);
            return SIMDVecMask<16>(m0, m1);
        }
        inline SIMDVecMask<16> operator<= (uint32_t b) const {
            return cmple(b);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_u const & b) const {
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
        inline bool cmpe(uint32_t b) const {
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
        inline bool unique() const {
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
        inline uint32_t hadd() const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_add_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            uint32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval;
        }
        // MHADD
        inline uint32_t hadd(SIMDVecMask<16> const & mask) const {
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
        inline uint32_t hadd(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(0);
            __m256i t1 = _mm256_add_epi32(mVec[0], mVec[1]);
            __m256i t2 = _mm256_hadd_epi32(t1, t0);
            __m256i t3 = _mm256_hadd_epi32(t2, t0);
            uint32_t retval = _mm256_extract_epi32(t3, 0);
            retval += _mm256_extract_epi32(t3, 4);
            return retval + b;
        }
        // MHADDS
        inline uint32_t hadd(SIMDVecMask<16> const & mask, uint32_t b) const {
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
        inline uint32_t hmul() const {
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
        inline uint32_t hmul(SIMDVecMask<16> const & mask) const {
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
        inline uint32_t hmul(uint32_t b) const {
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
        inline uint32_t hmul(SIMDVecMask<16> const & mask, uint32_t b) const {
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
        inline SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_add_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFMULADDV
        inline SIMDVec_u fmuladd(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_add_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_add_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }
        // FMULSUBV
        inline SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_sub_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFMULSUBV
        inline SIMDVec_u fmulsub(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_mullo_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_sub_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_mullo_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_sub_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }
        // FADDMULV
        inline SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_mullo_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFADDMULV
        inline SIMDVec_u faddmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_add_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_add_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_mullo_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }
        // FSUBMULV
        inline SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_mullo_epi32(t2, c.mVec[1]);
            return SIMDVec_u(t1, t3);
        }
        // MFSUBMULV
        inline SIMDVec_u fsubmul(SIMDVecMask<16> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            __m256i t0 = _mm256_sub_epi32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_mullo_epi32(t0, c.mVec[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_sub_epi32(mVec[1], b.mVec[1]);
            __m256i t4 = _mm256_mullo_epi32(t3, c.mVec[1]);
            __m256i t5 = _mm256_blendv_epi8(mVec[1], t4, mask.mMask[1]);
            return SIMDVec_u(t2, t5);
        }

        // MAXV
        inline SIMDVec_u max(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMAXV
        inline SIMDVec_u max(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MAXS
        inline SIMDVec_u max(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0], t0);
            __m256i t2 = _mm256_max_epu32(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        // MMAXS
        inline SIMDVec_u max(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_max_epu32(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // MAXVA
        inline SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec[0] = _mm256_max_epu32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_max_epu32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_u & maxa(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_max_epu32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_max_epu32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MAXSA
        inline SIMDVec_u & maxa(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_max_epu32(mVec[0], t0);
            mVec[1] = _mm256_max_epu32(mVec[1], t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_u & maxa(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epu32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_max_epu32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // MINV
        inline SIMDVec_u min(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_min_epu32(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        // MMINV
        inline SIMDVec_u min(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_min_epu32(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_min_epu32(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // MINS
        inline SIMDVec_u min(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec[0], t0);
            __m256i t2 = _mm256_min_epu32(mVec[1], t1);
            return SIMDVec_u(t1, t2);
        }
        // MMINS
        inline SIMDVec_u min(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_min_epu32(mVec[1], t1);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // MINVA
        inline SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec[0] = _mm256_min_epu32(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_min_epu32(mVec[1], b.mVec[1]);
            return *this;
        }
        // MMINVA
        inline SIMDVec_u & mina(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_min_epu32(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_min_epu32(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // MINSA
        inline SIMDVec_u & mina(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_min_epu32(mVec[0], t0);
            mVec[1] = _mm256_min_epu32(mVec[1], t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_u & mina(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epu32(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_min_epu32(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HMAX
        inline uint32_t hmax() const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::min());
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
        inline uint32_t hmax(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::min());
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
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
        inline uint32_t hmin() const {
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
        inline uint32_t hmin(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(std::numeric_limits<uint32_t>::max());
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_blendv_epi8(mVec[1], t0, mask.mMask[1]);
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
        inline SIMDVec_u band(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_and_si256(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        inline SIMDVec_u band(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_and_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // BANDS
        inline SIMDVec_u band(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            __m256i t2 = _mm256_and_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        inline SIMDVec_u operator& (uint32_t b) const {
            return band(b);
        }
        // MBANDS
        inline SIMDVec_u band(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_and_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BANDVA
        inline SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec[0] = _mm256_and_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_and_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        inline SIMDVec_u & banda(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_and_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_and_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BANDSA
        inline SIMDVec_u & banda(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_and_si256(mVec[0], t0);
            mVec[1] = _mm256_and_si256(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        inline SIMDVec_u & banda(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_and_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BORV
        inline SIMDVec_u bor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_or_si256(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        inline SIMDVec_u bor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_or_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // BORS
        inline SIMDVec_u bor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            __m256i t2 = _mm256_or_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        inline SIMDVec_u operator| (uint32_t b) const {
            return bor(b);
        }
        // MBORS
        inline SIMDVec_u bor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_or_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BORVA
        inline SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec[0] = _mm256_or_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_or_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        inline SIMDVec_u & bora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_or_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_or_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BORSA
        inline SIMDVec_u & bora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_or_si256(mVec[0], t0);
            mVec[1] = _mm256_or_si256(mVec[1], t0);
            return *this;
        }
        inline SIMDVec_u & operator|= (uint32_t b) {
            return bora(b);
        }
        // MBORSA
        inline SIMDVec_u & bora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_or_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BXORV
        inline SIMDVec_u bxor(SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            return SIMDVec_u(t0, t1);
        }
        inline SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        inline SIMDVec_u bxor(SIMDVecMask<16> const & mask, SIMDVec_u const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            __m256i t1 = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            __m256i t3 = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return SIMDVec_u(t1, t3);
        }
        // BXORS
        inline SIMDVec_u bxor(uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        inline SIMDVec_u operator^ (uint32_t b) const {
            return bxor(b);
        }
        // MBXORS
        inline SIMDVec_u bxor(SIMDVecMask<16> const & mask, uint32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BXORVA
        inline SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec[0] = _mm256_xor_si256(mVec[0], b.mVec[0]);
            mVec[1] = _mm256_xor_si256(mVec[1], b.mVec[1]);
            return *this;
        }
        inline SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        inline SIMDVec_u & bxora(SIMDVecMask<16> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_xor_si256(mVec[0], b.mVec[0]);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_xor_si256(mVec[1], b.mVec[1]);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // BXORSA
        inline SIMDVec_u & bxora(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec[0] = _mm256_xor_si256(mVec[0], t0);
            __m256i t1 = _mm256_set1_epi32(b);
            mVec[1] = _mm256_xor_si256(mVec[1], t1);
            return *this;
        }
        inline SIMDVec_u & operator^= (uint32_t b) {
            return bxora(b);
        }
        // MBXORSA
        inline SIMDVec_u & bxora(SIMDVecMask<16> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // BNOT
        inline SIMDVec_u bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            return SIMDVec_u(t1, t2);
        }
        inline SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        inline SIMDVec_u bnot(SIMDVecMask<16> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            __m256i t2 = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t3 = _mm256_xor_si256(mVec[1], t0);
            __m256i t4 = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return SIMDVec_u(t2, t4);
        }
        // BNOTA
        inline SIMDVec_u & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec[0] = _mm256_xor_si256(mVec[0], t0);
            mVec[1] = _mm256_xor_si256(mVec[1], t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_u bnota(SIMDVecMask<16> const & mask) {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_xor_si256(mVec[0], t0);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_xor_si256(mVec[1], t0);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t2, mask.mMask[1]);
            return *this;
        }
        // HBAND
        inline uint32_t hband() const {
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
        inline uint32_t hband(SIMDVecMask<16> const & mask) const {
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
        inline uint32_t hband(uint32_t b) const {
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
        inline uint32_t hband(SIMDVecMask<16> const & mask, uint32_t b) const {
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
        inline uint32_t hbor() const {
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
        inline uint32_t hbor(SIMDVecMask<16> const & mask) const {
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
        inline uint32_t hbor(uint32_t b) const {
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
        inline uint32_t hbor(SIMDVecMask<16> const & mask, uint32_t b) const {
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
        inline uint32_t hbxor() const {
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
        inline uint32_t hbxor(SIMDVecMask<16> const & mask) const {
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
        inline uint32_t hbxor(uint32_t b) const {
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
        inline uint32_t hbxor(SIMDVecMask<16> const & mask, uint32_t b) const {
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
        inline SIMDVec_u & gather(uint32_t const * baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            mVec[0] = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            __m256i t1 = _mm256_loadu_si256((__m256i*)(indices + 8));
            mVec[1] = _mm256_i32gather_epi32((const int *)baseAddr, t1, 4);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t const * baseAddr, uint32_t* indices) {
            __m256i t0 = _mm256_loadu_si256((__m256i*)indices);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, t0, 4);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t1, mask.mMask[0]);
            __m256i t2 = _mm256_loadu_si256((__m256i*)(indices + 8));
            __m256i t3 = _mm256_i32gather_epi32((const int *)baseAddr, t2, 4);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t3, mask.mMask[1]);
            return *this;
        }
        // GATHERV
        inline SIMDVec_u & gather(uint32_t const * baseAddr, SIMDVec_u const & indices) {
            mVec[0] = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[0], 4);
            mVec[1] = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[1], 4);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_u & gather(SIMDVecMask<16> const & mask, uint32_t const * baseAddr, SIMDVec_u const & indices) {
            __m256i t0 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[0], 4);
            mVec[0] = _mm256_blendv_epi8(mVec[0], t0, mask.mMask[0]);
            __m256i t1 = _mm256_i32gather_epi32((const int *)baseAddr, indices.mVec[1], 4);
            mVec[1] = _mm256_blendv_epi8(mVec[1], t1, mask.mMask[1]);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint32_t* indices) const {
            alignas(32) uint32_t raw[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            for (int i = 0; i < 16; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, uint32_t* indices) const {
            alignas(32) uint32_t raw[16];
            alignas(32) uint32_t rawMask[16];
            _mm256_store_si256((__m256i*) raw, mVec[0]);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask[0]);
            _mm256_store_si256((__m256i*) (raw + 8), mVec[1]);
            _mm256_store_si256((__m256i*) (rawMask + 8), mask.mMask[1]);
            for (int i = 0; i < 16; i++) { if (rawMask[i] == SIMDVecMask<16>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) const {
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
        inline uint32_t* scatter(SIMDVecMask<16> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) const {
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

        // PACK
        inline SIMDVec_u & pack(SIMDVec_u<uint32_t, 8> const & a, SIMDVec_u<uint32_t, 8> const & b) {
            mVec[0] = a.mVec;
            mVec[1] = b.mVec;
            return *this;
        }
        // PACKLO
        inline SIMDVec_u & packlo(SIMDVec_u<uint32_t, 8> const & a) {
            mVec[0] = a.mVec;
            return *this;
        }
        // PACKHI
        inline SIMDVec_u & packhi(SIMDVec_u<uint32_t, 8> const & b) {
            mVec[1] = b.mVec;
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_u<uint32_t, 8> & a, SIMDVec_u<uint32_t, 8> & b) const {
            a.mVec = mVec[0];
            b.mVec = mVec[1];
        }
        // UNPACKLO
        inline SIMDVec_u<uint32_t, 8> unpacklo() const {
            return SIMDVec_u<uint32_t, 8>(mVec[0]);
        }
        // UNPACKHI
        inline SIMDVec_u<uint32_t, 8> unpackhi() const {
            return SIMDVec_u<uint32_t, 8>(mVec[1]);
        }

        // PROMOTE
        inline operator SIMDVec_u<uint64_t, 16>() const;
        // DEGRADE
        inline operator SIMDVec_u<uint16_t, 16>() const;

        // UTOI
        inline operator SIMDVec_i<int32_t, 16>() const;
        // UTOF
        inline operator SIMDVec_f<float, 16>() const;

    };

}
}

#endif
