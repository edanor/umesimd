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

#ifndef UME_SIMD_VEC_UINT32_8_H_
#define UME_SIMD_VEC_UINT32_8_H_

#include <type_traits>
#include "../../../UMESimdInterface.h"
#include <immintrin.h>

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint32_t, 8> :
        public SIMDVecUnsignedInterface <
        SIMDVec_u<uint32_t, 8>,
        uint32_t,
        8,
        SIMDVecMask<8>,
        SIMDVecSwizzle < 8 >> ,
        public SIMDVecPackableInterface <
        SIMDVec_u<uint32_t, 8>,
        SIMDVec_u < uint32_t, 4 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 8>;

    private:
        __m256i mVec;

        inline explicit SIMDVec_u(__m256i & x) { mVec = x; }
        inline explicit SIMDVec_u(const __m256i & x) { mVec = x; }
    public:
        inline SIMDVec_u() {
            mVec = _mm256_setzero_si256();
        }

        inline explicit SIMDVec_u(uint32_t i) {
            mVec = _mm256_set1_epi32(i);
        }

        inline explicit SIMDVec_u(uint32_t const *p) { this->load(p); }

        inline SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
            uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        inline uint32_t extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING(); // This routine can be optimized
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_u, SIMDVecMask<8>> operator[] (SIMDVecMask<8> & mask) {
            return IntermediateMask<SIMDVec_u, SIMDVecMask<8>>(mask, static_cast<SIMDVec_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING();
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // STOREA
        inline uint32_t * storea(uint32_t * addrAligned) {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }

        // ADDV
        inline SIMDVec_u add(SIMDVec_u const & b) {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }

        inline SIMDVec_u operator+ (SIMDVec_u const & b) {
            return add(b);
        }

        // MADDV
        inline SIMDVec_u add(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u(t0);
        }

        // ADDS
        inline SIMDVec_u add(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }

        // MADDS
        inline SIMDVec_u add(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }

        // ADDVA
        inline SIMDVec_u adda(SIMDVec_u const & b) {
            mVec = _mm256_add_epi32(mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA 
        inline SIMDVec_u & adda(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        inline SIMDVec_u mul(SIMDVec_u const & b) {
            __m256i t0 = _mm256_mul_epi32(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m256i t0 = _mm256_mask_mul_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_u();
        }
        // MULS
        inline SIMDVec_u mul(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mul_epi32(mVec, t0);
            return SIMDVec_u(t1);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<8> const & mask, uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mul_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_u(t1);
        }
        // CMPEQV
        inline SIMDVecMask<8> cmpeq(SIMDVec_u const & b) {
            __mmask8 m0 = _mm256_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(m0);
        }
        // MCMPEQ
        inline SIMDVecMask<8> cmpeq(uint32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            __mmask8 m0 = _mm256_cmpeq_epi32_mask(mVec, t0);
            return SIMDVecMask<8>(m0);
        }

        // GATHERS
        // MGATHERS
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV

        inline  operator SIMDVec_i<int32_t, 8> () const;
    };

}
}

#endif
