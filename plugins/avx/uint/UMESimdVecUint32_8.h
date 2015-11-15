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
        public SIMDVecUnsignedInterface<
        SIMDVec_u<uint32_t, 8>,
        uint32_t,
        8,
        SIMDVecMask<8>,
        SIMDVecSwizzle<8 >> ,
        public SIMDVecPackableInterface<
        SIMDVec_u<uint32_t, 8>,
        SIMDVec_u<uint32_t, 4 >>
    {
    public:
        // Conversion operators require access to private members.
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_f<float, 8>;

    private:
        __m256i mVec;

        inline explicit SIMDVec_u(__m256i & x) { this->mVec = x; }
        inline explicit SIMDVec_u(const __m256i & x) { this->mVec = x; }
    public:

        // ZERO-CONSTR
        inline SIMDVec_u() {
            mVec = _mm256_setzero_si256();
        }

        // SET-CONSTR
        inline explicit SIMDVec_u(uint32_t i) {
            mVec = _mm256_set1_epi32(i);
        }

        // LOAD-CONSTR
        inline explicit SIMDVec_u(uint32_t const * p) {
            mVec = _mm256_loadu_si256((__m256i*)p);
        }

        inline SIMDVec_u(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3,
            uint32_t i4, uint32_t i5, uint32_t i6, uint32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        inline uint32_t extract(uint32_t index) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[index];
        }

        // Override Access operators
        inline uint32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_u, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_u, SIMDVecMask<8>>(mask, static_cast<SIMDVec_u &>(*this));
        }

        // insert[] (scalar)
        inline SIMDVec_u & insert(uint32_t index, uint32_t value) {
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
        inline SIMDVec_u add(SIMDVec_u const & b) const {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t3, t4);
            __m256i t6 = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t5), 0x1);
            return SIMDVec_u(t6);
        }

        inline SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        inline SIMDVec_u add(SIMDVecMask<8> const & mask, SIMDVec_u const & b) const {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t5 = _mm256_extractf128_si256(mVec, 1);
            __m128i t6 = _mm_add_epi32(t4, t5);
            __m128i t7 = _mm_blendv_epi8(t5, t6, _mm256_extractf128_si256(mask.mMask, 1));
            __m256i t8 = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t7), 0x1);
            return SIMDVec_u(t8);
        }
        // ADDS
        inline SIMDVec_u add(uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(mVec, 1);
            __m128i t4 = _mm_add_epi32(t0, t3);
            __m256i t5 = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t4), 0x1);
            return SIMDVec_u(t5);
        }
        // MADDS
        inline SIMDVec_u add(SIMDVecMask<8> const & mask, uint32_t b) const {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t0, t4);
            __m128i t6 = _mm_blendv_epi8(t4, t5, _mm256_extractf128_si256(mask.mMask, 1));
            __m256i t7 = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t6), 0x1);
            return SIMDVec_u(t7);
        }
        // ADDVA
        inline SIMDVec_u & adda(SIMDVec_u const & b) {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t3, t4);
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t5), 0x1);
            return *this;
        }
        // MADDVA
        inline SIMDVec_u & adda(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m128i t0 = _mm256_extractf128_si256(b.mVec, 0);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4 = _mm256_extractf128_si256(b.mVec, 1);
            __m128i t5 = _mm256_extractf128_si256(mVec, 1);
            __m128i t6 = _mm_add_epi32(t4, t5);
            __m128i t7 = _mm_blendv_epi8(t5, t6, _mm256_extractf128_si256(mask.mMask, 1));
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t7), 0x1);
            return *this;
        }
        // ADDSA 
        inline SIMDVec_u & adda(uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm256_extractf128_si256(mVec, 1);
            __m128i t4 = _mm_add_epi32(t0, t3);
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t2), (t4), 0x1);
            return *this;
        }
        // MADDSA
        inline SIMDVec_u & adda(SIMDVecMask<8> const & mask, uint32_t b) {
            __m128i t0 = _mm_set1_epi32(b);
            __m128i t1 = _mm256_extractf128_si256(mVec, 0);
            __m128i t2 = _mm_add_epi32(t0, t1);
            __m128i t3 = _mm_blendv_epi8(t1, t2, _mm256_extractf128_si256(mask.mMask, 0));
            __m128i t4 = _mm256_extractf128_si256(mVec, 1);
            __m128i t5 = _mm_add_epi32(t0, t4);
            __m128i t6 = _mm_blendv_epi8(t4, t5, _mm256_extractf128_si256(mask.mMask, 1));
            mVec = _mm256_insertf128_si256(_mm256_castsi128_si256(t3), (t6), 0x1);
            return *this;
        }

        // MULV
        inline SIMDVec_u mul(SIMDVec_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);
            __m128i r_low = _mm_mullo_epi32(a_low, b_low);
            __m128i r_high = _mm_mullo_epi32(a_high, b_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVec_u(ret);
        }
        // MMULV
        inline SIMDVec_u mul(SIMDVecMask<8> const & mask, SIMDVec_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);
            __m128i r_low = _mm_mullo_epi32(a_low, b_low);
            __m128i r_high = _mm_mullo_epi32(a_high, b_high);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVec_u(ret);
        }
        // MULS
        inline SIMDVec_u mul(uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_mullo_epi32(a_low, b_vec);
            __m128i r_high = _mm_mullo_epi32(a_high, b_vec);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVec_u(ret);
        }
        // MMULS
        inline SIMDVec_u mul(SIMDVecMask<8> const & mask, uint32_t b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i r_low = _mm_mullo_epi32(a_low, b_vec);
            __m128i r_high = _mm_mullo_epi32(a_high, b_vec);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            r_low = _mm_blendv_epi8(a_low, r_low, m_low);
            r_high = _mm_blendv_epi8(a_high, r_high, m_high);
            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVec_u(ret);
        }
        // CMPEQV
        inline SIMDVecMask<8> cmpeq(SIMDVec_u const & b) {
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm256_extractf128_si256(b.mVec, 0);
            __m128i b_high = _mm256_extractf128_si256(b.mVec, 1);

            __m128i r_low = _mm_cmpeq_epi32(a_low, b_low);
            __m128i r_high = _mm_cmpeq_epi32(a_high, b_high);

            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecMask<8>(ret);
        }
        // CMPEQS
        inline SIMDVecMask<8> cmpeq(uint32_t b) {
            __m128i b_vec = _mm_set1_epi32(b);
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);

            __m128i r_low = _mm_cmpeq_epi32(a_low, b_vec);
            __m128i r_high = _mm_cmpeq_epi32(a_high, b_vec);

            __m256i ret = _mm256_setzero_si256();
            ret = _mm256_insertf128_si256(ret, r_low, 0);
            ret = _mm256_insertf128_si256(ret, r_high, 1);
            return SIMDVecMask<8>(ret);
        }

        // UNIQUE
        inline bool unique() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            for (unsigned int i = 0; i < 7; i++) {
                for (unsigned int j = i + 1; j < 8; j++) {
                    if (raw[i] == raw[j]) {
                        return false;
                    }
                }
            }
            return true;
        }

        // GATHERS
        inline SIMDVec_u & gather(uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]] };
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_u & gather(SIMDVecMask<8> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8] = { baseAddr[indices[0]], baseAddr[indices[1]], baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]], baseAddr[indices[6]], baseAddr[indices[7]] };
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // GATHERV
        inline SIMDVec_u & gather(uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(32) uint32_t rawInd[8];
            alignas(32) uint32_t raw[8];

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for (int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_u & gather(SIMDVecMask<8> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(32) uint32_t rawInd[8];
            alignas(32) uint32_t raw[8];

            _mm256_store_si256((__m256i*) rawInd, indices.mVec);
            for (int i = 0; i < 8; i++) { raw[i] = baseAddr[rawInd[i]]; }
            __m128i a_low = _mm256_extractf128_si256(mVec, 0);
            __m128i a_high = _mm256_extractf128_si256(mVec, 1);
            __m128i b_low = _mm_load_si128((__m128i*)&raw[0]);
            __m128i b_high = _mm_load_si128((__m128i*)&raw[4]);
            __m128i m_low = _mm256_extractf128_si256(mask.mMask, 0);
            __m128i m_high = _mm256_extractf128_si256(mask.mMask, 1);
            __m128i r_low = _mm_blendv_epi8(a_low, b_low, m_low);
            __m128i r_high = _mm_blendv_epi8(a_high, b_high, m_high);
            mVec = _mm256_insertf128_si256(mVec, r_low, 0);
            mVec = _mm256_insertf128_si256(mVec, r_high, 1);
            return *this;
        }
        // SCATTERS
        inline uint32_t* scatter(uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            for (int i = 0; i < 8; i++) { baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERS
        inline uint32_t* scatter(SIMDVecMask<8> const & mask, uint32_t* baseAddr, uint64_t* indices) {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) { if (rawMask[i] == SIMDVecMask<8>::TRUE()) baseAddr[indices[i]] = raw[i]; };
            return baseAddr;
        }
        // SCATTERV
        inline uint32_t* scatter(uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            for (int i = 0; i < 8; i++) { baseAddr[rawIndices[i]] = raw[i]; };
            return baseAddr;
        }
        // MSCATTERV
        inline uint32_t* scatter(SIMDVecMask<8> const & mask, uint32_t* baseAddr, SIMDVec_u const & indices) {
            alignas(32) uint32_t raw[8];
            alignas(32) uint32_t rawIndices[8];
            alignas(32) uint32_t rawMask[8];
            _mm256_store_si256((__m256i*) raw, mVec);
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            _mm256_store_si256((__m256i*) rawMask, mask.mMask);
            for (int i = 0; i < 8; i++) {
                if (rawMask[i] == SIMDVecMask<8>::TRUE())
                    baseAddr[rawIndices[i]] = raw[i];
            };
            return baseAddr;
        }

        inline  operator SIMDVec_i<int32_t, 8> () const;
    };

}
}

#endif
