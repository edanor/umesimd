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

#ifndef UME_SIMD_MASK_8_H_
#define UME_SIMD_MASK_8_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<8> :
        public SIMDMaskBaseInterface<
            SIMDVecMask<8>,
            uint32_t,
            8>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVec_u<uint8_t, 8>;
        friend class SIMDVec_u<uint16_t, 8>;
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_u<uint64_t, 8>;
        friend class SIMDVec_i<int8_t, 8>;
        friend class SIMDVec_i<int16_t, 8>;
        friend class SIMDVec_i<int32_t, 8>;
        friend class SIMDVec_i<int64_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;
    private:
        __m256i mMask;

        inline explicit SIMDVecMask(__m256i const & x) { mMask = x; };
    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        inline explicit SIMDVecMask(bool m) {
            mMask = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR
        inline explicit SIMDVecMask(bool const *p) {
            alignas(32) uint32_t raw[8];
            for (int i = 0; i < 8; i++) {
                raw[i] = p[i] ? TRUE() : FALSE();
            }
            mMask = _mm256_loadu_si256((__m256i*)raw);
        }
        // FULL-CONSTR
        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3, bool m4, bool m5, bool m6, bool m7) {
            mMask = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            this->mMask = mask.mMask;
        }
        // EXTRACT
        inline bool extract(uint32_t index) const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mMask);
            return raw[index] == TRUE();
        }
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline void insert(uint32_t index, bool x) {
            alignas(32) static uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm256_load_si256((__m256i*)raw);
        }
        // ASSIGNV
        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMask = x.mMask;
            return *this;
        }
        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & b) const {
            __m256 t0 = _mm256_castsi256_ps(mMask);
            __m256 t1 = _mm256_castsi256_ps(b.mMask);
            __m256 t2 = _mm256_and_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            return SIMDVecMask(t3);
        }
        // LANDS
        inline SIMDVecMask land(bool b) const {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256 t1 = _mm256_castsi256_ps(t0);
            __m256 t2 = _mm256_castsi256_ps(mMask);
            __m256 t3 = _mm256_and_ps(t1, t2);
            __m256i t4 = _mm256_castps_si256(t3);
            return SIMDVecMask(t4);
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & b) {
            __m256 t0 = _mm256_castsi256_ps(mMask);
            __m256 t1 = _mm256_castsi256_ps(b.mMask);
            __m256 t2 = _mm256_and_ps(t0, t1);
            mMask = _mm256_castps_si256(t2);
            return *this;
        }
        // LANDSA
        inline SIMDVecMask & landa(bool b) {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256 t1 = _mm256_castsi256_ps(mMask);
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_and_ps(t1, t2);
            mMask = _mm256_castps_si256(t3);
            return *this;
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & b) const {
            __m256 t0 = _mm256_castsi256_ps(mMask);
            __m256 t1 = _mm256_castsi256_ps(b.mMask);
            __m256 t2 = _mm256_or_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            return SIMDVecMask(t3);
        }
        // LORS
        inline SIMDVecMask lor(bool b) const {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256 t1 = _mm256_castsi256_ps(mMask);
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_or_ps(t1, t2);
            __m256i t4 = _mm256_castps_si256(t3);
            return SIMDVecMask(t4);
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & b) {
            __m256 t0 = _mm256_castsi256_ps(mMask);
            __m256 t1 = _mm256_castsi256_ps(b.mMask);
            __m256 t2 = _mm256_or_ps(t0, t1);
            mMask = _mm256_castps_si256(t2);
            return *this;
        }
        // LORSA
        inline SIMDVecMask & lora(bool b) {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256 t1 = _mm256_castsi256_ps(mMask);
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_or_ps(t1, t2);
            mMask = _mm256_castps_si256(t3);
            return *this;
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & b) const {
            __m256 t0 = _mm256_castsi256_ps(mMask);
            __m256 t1 = _mm256_castsi256_ps(b.mMask);
            __m256 t2 = _mm256_xor_ps(t0, t1);
            __m256i t3 = _mm256_castps_si256(t2);
            return SIMDVecMask(t3);
        }
        // LXORS
        inline SIMDVecMask lxor(bool b) const {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256 t1 = _mm256_castsi256_ps(mMask);
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_xor_ps(t1, t2);
            __m256i t4 = _mm256_castps_si256(t3);
            return SIMDVecMask(t4);
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & b) {
            __m256 t0 = _mm256_castsi256_ps(mMask);
            __m256 t1 = _mm256_castsi256_ps(b.mMask);
            __m256 t2 = _mm256_xor_ps(t0, t1);
            mMask = _mm256_castps_si256(t2);
            return *this;
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool b) {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256 t1 = _mm256_castsi256_ps(mMask);
            __m256 t2 = _mm256_castsi256_ps(t0);
            __m256 t3 = _mm256_xor_ps(t1, t2);
            mMask = _mm256_castps_si256(t3);
            return *this;
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __m256i t0 = _mm256_set1_epi32(TRUE());
            __m256  t1 = _mm256_castsi256_ps(t0);
            __m256  t2 = _mm256_castsi256_ps(mMask);
            __m256  t3 = _mm256_xor_ps(t1, t2);
            __m256i t4 = _mm256_castps_si256(t3);
            return SIMDVecMask(t4);
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            __m256i t0 = _mm256_set1_epi32(TRUE());
            __m256  t1 = _mm256_castsi256_ps(t0);
            __m256  t2 = _mm256_castsi256_ps(mMask);
            __m256  t3 = _mm256_xor_ps(t1, t2);
            mMask = _mm256_castps_si256(t3);
            return *this;
        }
        // HLAND
        inline bool hland() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mMask);
            return raw[0] && raw[1] && raw[2] && raw[3] && raw[4] && raw[5] && raw[6] && raw[7];
        }
        // HLOR
        inline bool hlor() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mMask);
            return raw[0] || raw[1] || raw[2] || raw[3] || raw[4] || raw[5] || raw[6] || raw[7];
        }
        // HLXOR
        inline bool hlxor() const {
            alignas(32) uint32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mMask);
            return (raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^ raw[4] ^ raw[5] ^ raw[6] ^ raw[7]) == TRUE();
        }
    };
}
}

#endif
