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

#ifndef UME_SIMD_MASK_16_H_
#define UME_SIMD_MASK_16_H_

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<16> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<16>,
        uint32_t,
        16>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVec_u<uint8_t, 16>;
        friend class SIMDVec_u<uint16_t, 16>;
        friend class SIMDVec_u<uint32_t, 16>;
        friend class SIMDVec_u<uint64_t, 16>;
        friend class SIMDVec_i<int8_t, 16>;
        friend class SIMDVec_i<int16_t, 16>;
        friend class SIMDVec_i<int32_t, 16>;
        friend class SIMDVec_i<int64_t, 16>;
        friend class SIMDVec_f<float, 16>;
        friend class SIMDVec_f<double, 16>;
    private:
        __m256i mMask[2];

        inline SIMDVecMask(__m256i const & x0, __m256i const & x1) { mMask[0] = x0; mMask[1] = x1; };
    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        inline explicit SIMDVecMask(bool m) {
            mMask[0] = _mm256_set1_epi32(toMaskBool(m));
            mMask[1] = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR
        inline explicit SIMDVecMask(bool const *p) {
            alignas(32) uint32_t raw[16];
            for (int i = 0; i < 16; i++) {
                raw[i] = p[i] ? TRUE() : FALSE();
            }
            mMask[0] = _mm256_loadu_si256((__m256i*)raw);
            mMask[1] = _mm256_loadu_si256((__m256i*)(raw + 8));
        }
        // FULL-CONSTR
        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15) {
            mMask[0] = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
            mMask[1] = _mm256_setr_epi32(toMaskBool(m8), toMaskBool(m9),
                toMaskBool(m10), toMaskBool(m11),
                toMaskBool(m12), toMaskBool(m13),
                toMaskBool(m14), toMaskBool(m15));
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            mMask[0] = mask.mMask[0];
            mMask[1] = mask.mMask[1];
        }
        // EXTRACT
        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) uint32_t raw[8];
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMask[0]);
                return raw[index] == TRUE();
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMask[1]);
                return raw[index - 8] == TRUE();
            }
        }
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMask[0]);
                raw[index] = toMaskBool(x);
                mMask[0] = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMask[1]);
                raw[index - 8] = toMaskBool(x);
                mMask[1] = _mm256_load_si256((__m256i*)raw);
            }
        }
        // ASSIGNV
        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMask[0] = x.mMask[0];
            mMask[1] = x.mMask[1];
            return *this;
        }
        // LANDV
        inline SIMDVecMask land(SIMDVecMask const & b) const {
            __m256i t0 = _mm256_and_si256(mMask[0], b.mMask[0]);
            __m256i t1 = _mm256_and_si256(mMask[1], b.mMask[1]);
            return SIMDVecMask(t0, t1);
        }
        inline SIMDVecMask operator& (SIMDVecMask const & b) const {
            return land(b);
        }
        inline SIMDVecMask operator&& (SIMDVecMask const & b) const {
            return land(b);
        }
        // LANDS
        inline SIMDVecMask land(bool b) const {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256i t1 = _mm256_and_si256(mMask[0], t0);
            __m256i t2 = _mm256_and_si256(mMask[1], t0);
            return SIMDVecMask(t1, t2);
        }
        inline SIMDVecMask operator& (bool b) const {
            return land(b);
        }
        inline SIMDVecMask operator&& (bool b) const {
            return land(b);
        }
        // LANDVA
        inline SIMDVecMask & landa(SIMDVecMask const & b) {
            mMask[0] = _mm256_and_si256(mMask[0], b.mMask[0]);
            mMask[1] = _mm256_and_si256(mMask[1], b.mMask[1]);
            return *this;
        }
        inline SIMDVecMask operator&= (SIMDVecMask const & b) {
            return landa(b);
        }
        // LANDSA
        inline SIMDVecMask & landa(bool b) {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            mMask[0] = _mm256_and_si256(mMask[0], t0);
            mMask[1] = _mm256_and_si256(mMask[1], t0);
            return *this;
        }
        inline SIMDVecMask operator&= (bool b) {
            return landa(b);
        }
        // LORV
        inline SIMDVecMask lor(SIMDVecMask const & b) const {
            __m256i t0 = _mm256_or_si256(mMask[0], b.mMask[0]);
            __m256i t1 = _mm256_or_si256(mMask[1], b.mMask[1]);
            return SIMDVecMask(t0, t1);
        }
        inline SIMDVecMask operator| (SIMDVecMask const & b) const {
            return lor(b);
        }
        inline SIMDVecMask operator|| (SIMDVecMask const & b) const {
            return lor(b);
        }
        // LORS
        inline SIMDVecMask lor(bool b) const {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256i t1 = _mm256_or_si256(mMask[0], t0);
            __m256i t2 = _mm256_or_si256(mMask[1], t0);
            return SIMDVecMask(t1, t2);
        }
        inline SIMDVecMask operator| (bool b) const {
            return lor(b);
        }
        inline SIMDVecMask operator|| (bool b) const {
            return lor(b);
        }
        // LORVA
        inline SIMDVecMask & lora(SIMDVecMask const & b) {
            mMask[0] = _mm256_or_si256(mMask[0], b.mMask[0]);
            mMask[1] = _mm256_or_si256(mMask[1], b.mMask[1]);
            return *this;
        }
        inline SIMDVecMask & operator|= (SIMDVecMask const & b) {
            return lora(b);
        }
        // LORSA
        inline SIMDVecMask & lora(bool b) {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            mMask[0] = _mm256_or_si256(mMask[0], t0);
            mMask[1] = _mm256_or_si256(mMask[1], t0);
            return *this;
        }
        inline SIMDVecMask & operator |= (bool b) {
            return lora(b);
        }
        // LXORV
        inline SIMDVecMask lxor(SIMDVecMask const & b) const {
            __m256i t0 = _mm256_xor_si256(mMask[0], b.mMask[0]);
            __m256i t1 = _mm256_xor_si256(mMask[1], b.mMask[1]);
            return SIMDVecMask(t0, t1);
        }
        inline SIMDVecMask operator^ (SIMDVecMask const & b) const {
            return lxor(b);
        }
        // LXORS
        inline SIMDVecMask lxor(bool b) const {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            __m256i t1 = _mm256_xor_si256(mMask[0], t0);
            __m256i t2 = _mm256_xor_si256(mMask[1], t0);
            return SIMDVecMask(t1, t2);
        }
        inline SIMDVecMask operator^ (bool b) const {
            return lxor(b);
        }
        // LXORVA
        inline SIMDVecMask & lxora(SIMDVecMask const & b) {
            mMask[0] = _mm256_xor_si256(mMask[0], b.mMask[0]);
            mMask[1] = _mm256_xor_si256(mMask[1], b.mMask[1]);
            return *this;
        }
        inline SIMDVecMask operator^= (SIMDVecMask const & b) {
            return lxora(b);
        }
        // LXORSA
        inline SIMDVecMask & lxora(bool b) {
            __m256i t0 = _mm256_set1_epi32(b ? TRUE() : FALSE());
            mMask[0] = _mm256_xor_si256(mMask[0], t0);
            mMask[1] = _mm256_xor_si256(mMask[1], t0);
            return *this;
        }
        inline SIMDVecMask operator^= (bool b) {
            return lxora(b);
        }
        // LNOT
        inline SIMDVecMask lnot() const {
            __m256i t0 = _mm256_set1_epi32(TRUE());
            __m256i t1 = _mm256_xor_si256(mMask[0], t0);
            __m256i t2 = _mm256_xor_si256(mMask[1], t0);
            return SIMDVecMask(t1, t2);
        }
        inline SIMDVecMask operator! () const {
            return lnot();
        }
        // LNOTA
        inline SIMDVecMask & lnota() {
            __m256i t0 = _mm256_set1_epi32(TRUE());
            mMask[0] = _mm256_xor_si256(mMask[0], t0);
            mMask[1] = _mm256_xor_si256(mMask[1], t0);
            return *this;
        }
        // HLAND
        inline bool hland() const {
            alignas(32) uint32_t raw[16];
            _mm256_store_si256((__m256i*)raw, mMask[0]);
            _mm256_store_si256((__m256i*)&raw[8], mMask[1]);
            return raw[0] && raw[1] && raw[2] && raw[3] && raw[4] && raw[5] && raw[6] && raw[7]
                && raw[8] && raw[9] && raw[10] && raw[11] && raw[12] && raw[13] && raw[14] && raw[15];
        }
        // HLOR
        inline bool hlor() const {
            int t0 = _mm256_testz_si256(mMask[0], mMask[0]);
            int t1 = _mm256_testz_si256(mMask[1], mMask[1]);
            return (t0 == 0) | (t1 == 0);
        }
    };
}
}

#endif
