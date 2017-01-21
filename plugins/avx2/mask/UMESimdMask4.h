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

#ifndef UME_SIMD_MASK_4_H_
#define UME_SIMD_MASK_4_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<4> :
        public SIMDMaskBaseInterface<
            SIMDVecMask<4>,
            uint32_t,
            4>
    {

        // This function returns internal representation of boolean value based on bool input
#if defined UME_USE_MASK_64B
        static uint64_t TRUE_VAL() { return 0xFFFFFFFFFFFFFFFF; };
        static uint64_t FALSE_VAL() { return 0x0000000000000000; };
        static UME_FORCE_INLINE uint64_t toMaskBool(bool m) {if (m == true) return TRUE_VAL(); else return FALSE_VAL(); }
#else
        static uint32_t TRUE_VAL() { return 0xFFFFFFFF; };
        static uint32_t FALSE_VAL() { return 0x00000000; };
        static UME_FORCE_INLINE uint32_t toMaskBool(bool m) {if (m == true) return TRUE_VAL(); else return FALSE_VAL(); }
#endif

        friend class SIMDVec_u<uint8_t, 4>;
        friend class SIMDVec_u<uint16_t, 4>;
        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_u<uint64_t, 4>;
        friend class SIMDVec_i<int8_t, 4>;
        friend class SIMDVec_i<int16_t, 4>;
        friend class SIMDVec_i<int32_t, 4>;
        friend class SIMDVec_i<int64_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;
    private:

#if defined UME_USE_MASK_64B
        __m256i mMask;
#else
        __m128i mMask;
#endif

#if defined UME_USE_MASK_64B
        UME_FORCE_INLINE explicit SIMDVecMask(__m256i const & x) { mMask = x; };
        // This is usefull to have in 64b mode as the comparison operations might return vector of 32b scalars
        UME_FORCE_INLINE explicit SIMDVecMask(__m128i const & x) {
            mMask = _mm256_cvtepi32_epi64(x);
        };
#else
        UME_FORCE_INLINE explicit SIMDVecMask(__m128i const & x) { mMask = x; };
#endif
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }
        UME_FORCE_INLINE SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVecMask(bool m) {

#if defined UME_USE_MASK_64B
            mMask = _mm256_set1_epi64x(toMaskBool(m));
#else
            mMask = _mm_set1_epi32(toMaskBool(m));
#endif
        }

        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool const *p) {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            for (int i = 0; i < 4; i++) {
                raw[i] = p[i] ? TRUE_VAL() : FALSE_VAL();
            }
            mMask = _mm256_load_si256((__m256i*)raw);
#else
            alignas(16) uint32_t raw[4];
            for (int i = 0; i < 4; i++) {
                raw[i] = p[i] ? TRUE_VAL() : FALSE_VAL();
            }
            mMask = _mm_load_si128((__m128i*)raw);
#endif
        }
        // FULL-CONSTR
        UME_FORCE_INLINE explicit SIMDVecMask(bool m0, bool m1, bool m2, bool m3) {
#if defined UME_USE_MASK_64B
            mMask = _mm256_setr_epi64x(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3));
#else
            mMask = _mm_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3));
#endif
        }

        UME_FORCE_INLINE SIMDVecMask(SIMDVecMask const & mask) {
            this->mMask = mask.mMask;
        }
        // EXTRACT
        UME_FORCE_INLINE bool extract(uint32_t index) const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mMask);
            return raw[index] == TRUE_VAL();
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mMask);
            return raw[index] == TRUE_VAL();
#endif
        }
        UME_FORCE_INLINE bool operator[] (uint32_t index) const {
            return extract(index);
        }
        // INSERT
        UME_FORCE_INLINE void insert(uint32_t index, bool x) {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4] = { 0, 0, 0, 0 };
            _mm256_store_si256((__m256i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm256_load_si256((__m256i*)raw);
#else
            alignas(16) uint32_t raw[4] = { 0, 0, 0, 0 };
            _mm_store_si128((__m128i*)raw, mMask);
            raw[index] = toMaskBool(x);
            mMask = _mm_load_si128((__m128i*)raw);
#endif
        }
        // LOAD
        UME_FORCE_INLINE SIMDVecMask & load(bool const * p) {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            raw[0] = p[0] ? TRUE_VAL() : FALSE_VAL();
            raw[1] = p[1] ? TRUE_VAL() : FALSE_VAL();
            raw[2] = p[2] ? TRUE_VAL() : FALSE_VAL();
            raw[3] = p[3] ? TRUE_VAL() : FALSE_VAL();
            mMask = _mm256_load_si256((__m256i*)raw);
#else
            alignas(16) uint32_t raw[4];
            raw[0] = p[0] ? TRUE_VAL() : FALSE_VAL();
            raw[1] = p[1] ? TRUE_VAL() : FALSE_VAL();
            raw[2] = p[2] ? TRUE_VAL() : FALSE_VAL();
            raw[3] = p[3] ? TRUE_VAL() : FALSE_VAL();
            mMask = _mm_load_si128((__m128i*)raw);
#endif
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVecMask & loada(bool const * p) {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            raw[0] = p[0] ? TRUE_VAL() : FALSE_VAL();
            raw[1] = p[1] ? TRUE_VAL() : FALSE_VAL();
            raw[2] = p[2] ? TRUE_VAL() : FALSE_VAL();
            raw[3] = p[3] ? TRUE_VAL() : FALSE_VAL();
            mMask = _mm256_load_si256((__m256i*)raw);
#else
            alignas(16) uint32_t raw[4];
            raw[0] = p[0] ? TRUE_VAL() : FALSE_VAL();
            raw[1] = p[1] ? TRUE_VAL() : FALSE_VAL();
            raw[2] = p[2] ? TRUE_VAL() : FALSE_VAL();
            raw[3] = p[3] ? TRUE_VAL() : FALSE_VAL();
            mMask = _mm_load_si128((__m128i*)raw);
#endif
            return *this;
        }
        // STORE
        UME_FORCE_INLINE bool* store(bool * p) const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mMask);
            p[0] = raw[0] == TRUE_VAL();
            p[1] = raw[1] == TRUE_VAL();
            p[2] = raw[2] == TRUE_VAL();
            p[3] = raw[3] == TRUE_VAL();
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mMask);
            p[0] = raw[0] == TRUE_VAL();
            p[1] = raw[1] == TRUE_VAL();
            p[2] = raw[2] == TRUE_VAL();
            p[3] = raw[3] == TRUE_VAL();
#endif
            return p;
        }
        // STOREA
        UME_FORCE_INLINE bool* storea(bool * p) const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mMask);
            p[0] = raw[0] == TRUE_VAL();
            p[1] = raw[1] == TRUE_VAL();
            p[2] = raw[2] == TRUE_VAL();
            p[3] = raw[3] == TRUE_VAL();
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mMask);
            p[0] = raw[0] == TRUE_VAL();
            p[1] = raw[1] == TRUE_VAL();
            p[2] = raw[2] == TRUE_VAL();
            p[3] = raw[3] == TRUE_VAL();
#endif
            return p;
        }
        // ASSIGN
        UME_FORCE_INLINE SIMDVecMask & operator= (SIMDVecMask const & x) {
#if defined UME_USE_MASK_64B
            mMask = _mm256_load_si256(&x.mMask);
#else
            mMask = _mm_load_si128(&x.mMask);
#endif
            return *this;
        }
        // LANDV
        UME_FORCE_INLINE SIMDVecMask land(SIMDVecMask const & b) const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_and_si256(mMask, b.mMask);
#else
            __m128i t0 = _mm_and_si128(mMask, b.mMask);
#endif
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (SIMDVecMask const & b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (SIMDVecMask const & b) const {
            return land(b);
        }
        // LANDS
        UME_FORCE_INLINE SIMDVecMask land(bool b) const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(b ? TRUE_VAL() : FALSE_VAL());
            __m256i t1 = _mm256_and_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(b ? TRUE_VAL() : FALSE_VAL());
            __m128i t1 = _mm_and_si128(mMask, t0);
#endif
            return SIMDVecMask(t1);
        }
        UME_FORCE_INLINE SIMDVecMask operator& (bool b) const {
            return land(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator&& (bool b) const {
            return land(b);
        }
        // LANDVA
        UME_FORCE_INLINE SIMDVecMask & landa(SIMDVecMask const & b) {
#if defined UME_USE_MASK_64B
            mMask = _mm256_and_si256(mMask, b.mMask);
#else
            mMask = _mm_and_si128(mMask, b.mMask);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask operator&= (SIMDVecMask const & b) {
            return landa(b);
        }
        // LANDSA
        UME_FORCE_INLINE SIMDVecMask & landa(bool b) {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(b ? TRUE_VAL() : FALSE_VAL());
            mMask = _mm256_and_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(b ? TRUE_VAL() : FALSE_VAL());
            mMask = _mm_and_si128(mMask, t0);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask operator&= (bool b) {
            return landa(b);
        }
        // LORV
        UME_FORCE_INLINE SIMDVecMask lor(SIMDVecMask const & b) const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_or_si256(mMask, b.mMask);
#else
            __m128i t0 = _mm_or_si128(mMask, b.mMask);
#endif
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (SIMDVecMask const & b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (SIMDVecMask const & b) const {
            return lor(b);
        }
        // LORS
        UME_FORCE_INLINE SIMDVecMask lor(bool b) const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(b ? TRUE_VAL() : FALSE_VAL());
            __m256i t1 = _mm256_or_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(b ? TRUE_VAL() : FALSE_VAL());
            __m128i t1 = _mm_or_si128(mMask, t0);
#endif
            return SIMDVecMask(t1);
        }
        UME_FORCE_INLINE SIMDVecMask operator| (bool b) const {
            return lor(b);
        }
        UME_FORCE_INLINE SIMDVecMask operator|| (bool b) const {
            return lor(b);
        }
        // LORVA
        UME_FORCE_INLINE SIMDVecMask & lora(SIMDVecMask const & b) {
#if defined UME_USE_MASK_64B
            mMask = _mm256_or_si256(mMask, b.mMask);
#else
            mMask = _mm_or_si128(mMask, b.mMask);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator|= (SIMDVecMask const & b) {
            return lora(b);
        }
        // LORSA
        UME_FORCE_INLINE SIMDVecMask & lora(bool b) {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(b ? TRUE_VAL() : FALSE_VAL());
            mMask = _mm256_or_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(b ? TRUE_VAL() : FALSE_VAL());
            mMask = _mm_or_si128(mMask, t0);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask & operator |= (bool b) {
            return lora(b);
        }
        // LXORV
        UME_FORCE_INLINE SIMDVecMask lxor(SIMDVecMask const & b) const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_xor_si256(mMask, b.mMask);
#else
            __m128i t0 = _mm_xor_si128(mMask, b.mMask);
#endif
            return SIMDVecMask(t0);
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (SIMDVecMask const & b) const {
            return lxor(b);
        }
        // LXORS
        UME_FORCE_INLINE SIMDVecMask lxor(bool b) const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(b ? TRUE_VAL() : FALSE_VAL());
            __m256i t1 = _mm256_xor_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(b ? TRUE_VAL() : FALSE_VAL());
            __m128i t1 = _mm_xor_si128(mMask, t0);
#endif
            return SIMDVecMask(t1);
        }
        UME_FORCE_INLINE SIMDVecMask operator^ (bool b) const {
            return lxor(b);
        }
        // LXORVA
        UME_FORCE_INLINE SIMDVecMask & lxora(SIMDVecMask const & b) {
#if defined UME_USE_MASK_64B
            mMask = _mm256_xor_si256(mMask, b.mMask);
#else
            mMask = _mm_xor_si128(mMask, b.mMask);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask operator^= (SIMDVecMask const & b) {
            return lxora(b);
        }
        // LXORSA
        UME_FORCE_INLINE SIMDVecMask & lxora(bool b) {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(b ? TRUE_VAL() : FALSE_VAL());
            mMask = _mm256_xor_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(b ? TRUE_VAL() : FALSE_VAL());
            mMask = _mm_xor_si128(mMask, t0);
#endif
            return *this;
        }
        UME_FORCE_INLINE SIMDVecMask operator^= (bool b) {
            return lxora(b);
        }
        // LNOT
        UME_FORCE_INLINE SIMDVecMask lnot() const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_setzero_si256();
            __m256i t1 = _mm256_cmpeq_epi64(mMask, t0);
#else
            __m128i t0 = _mm_setzero_si128();
            __m128i t1 = _mm_cmpeq_epi32(mMask, t0);
#endif
            return SIMDVecMask(t1);
        }
        UME_FORCE_INLINE SIMDVecMask operator! () const {
            return lnot();
        }
        // LNOTA
        UME_FORCE_INLINE SIMDVecMask lnota() {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_setzero_si256();
            mMask = _mm256_cmpeq_epi64(mMask, t0);
#else
            __m128i t0 = _mm_setzero_si128();
            mMask = _mm_cmpeq_epi32(mMask, t0);
#endif
            return *this;
        }
        // HLAND
        UME_FORCE_INLINE bool hland() const {
#if defined UME_USE_MASK_64B
            __m256i t0 = _mm256_set1_epi64x(TRUE_VAL());
            int t1 = _mm256_testc_si256(mMask, t0);
#else
            __m128i t0 = _mm_set1_epi32(TRUE_VAL());
            int t1 = _mm_testc_si128(mMask, t0);
#endif
            return t1 != 0;
        }
        // HLOR
        UME_FORCE_INLINE bool hlor() const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mMask);
            return (raw[0] | raw[1] | raw[2] | raw[3]) != 0;
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mMask);
            return (raw[0] | raw[1] | raw[2] | raw[3]) != 0;
#endif
        }
        // HLXOR
        UME_FORCE_INLINE bool hlxor() const {
#if defined UME_USE_MASK_64B
            alignas(32) uint64_t raw[4];
            _mm256_store_si256((__m256i*)raw, mMask);
            return (raw[0] ^ raw[1] ^ raw[2] ^ raw[3]) != 0;
#else
            alignas(16) uint32_t raw[4];
            _mm_store_si128((__m128i*)raw, mMask);
            return (raw[0] ^ raw[1] ^ raw[2] ^ raw[3]) != 0;
#endif
        }
    };

}
}

#endif
