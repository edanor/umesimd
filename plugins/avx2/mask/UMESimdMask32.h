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

#ifndef UME_SIMD_MASK_32_H_
#define UME_SIMD_MASK_32_H_

#include "UMESimdMaskPrototype.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVecMask<32> :
        public SIMDMaskBaseInterface<
        SIMDVecMask<32>,
        uint32_t,
        32>
    {
        static uint32_t TRUE() { return 0xFFFFFFFF; };
        static uint32_t FALSE() { return 0x00000000; };

        // This function returns internal representation of boolean value based on bool input
        static inline uint32_t toMaskBool(bool m) { if (m == true) return TRUE(); else return FALSE(); }
        // This function returns a boolean value based on internal representation
        static inline bool toBool(uint32_t m) { if ((m & 0x80000000) != 0) return true; else return false; }

        friend class SIMDVec_u<uint32_t, 32>;
        friend class SIMDVec_i<int32_t, 32>;
        friend class SIMDVec_f<float, 32>;
        friend class SIMDVec_f<double, 32>;
    private:
        __m256i mMaskLoLo;
        __m256i mMaskLoHi;
        __m256i mMaskHiLo;
        __m256i mMaskHiHi;

        inline SIMDVecMask(__m256i const & xLoLo, __m256i const & xLoHi,
            __m256i const & xHiLo, __m256i const & xHiHi) {
            mMaskLoLo = xLoLo;
            mMaskLoHi = xLoHi;
            mMaskHiLo = xHiLo;
            mMaskHiHi = xHiHi;
        };

    public:
        inline SIMDVecMask() {}

        // Regardless of the mask representation, the interface should only allow initialization using 
        // standard bool or using equivalent mask
        inline explicit SIMDVecMask(bool m) {
            mMaskLoLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskLoHi = _mm256_set1_epi32(toMaskBool(m));
            mMaskHiLo = _mm256_set1_epi32(toMaskBool(m));
            mMaskHiHi = _mm256_set1_epi32(toMaskBool(m));
        }

        // LOAD-CONSTR - Construct by loading from memory
        inline explicit SIMDVecMask(bool const *p) {
            alignas(32) uint32_t raw[32];
            for (int i = 0; i < 32; i++) {
                raw[i] = p[i] ? TRUE() : FALSE();
            }
            mMaskLoLo = _mm256_loadu_si256((__m256i*)raw);
            mMaskLoHi = _mm256_loadu_si256((__m256i*)(raw + 8));
            mMaskHiLo = _mm256_loadu_si256((__m256i*)(raw + 16));
            mMaskHiHi = _mm256_loadu_si256((__m256i*)(raw + 24));
        }

        inline SIMDVecMask(bool m0, bool m1, bool m2, bool m3,
            bool m4, bool m5, bool m6, bool m7,
            bool m8, bool m9, bool m10, bool m11,
            bool m12, bool m13, bool m14, bool m15,
            bool m16, bool m17, bool m18, bool m19,
            bool m20, bool m21, bool m22, bool m23,
            bool m24, bool m25, bool m26, bool m27,
            bool m28, bool m29, bool m30, bool m31)
        {
            mMaskLoLo = _mm256_setr_epi32(toMaskBool(m0), toMaskBool(m1),
                toMaskBool(m2), toMaskBool(m3),
                toMaskBool(m4), toMaskBool(m5),
                toMaskBool(m6), toMaskBool(m7));
            mMaskLoHi = _mm256_setr_epi32(toMaskBool(m8), toMaskBool(m9),
                toMaskBool(m10), toMaskBool(m11),
                toMaskBool(m12), toMaskBool(m13),
                toMaskBool(m14), toMaskBool(m15));
            mMaskHiLo = _mm256_setr_epi32(toMaskBool(m16), toMaskBool(m17),
                toMaskBool(m18), toMaskBool(m19),
                toMaskBool(m20), toMaskBool(m21),
                toMaskBool(m22), toMaskBool(m23));
            mMaskHiHi = _mm256_setr_epi32(toMaskBool(m24), toMaskBool(m25),
                toMaskBool(m26), toMaskBool(m27),
                toMaskBool(m28), toMaskBool(m29),
                toMaskBool(m30), toMaskBool(m31));
        }

        inline SIMDVecMask(SIMDVecMask const & mask) {
            mMaskLoLo = mask.mMaskLoLo;
            mMaskLoHi = mask.mMaskLoHi;
            mMaskHiLo = mask.mMaskHiLo;
            mMaskHiHi = mask.mMaskHiHi;
        }

        inline bool extract(uint32_t index) const {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) uint32_t raw[8];

            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLoLo);
                return raw[index] == TRUE();
            }
            else if (index < 16) {
                _mm256_store_si256((__m256i*)raw, mMaskLoHi);
                return raw[index - 8] == TRUE();
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i*)raw, mMaskHiLo);
                return raw[index - 16] == TRUE();
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHiHi);
                return raw[index - 24] == TRUE();
            }
        }

        // A non-modifying element-wise access operator
        inline bool operator[] (uint32_t index) const {
            return extract(index);
        }

        // Element-wise modification operator
        inline void insert(uint32_t index, bool x) {
            UME_PERFORMANCE_UNOPTIMAL_WARNING()
                alignas(32) static uint32_t raw[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            if (index < 8) {
                _mm256_store_si256((__m256i*)raw, mMaskLoLo);
                raw[index] = toMaskBool(x);
                mMaskLoLo = _mm256_load_si256((__m256i*)raw);
            }
            else if (index < 16) {
                _mm256_store_si256((__m256i*)raw, mMaskLoHi);
                raw[index - 8] = toMaskBool(x);
                mMaskLoHi = _mm256_load_si256((__m256i*)raw);
            }
            else if (index < 24) {
                _mm256_store_si256((__m256i*)raw, mMaskHiLo);
                raw[index - 16] = toMaskBool(x);
                mMaskHiLo = _mm256_load_si256((__m256i*)raw);
            }
            else {
                _mm256_store_si256((__m256i*)raw, mMaskHiHi);
                raw[index - 24] = toMaskBool(x);
                mMaskHiHi = _mm256_load_si256((__m256i*)raw);
            }
        }

        inline SIMDVecMask & operator= (SIMDVecMask const & x) {
            mMaskLoLo = x.mMaskLoLo;
            mMaskLoHi = x.mMaskLoHi;
            mMaskHiLo = x.mMaskHiLo;
            mMaskHiHi = x.mMaskHiHi;
            return *this;
        }
    };
}
}

#endif
